import json
import logging
import math
import os
import re
import time
import urllib
import requests
import sqlite3
from time import sleep
import torch
import urllib.parse as urlparse
from urllib.parse import parse_qs

import nltk
import numpy as np
from scipy.special import softmax
from dateutil import parser
from flask import Flask
from flask_cors import CORS
from scipy import spatial

from models.nli import NLI
from models.dense_retriever import DenseRetriever
from models.qa_ranker import PassageRanker
from models.text_encoder import SentenceTransformer

import pdb

logging.basicConfig(format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s")
logging.getLogger().setLevel(logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
API_URLS = ['http://0.0.0.0/quin']
BATCH_SIZE = 16
DEBUG_MODE = False


def is_question(query):
    if re.match(r'^(who|when|what|why|is|are|was|were|do|does|did|how) ', query) or query.endswith('?'):
        return True
    pos_tags = nltk.pos_tag(nltk.word_tokenize(query))
    for tag in pos_tags:
        if tag[1].startswith('VB'):
            return False
    return True


class Quin:
    def __init__(self, mode='serve'):
        logging.info('Initializing Quin')
        logging.debug('Setting up NLP libraries')
        nltk.download('punkt')

        self.db = sqlite3.connect('data/quin-covid.db')
        self.db.row_factory = sqlite3.Row
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sent_tokenizer._params.abbrev_types.update(['e.g', 'i.e', 'subsp'])

        self.text_embedding_model = SentenceTransformer('models/weights/encoder/qrbert-multitask', parallel=False,
                                                        device='cuda:0')

        if mode == 'serve':
            logging.debug('Initializing for web serving mode')
            self.app = Flask(__name__)
            CORS(self.app)

            logging.debug('Loading PassageRanker')
            self.passage_ranking_model = PassageRanker(
                model_path='models/weights/passage_ranker/passage_ranker.state_dict',
                gpu=True,
                batch_size=16)
            self.passage_ranking_model.eval()
            torch.cuda.empty_cache()
            logging.debug('Loading NLI model')
            self.nli_model = NLI('models/weights/nli/nli.state_dict', batch_size=16, parallel=True, device='cuda:0')
            self.nli_model.eval()

            torch.cuda.empty_cache()

            logging.debug('Loading dense index for CORD')
            if os.path.exists('data/cord.index'):
                self.cord_dense_index = DenseRetriever(model=self.text_embedding_model, db_path='data/quin-covid.db',
                                                       use_gpu=True, debug=DEBUG_MODE)
                self.cord_dense_index.load_pretrained_index('data/cord.index')
                self.cord_dense_index.populate_index('cord')
            torch.cuda.empty_cache()

            logging.debug('Loading dense index for news')
            if os.path.exists('data/news.index'):
                self.news_dense_index = DenseRetriever(model=self.text_embedding_model, db_path='data/quin-covid.db',
                                                       use_gpu=True, debug=DEBUG_MODE)
                self.news_dense_index.load_pretrained_index('data/news.index')
                self.news_dense_index.populate_index('news')
            torch.cuda.empty_cache()

        logging.info('Initialized!')

    def get_cord_metadata(self, cord_uid):
        cur = self.db.cursor()
        r = cur.execute('SELECT title, authors, journal, url, date FROM cord_metadata WHERE cord_uid=(?)', [cord_uid])
        row = r.fetchone()
        metadata = {
            'title': row['title'],
            'authors': row['authors'],
            'journal': row['journal'],
            'url': row['url'],
            'date': row['date']
        }
        return metadata

    def get_cord_document(self, rowid):
        cur = self.db.cursor()
        rowid = int(rowid)
        r = cur.execute('SELECT cord_uid,snippet FROM cord WHERE rowid=(?)', [rowid + 1])
        return r.fetchone()

    def get_news(self, rowid):
        cur = self.db.cursor()
        rowid = int(rowid)
        r = cur.execute('SELECT title,url,date,snippet FROM news WHERE rowid=(?)', [rowid + 1])
        return r.fetchone()

    def search_evidence(self, query, limit=20, unique_docs=True, source='cord', highlight=True):
        if source == 'cord':
            dense_index = self.cord_dense_index
        else:
            dense_index = self.news_dense_index

        search_results = {}
        logging.info('Running dense retriever for: {}'.format(query))
        dense_results = dense_index.search([query], limit=limit, min_similarity=-9999)[0]
        dense_results = [r[0] for r in dense_results]
        results = list(set(dense_results))

        if len(results) > 0:
            for i in range(len(results)):
                row_id = results[i]
                if source == 'cord':
                    try:
                        d = self.get_cord_document(row_id)
                        result = {
                            'cord_uid': d['cord_uid'],
                            'snippet': d['snippet'],
                            'evidence': d['snippet']
                        }
                    except Exception as ex:
                        logging.warning(f'Encountered problem in retrieving cord document at row {row_id}')
                        logging.warning(ex)
                        continue
                else:
                    try:
                        d = self.get_news(row_id)
                        result = {
                            'url': d['url'],
                            'title': d['title'],
                            'date': d['date'],
                            'snippet': d['snippet'],
                            'evidence': d['snippet']
                        }
                    except Exception as ex:
                        logging.warning(f'Encountered problem in retrieving news at row {row_id}')
                        logging.warning(ex)
                        continue
                search_results[row_id] = result

        search_results = list(search_results.values())

        if unique_docs or source == 'news':
            added = set()
            filtered_results = []
            for r in search_results:
                if source == 'cord':
                    if r['cord_uid'] not in added:
                        filtered_results.append(r)
                        added.add(r['cord_uid'])
                else:
                    if r['snippet'] not in added:
                        filtered_results.append(r)
                        added.add(r['snippet'])
            search_results = filtered_results
        query_is_question = is_question(query)

        # re-ranking
        logging.info(f'{len(search_results)} results found.')
        snippets = [s['snippet'] for s in search_results]
        qa_pairs = [(query, snippet) for snippet in snippets]
        _, probs = self.passage_ranking_model(qa_pairs)
        probs = [softmax(p)[1] for p in probs]
        filtered_results = []
        k = 10
        top_k_probs = np.flip(np.argsort(probs)[-k:])
        for i in top_k_probs:
            search_results[i]['score'] = probs[i]
            filtered_results.append(search_results[i])
        search_results = filtered_results

        # time-aware score / limit results
        if source == 'cord':
            for i, result in enumerate(search_results):
                search_results[i].update(self.get_cord_metadata(search_results[i]['cord_uid']))
        current_time = time.time()
        for r in search_results:
            try:
                t = parser.parse(r['date']).timestamp()
                r['timeaware_score'] = r['score'] * math.pow(2, -(current_time - t) / (365 * 86400))
            except Exception as ex:
                logging.warning('Encountered problem in calculating timeaware score')
                logging.warning(ex)
                r['timeaware_score'] = 0
        search_results = sorted(search_results, key=lambda x: x['timeaware_score'], reverse=True)
        search_results = search_results[:limit]
        for i, result in enumerate(search_results):
            search_results[i]['score'] = str(search_results[i]['score'])
            search_results[i]['timeaware_score'] = str(search_results[i]['timeaware_score'])

        if highlight:
            # highlight most relevant sentences
            results_sentences = []
            sentences_texts = []
            sentences_vectors = {}
            for i, r in enumerate(search_results):
                sentences = self.sent_tokenizer.tokenize(r['snippet'])
                sentences = [s for s in sentences if len(s.split(' ')) > 4]
                sentences_texts.extend(sentences)
                results_sentences.append(sentences)

            vectors = self.text_embedding_model.encode(sentences=sentences_texts, batch_size=64)
            for i, v in enumerate(vectors):
                sentences_vectors[sentences_texts[i]] = v

            query_vector = self.text_embedding_model.encode(sentences=[query], batch_size=1)[0]
            for i, sentences in enumerate(results_sentences):
                best_sentences = set()
                evidence_sentences = []
                for sentence in sentences:
                    sentence_vector = sentences_vectors[sentence]
                    score = 1 - spatial.distance.cosine(query_vector, sentence_vector)
                    if score > 0.91:
                        best_sentences.add(sentence)
                        evidence_sentences.append(sentence)
                if len(evidence_sentences) > 0:
                    search_results[i]['evidence'] = ' '.join(evidence_sentences)
                search_results[i]['snippet'] = \
                    ' '.join([s if s not in best_sentences else '<b>{}</b>'.format(s) for s in sentences])

        # fact verification
        supporting = 0
        refuting = 0
        if not query_is_question:
            es_pairs = []
            for result in search_results:
                evidence = result['evidence']
                es_pairs.append((evidence, query))
            try:
                labels, probs = self.nli_model(es_pairs)
                for i in range(len(labels)):
                    confidence = np.exp(np.max(probs[i]))
                    if confidence > 0.4:
                        search_results[i]['nli_class'] = labels[i]
                        if labels[i] == 'entailment':
                            supporting += 1
                        elif labels[i] == 'contradiction':
                            refuting += 1
                    else:
                        search_results[i]['nli_class'] = 'neutral'
                    search_results[i]['nli_confidence'] = str(confidence)
            except Exception as ex:
                logging.warning(ex)
                pass

        logging.info(f'{supporting} supporting, {refuting} refuting')
        if supporting > 2 or refuting > 2:
            if supporting / (refuting + 0.001) > 1.7:
                veracity = 'Probably True'
            elif refuting / (supporting + 0.001) > 1.7:
                veracity = 'Probably False'
            else:
                veracity = '? Ambiguous'
        else:
            veracity = 'Not enough evidence'

        if query_is_question:
            return {'type': 'question',
                    'results': search_results}
        else:
            return {'type': 'statement',
                    'supporting': supporting,
                    'refuting': refuting,
                    'veracity_rating': veracity,
                    'results': search_results}

    def serve_middleware(self):
        while True:
            sleep(1)
            restart_flag = open('restart_flag', 'r').readline().strip()
            if restart_flag == '1':
                w = open('restart_flag', 'w')
                w.write('0')
                w.close()
                logging.info("Restart flag is set; restarting...")
                return

            for API_URL in API_URLS:
                try:
                    res = requests.post('{}/get_pending_requests?secret=123'.format(API_URL), data={}, timeout=10)
                    res = json.loads(res.text)
                    pending_requests = res['data']
                    for req in pending_requests:
                        if 'api2' in req:
                            continue
                        try:
                            logging.info('Processing request {}'.format(req))
                            parsed = urlparse.urlparse(req)
                            parameters = parse_qs(parsed.query)

                            encoded_req = urllib.parse.quote(req.encode("utf-8"))

                            r = requests.post('{}/remove_pending_request?secret=123&request_url={}'.format(API_URL,
                                                                                                           encoded_req),
                                              timeout=10)

                            if 'query' in parameters:
                                query = urllib.parse.unquote_plus(parameters['query'][0])
                                source = parameters['source'][0]
                                results = json.dumps(self.search_evidence(query=query,
                                                                          limit=100,
                                                                          source=source), indent=4)

                                r = requests.post(
                                    '{}/save_cache?secret=123&request_url={}'.format(API_URL, encoded_req),
                                    data={'data': results},
                                    timeout=10)

                                if r.status_code != 200:
                                    logging.info('Error saving cache')

                            if r.status_code != 200:
                                logging.info('Error removing pending request')
                        except Exception:
                            logging.exception('Processing error!')
                            pdb.set_trace()
                except Exception:
                    logging.exception('Waiting 5s to send API call to the frontend...')
                    sleep(5)


if __name__ == '__main__':
    while True:
        try:
            fchecker = Quin(mode='serve')
            fchecker.serve_middleware()
        except Exception as e:
            logging.warning(e)
            logging.info("Encountered exception, restarting...")
            logging.info("====================================")
