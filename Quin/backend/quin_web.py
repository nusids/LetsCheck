import argparse
import json
import logging
import os
import re
import urllib
import requests
import numpy
import nltk

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
API_URLS = ['http://0.0.0.0/quin']

import urllib.parse as urlparse

from multiprocessing.pool import ThreadPool
from time import sleep
from urllib.parse import parse_qs
from bs4 import BeautifulSoup
from scipy.special import softmax

from web_search import get_html, WebParser, news_search, brave_search, bing_search, duckduckgo_search

from flask import request, Flask, jsonify
from flask_cors import CORS
from scipy import spatial
from models.nli import NLI
from models.qa_ranker import PassageRanker
from models.text_encoder import SentenceTransformer

import pdb

logging.getLogger().setLevel(logging.WARNING)


def is_question(query):
    if re.match(r'^(who|when|what|why|which|whose|is|are|was|were|do|does|did|how) ', query) or query.endswith('?'):
        return True

    pos_tags = nltk.pos_tag(nltk.word_tokenize(query))
    for tag in pos_tags:
        if tag[1].startswith('VB'):
            return False

    return True

def clear_cache():
    for url in API_URLS:
        print(f"Clearing cache from {url}")
        r = requests.post(f'{url}/clear_cache?secret=123')
        print(r.status_code)
        print(r)
 

class Quin:
    def __init__(self):
        nltk.download('punkt')

        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sent_tokenizer._params.abbrev_types.update(['e.g', 'i.e', 'subsp'])

        self.text_embedding_model = SentenceTransformer('models/weights/encoder/qrbert-multitask',
                device='cuda:0', parallel=True)

        self.app = Flask(__name__)
        CORS(self.app)

        self.passage_ranking_model = PassageRanker(
            model_path='models/weights/passage_ranker/passage_ranker.state_dict',
            device='cuda:0',
            gpu=True,
            batch_size=16)
        self.passage_ranking_model.eval()
        self.nli_model = NLI('models/weights/nli/nli.state_dict',
                             batch_size=16,
                             device='cuda:0',
                             parallel=True)
        self.nli_model.eval()

        logging.info('Initialized!')

    def extract_snippets(self, text, sentences_per_snippet=4):
        sentences = self.sent_tokenizer.tokenize(text)
        snippets = []
        i = 0
        last_index = 0
        while i < len(sentences):
            snippet = ' '.join(sentences[i:i + sentences_per_snippet])
            if len(snippet.split(' ')) > 4:
                snippets.append(snippet)
            last_index = i + sentences_per_snippet
            i += sentences_per_snippet
        if last_index < len(sentences):
            snippet = ' '.join(sentences[last_index:])
            if len(snippet.split(' ')) > 4:
                snippets.append(snippet)
        return snippets

    def search_web_evidence(self, query, limit=10, no_class=False, subqueries=[], nli_confidence=0.4):
        logging.info('searching the web...')
        #pdb.set_trace()
        #urls = brave_search(query, pages=1)[:limit]
        #urls = bing_search(query, pages=1)[:limit]
        urls = duckduckgo_search(query, pages=1)[:limit]
        #urls = news_search(query, pages=3)[:limit]

        logging.info('downloading {} web pages...'.format(len(urls)))
        search_results = []

        def download(url):
            nonlocal search_results
            data = get_html(url)
            soup = BeautifulSoup(data, features='lxml')
            title = soup.title.string
            w = WebParser()
            w.feed(data)
            full_text = '\n'.join([b for b in w.blocks if b.count(' ') > 20])
            new_snippets = sum([self.extract_snippets(b) for b in w.blocks if b.count(' ') > 20], [])
            new_snippets = [{'snippet': p, 'url': url, 'title': title, 'full_text': full_text} for p in new_snippets]
            search_results += new_snippets

        def timeout_download(arg):
            pool = ThreadPool(1)
            try:
                pool.apply_async(download, [arg]).get(timeout=2)
            except:
                pass
            pool.close()
            pool.join()

        p = ThreadPool(32)
        p.map(timeout_download, urls)
        p.close()
        p.join()

        query_is_question = is_question(query)

        # re-ranking
        if query_is_question:
            logging.info('re-ranking...')
            snippets = [s['snippet'] for s in search_results]
            qa_pairs = [(query, snippet) for snippet in snippets]
            _, probs = self.passage_ranking_model(qa_pairs)
            probs = [softmax(p)[1] for p in probs]
            filtered_results = []
            for i in range(len(search_results)):
                if probs[i] > 0.35:
                    search_results[i]['score'] = str(probs[i])
                    filtered_results.append(search_results[i])
            search_results = filtered_results
            search_results = sorted(search_results, key=lambda x: float(x['score']), reverse=True)

        # highlight most relevant sentences
        logging.info('highlighting...')
        results_sentences = []
        sentences_texts = []
        sentences_vectors = {}
        for i, r in enumerate(search_results):
            sentences = self.sent_tokenizer.tokenize(r['snippet'])
            sentences = [s for s in sentences if len(s.split(' ')) > 4]
            sentences_texts.extend(sentences)
            results_sentences.append(sentences)

        vectors = self.text_embedding_model.encode(sentences=sentences_texts, batch_size=128)
        for i, v in enumerate(vectors):
            sentences_vectors[sentences_texts[i]] = v

        # Do the following steps for subqueries if they are given
        q_arr = subqueries[:]
        re = [{'sub_qn_id': i, 'subquery': q, 'results': search_results[:]} for i, q in enumerate(subqueries)]
        if len(q_arr) == 0:
            # No subqueries
            q_arr.append(query)
            re = [{'results': search_results[:]}]
        for q_id, q in enumerate(q_arr):
            query_vector = self.text_embedding_model.encode(sentences=[q], batch_size=1)[0]
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
                    re[q_id]['results'][i]['evidence'] = ' '.join(evidence_sentences)
                re[q_id]['results'][i]['snippet'] = \
                    ' '.join([s if s not in best_sentences else '<b>{}</b>'.format(s) for s in sentences])

            re[q_id]['results'] = [s for s in re[q_id]['results'] if 'evidence' in s]

            # fact verification
            if not (query_is_question or no_class):
                logging.info('entailment classification...')
                es_pairs = []
                for result in re[q_id]['results']:
                    evidence = result['evidence']
                    es_pairs.append((evidence, q))
                try:
                    labels, probs = self.nli_model(es_pairs)
                    logging.info(str(labels))
                    for i in range(len(labels)):
                        confidence = numpy.exp(numpy.max(probs[i]))
                        if confidence > nli_confidence:
                            re[q_id]['results'][i]['nli_class'] = labels[i]
                        else:
                            re[q_id]['results'][i]['nli_class'] = 'neutral'
                        re[q_id]['results'][i]['nli_confidence'] = str(confidence)
                except Exception as e:
                    logging.warning('Error doing entailment classification')
                    logging.warning(str(e))
            filtered_results = []
            added_urls = set()
            veracity = ''
            supporting = 0
            refuting = 0
            for r in re[q_id]['results']:
                if r['url'] not in added_urls:
                    filtered_results.append(r)
                    added_urls.add(r['url'])
                    if 'nli_class' in r:
                        if r['nli_class'] == 'entailment':
                            supporting += 1
                        elif r['nli_class'] == 'contradiction':
                            refuting += 1
            re[q_id]['results'] = filtered_results

            if supporting > 2 or refuting > 2:
                if supporting / (refuting + 0.001) > 1.7:
                    veracity = 'Probably True'
                elif refuting / (supporting + 0.001) > 1.7:
                    veracity = 'Probably False'
                else:
                    veracity = '? Ambiguous'
            else:
                veracity = 'Not enough evidence'
            
            try:
                if veracity != 'Not enough evidence' and veracity != '':
                    re[q_id]['results'] = [s for s in re[q_id]['results'] if s['nli_class'] != 'neutral']
            except:
                pass


            re[q_id]['results'] = re[q_id]['results'][:limit]
            if query_is_question:
                re[q_id]['type'] = 'question'
            elif no_class:
                re[q_id]['type'] = 'statement'
            else:
                re[q_id] = {**re[q_id],
                        'type': 'statement',
                        'supporting': supporting,
                        'refuting': refuting,
                        'veracity_rating': veracity}

        logging.info('done searching')
        if len(subqueries) == 0:
            return re[0]
        else:
            return re
            

    def serve_middleware(self):
        while True:
            sleep(1)
            for API_URL in API_URLS:
                logging.info(f'Checking {API_URL}')
                try:
                    res = requests.post('{}/get_pending_requests?secret=123'.format(API_URL), data={}, timeout=10)
                    res = json.loads(res.text)
                    pending_requests = res['data']
                    for req in pending_requests:
                        if 'api2' not in req:
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
                                no_class = False
                                nli_confidence = 0.4
                                if 'noclass' in parameters:
                                    no_class = True
                                subqueries = []
                                if 'subqueries' in parameters:
                                    subqueries = urllib.parse.unquote_plus(parameters['subqueries'][0]).split(';')
                                if 'nli_confidence' in parameters:
                                    nli_confidence = float(urllib.parse.unquote_plus(parameters['nli_confidence'][0]))
                               
                                e = self.search_web_evidence(query=query, limit=20, no_class=no_class, subqueries=subqueries, nli_confidence=nli_confidence)
                                results = json.dumps(e, indent=4)

                                r = requests.post(
                                    '{}/save_cache?secret=123&request_url={}'.format(API_URL, encoded_req),
                                    data={'data': results},
                                    timeout=10)

                                if r.status_code != 200:
                                    logging.debug(str(r))
                                    logging.info('Error saving cache')

                            if r.status_code != 200:
                                logging.info('Error removing pending request')
                        except Exception:
                            logging.exception('Processing error!')
                except json.decoder.JSONDecodeError as e:
                    logging.exception(e)
                    logging.warn(f'res: {res.text}')
                except Exception as e:
                    logging.warn(e)
                    logging.exception('Waiting 5s to send API call to the frontend...')
                    sleep(5)


if __name__ == '__main__':
    ap = argparse.ArgumentParser("")
    ap.add_argument('-m', '--mode', type=str, default='serve', help='serve/clear')
    args = ap.parse_args()

    if args.mode == 'clear':
        clear_cache()
    else:
        fchecker = Quin()
        fchecker.serve_middleware()
