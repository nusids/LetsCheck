import argparse
import csv
import json
import logging
import math
import os
import random
import sqlite3
import struct

import faiss
import nltk
import numpy as np
import torch

from models.text_encoder import SentenceTransformer
from models.dense_retriever import DenseRetriever

import gpustat
import pdb

logging.basicConfig(format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s")
logging.getLogger().setLevel(logging.DEBUG)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
BATCH_SIZE = 512
VECTORS_LIMIT = 1600000  # About 60% of OOM limit


def init_db():
    with open('init.sql', 'r') as script:
        db = sqlite3.connect('data/quin-covid.db')
        cur = db.cursor()
        queries = script.read().split(';')
        for q in queries:
            cur.execute(q)


class Quin:
    def __init__(self):
        logging.info('Initializing Quin')
        self.db = sqlite3.connect('data/quin-covid.db')
        logging.debug('Setting up NLP libraries')
        nltk.download('punkt')

        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sent_tokenizer._params.abbrev_types.update(['e.g', 'i.e', 'subsp'])

        self.text_embedding_model = SentenceTransformer('models/weights/encoder/qrbert-multitask', parallel=True)

        logging.info('Initialized!')

    # Methods for encoding and training CORD index
    def load_cord_dataset_to_db(self, num_batches_to_encode=199):
        """Encode snippets from CORD and save them into SQLITE database
        Arguments:
            - num_batches_to_encode: how many batches are snippets are stored in memory before they are encoded
        """
        cur = self.db.cursor()
        snippets_to_encode = []
        cord_uids = []
        snippets_to_encode_limit = BATCH_SIZE * num_batches_to_encode
        logging.info('Start processing data/cord/files/metadata.csv')
        with open('data/cord/files/metadata.csv') as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                add_metadata = True
                add_snippets = True
                # Check if the metadata has been added
                uid = line['cord_uid']
                q = cur.execute('SELECT * FROM cord_metadata WHERE cord_uid = (?)', [uid])
                t = q.fetchone()
                if t is not None:
                    add_metadata = False

                # Check if encoded snippets have been stored for this paper
                q = cur.execute('SELECT * FROM cord WHERE cord_uid = (?)', [uid])
                t = q.fetchone()
                if t is not None:
                    add_snippets = False

                if add_metadata:
                    cur.execute(''' INSERT INTO cord_metadata(
                                        cord_uid,title,authors,journal,url) 
                                    VALUES (?,?,?,?,?)''',
                                (uid, line['title'], line['authors'], line['journal'], line['url']))
                    self.db.commit()

                if add_snippets:
                    # Load full text from raw data
                    found_doc = False
                    path = ''
                    folders = ['document_parses']
                    sha = line['sha']
                    if sha != '':
                        for folder in folders:
                            path = '../data/cord/files/{}/pdf_json/{}.json'.format(folder, sha)
                            if os.path.exists(path):
                                found_doc = True
                                break

                    pmcid = line['pmcid']
                    if not found_doc and pmcid != '':
                        for folder in folders:
                            path = '../data/cord/files/{}/pmc_json/{}.xml.json'.format(folder, pmcid)
                            if os.path.exists(path):
                                found_doc = True
                                break

                    if line['publish_time'] == '' or line['abstract'] == '':
                        continue

                    abstract = line['abstract']
                    paragraphs = []

                    if found_doc:
                        data = json.loads(open(path).read())
                        for p in data['body_text']:
                            paragraphs.append(p['text'])

                    if len(paragraphs) > 0 or abstract != '':
                        # Get new snippets from both abstract and paragraphs
                        new_snippets = self.extract_snippets(abstract, sentences_per_snippet=5)
                        for p in paragraphs:
                            new_snippets += self.extract_snippets(p, sentences_per_snippet=5)

                        snippets_to_encode += new_snippets
                        cord_uids += [uid for _ in new_snippets]

                        if len(snippets_to_encode) > snippets_to_encode_limit:
                            # Encode new snippets and add them to DB
                            self.encode_and_save_cord(snippets_to_encode, cord_uids)
                            snippets_to_encode = []
                            cord_uids = []

        self.encode_and_save_cord(snippets_to_encode, cord_uids)
        cur.close()
        print('\nFinished processing snippets.')

    def add_cord_date_to_db(self):
        """In the previous function, dates were not added to DB
        This function adds date value to all the rows in cord_metadata
        """
        cur = self.db.cursor()
        logging.info('Start processing data/cord/files/metadata.csv')
        print()
        added_count = 0
        with open('data/cord/files/metadata.csv') as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                try:
                    cur.execute('''UPDATE cord_metadata
                                    SET date = (?)
                                    WHERE cord_metadata.cord_uid = (?)
                                    ''',
                                (line['publish_time'], line['cord_uid']))
                    self.db.commit()
                    added_count += 1
                    print(f'\rAdded date to {added_count} CORD entries.', end="")
                except:
                    logging.warn(f'Failed to add date for {line["cord_uid"]}')
        cur.close()
        print('\nFinished adding date to cord entries')

    def encode_and_save_cord(self, snippets, uids):
        cur = self.db.cursor()
        encoded = self.text_embedding_model.encode(snippets, batch_size=BATCH_SIZE)
        for i, snippet in enumerate(snippets):
            cur.execute(''' INSERT INTO cord(cord_uid,snippet,encoded) VALUES (?,?,?)''',
                        (uids[i], snippet, encoded[i]))
        logging.info(f'Added {len(uids)} new snippets.')
        self.db.commit()
        cur.close()

    # Encoding and training news index
    def load_news_to_db(self, num_batches_to_encode=100):
        cur = self.db.cursor()
        snippets_to_encode = []
        snippets_to_encode_limit = BATCH_SIZE * num_batches_to_encode
        logging.info('Start processing data/covid-news.jsonl')

        with open('data/covid-news.jsonl', encoding='utf8') as f:
            for i, line in enumerate(f):
                line = line.rstrip('\n')
                try:
                    document = json.loads(line)
                    print(f'\rReading news article from {document["url"][:64].strip()}', end='')
                    q = cur.execute('SELECT * FROM news WHERE url = (?)', [document['url']])
                    t = q.fetchone()
                    if t is not None:
                        continue
                    if document['text'] is None:
                        continue

                    # Get new snippets from both abstract and paragraphs
                    new_snippets = self.extract_snippets(document['text'], sentences_per_snippet=5)
                    snippets_to_encode += [{
                        'title': document['title'],
                        'snippet': s,
                        'url': document['url'],
                        'date': document['date']
                    } for s in new_snippets]
                    if len(snippets_to_encode) > snippets_to_encode_limit:
                        # Encode new snippets and add them to DB
                        self.encode_and_save_news(snippets_to_encode)
                        snippets_to_encode = []
                except:
                    continue

        self.encode_and_save_news(snippets_to_encode)
        cur.close()
        print('\nFinished processing snippets.')

    def encode_and_save_news(self, snippets):
        cur = self.db.cursor()
        to_encode = [s['snippet'] for s in snippets]
        encoded = self.text_embedding_model.encode(to_encode, batch_size=BATCH_SIZE)
        for i, s in enumerate(snippets):
            cur.execute(''' INSERT INTO news(title,url,date,snippet,encoded) VALUES (?,?,?,?,?)''',
                        (s['title'], s['url'], s['date'], s['snippet'], encoded[i]))
        logging.info(f'Added {len(snippets)} new snippets.')
        self.db.commit()
        cur.close()
        torch.cuda.empty_cache()

    # Util functions
    def train_empty_index(self, data_source):
        self.db.row_factory = sqlite3.Row
        cur = self.db.cursor()
        assert data_source == 'cord' or data_source == 'news'

        vectors_arr = []
        total_vectors = cur.execute(f'SELECT COUNT(*) FROM {data_source}').fetchone()[0]

        skipped = 0
        for r in cur.execute(f'SELECT * FROM {data_source}'):
            if random.random() < VECTORS_LIMIT / total_vectors:
                e = r['encoded']
                v = [np.float32(struct.unpack('f', e[i * 4:(i + 1) * 4])[0]) for i in range(int(len(e) / 4))]
                vectors_arr.append(v)
            else:
                skipped += 1
            print(f"\rAdded {len(vectors_arr)} encoded vectors; skipped {skipped}.", end='')

        dr = DenseRetriever(self.text_embedding_model, use_gpu=True, db_path=None)
        dr.vector_index.vectors = vectors_arr
        dr.vector_index.build(use_gpu=True)
        faiss.write_index(dr.vector_index.index, f'data/{data_source}.index')

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
            i += int(math.ceil(sentences_per_snippet / 2))
        if last_index < len(sentences):
            snippet = ' '.join(sentences[last_index:])
            if len(snippet.split(' ')) > 4:
                snippets.append(snippet)
        return snippets


if __name__ == '__main__':
    ap = argparse.ArgumentParser("")
    ap.add_argument('-m', '--mode', type=str, default='serve', help='index or serve')
    args = ap.parse_args()
    init_db()

    if args.mode == 'index':
        fchecker = Quin()
        fchecker.load_cord_dataset_to_db()
    elif args.mode == 'index-news':
        fchecker = Quin()
        #fchecker.load_news_to_db()
        fchecker.train_empty_index('news')
    elif args.mode == 'index-cord':
        fchecker = Quin()
        #fchecker.load_cord_dataset_to_db()
        #fchecker.add_cord_date_to_db()
        fchecker.train_empty_index('cord')
    else:
        logging.info('Options: index, index-news, index-cord')
