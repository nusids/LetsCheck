import h5py
import logging
import pickle
import sqlite3
import struct

import numpy as np

from models.vector_index import VectorIndex

import random


class DenseRetriever:
    def __init__(self, model, db_path, batch_size=64, use_gpu=False, debug=False):
        self.model = model
        self.vector_index = VectorIndex(768)
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        if db_path:
            self.db = sqlite3.connect(db_path)
            self.db.row_factory = sqlite3.Row
        self.debug = debug

    def load_pretrained_index(self, path):
        self.vector_index.load(path)

    def populate_index(self, table_name):
        cur = self.db.cursor()
        query = f'SELECT * FROM {table_name} ORDER BY idx' if not self.debug \
            else f'SELECT * FROM {table_name} ORDER BY idx LIMIT 10000'
        for r in cur.execute(query):
            e = r['encoded']
            v = [np.float32(struct.unpack('f', e[i*4:(i+1)*4])[0]) for i in range(int(len(e)/4))]
            self.vector_index.index.add(np.ascontiguousarray([v]))
            if self.vector_index.index.ntotal % 100000 == 0:
                print("Added {self.vector_index.index.ntotal} vectors")
        print()
        logging.info("Finished adding vectors.")


    def search(self, queries, limit=1000, probes=512, min_similarity=0):
        query_vectors = self.model.encode(queries, batch_size=self.batch_size)
        ids, similarities = self.vector_index.search(query_vectors, k=limit, probes=probes)
        results = []
        for j in range(len(ids)):
            results.append([
                (ids[j][i], similarities[j][i]) for i in range(len(ids[j])) if similarities[j][i] > min_similarity
            ])
        return results

    '''

    def create_index_from_documents(self, documents):
        logging.info('Building index...')

        self.vector_index.vectors = self.model.encode(documents, batch_size=self.batch_size)
        self.vector_index.build(use_gpu=self.use_gpu)

        logging.info('Built index')

    def create_index_from_vectors(self, vectors_path, mode='pickle'):
        logging.info('Building index...')
        logging.info('Loading vectors...')
        try:
            if mode == 'pickle':
                self.vector_index.vectors = pickle.load(open(vectors_path, 'rb'))
            else:
                self.vector_index.vectors = h5py.File(vectors_path, 'r')['data'][()]
            logging.info('Vectors loaded')
        except Exception as e:
            logging.error(str(e))
            logging.error(f'Failed to load index from {vectors_path}')
            raise e

        self.vector_index.build(use_gpu=self.use_gpu)

        logging.info('Built index')

        def save_index(self, index_path='', vectors_path=''):
        if vectors_path != '':
            self.vector_index.save_vectors(vectors_path)
        if index_path != '':
            self.vector_index.save(index_path)
    '''
