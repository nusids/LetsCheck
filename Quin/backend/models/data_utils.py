import csv
import json
import pickle

import logging
import re
from pprint import pprint

import pandas
import gzip
import os

import numpy as np

from random import randint, random
from tqdm import tqdm
from typing import List, Dict

from spacy.lang.en import English
from tensorflow.keras.utils import Sequence
from transformers import RobertaTokenizer, BertTokenizer, BertModel, AutoTokenizer, AutoModel

# from utils import WebParser
from models.dense_retriever import DenseRetriever
from models.tokenization import tokenize

from typing import Union, List

from unidecode import unidecode

import spacy


# from utils import WebParser


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(self, guid: str, texts: List[str], label: Union[int, float]):
        """
        Creates one InputExample with the given texts, guid and label
        str.strip() is called on both texts.
        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label

    def get_texts(self):
        return self.texts

    def get_label(self):
        return self.label


class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_examples(filename, max_examples=0):
    examples = []
    id = 0
    with open(filename, encoding='utf8') as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            sample = json.loads(line)
            label = sample['label']
            guid = "%s-%d" % (filename, id)
            id += 1
            if label == 'entailment':
                label = 0
            elif label == 'contradiction':
                label = 1
            else:
                label = 2
            examples.append(InputExample(guid=guid,
                                         texts=[sample['s1'], sample['s2']],
                                         label=label))
            if 0 < max_examples <= len(examples):
                break
    return examples


def get_qa_examples(filename, max_examples=0, dev=False):
    examples = []
    id = 0
    with open(filename, encoding='utf8') as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            sample = json.loads(line)
            label = sample['relevant']
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid,
                                         texts=[sample['question'], sample['answer']],
                                         label=label))
            if not dev:
                if label == 1:
                    for _ in range(13):
                        examples.append(InputExample(guid=guid,
                                                     texts=[sample['question'], sample['answer']],
                                                     label=label))
            if 0 < max_examples <= len(examples):
                break
    return examples


def map_label(label):
    labels = {"relevant": 0, "irrelevant": 1}
    return labels[label.strip().lower()]


def get_qar_examples(filename, max_examples=0):
    examples = []
    id = 0
    with open(filename, encoding='utf8') as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            sample = json.loads(line)
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid,
                                         texts=[sample['question'], sample['answer']],
                                         label=1.0))
            if 0 < max_examples <= len(examples):
                break
    return examples


def get_qar_artificial_examples():
    examples = []
    id = 0

    print('Loading passages...')

    passages = []
    file = open('data/msmarco/collection.tsv', 'r', encoding='utf8')
    while True:
        line = file.readline()
        if not line:
            break
        line = line.rstrip('\n').split('\t')
        passages.append(line[1])

    print('Loaded passages')

    with open('data/qar/qar_artificial_queries.csv') as f:
        for i, line in enumerate(f):
            queries = line.rstrip('\n').split('|')
            for query in queries:
                guid = "%s-%d" % ('', id)
                id += 1
                examples.append(InputExample(guid=guid,
                                             texts=[query, passages[i]],
                                             label=1.0))
    return examples


def get_single_examples(filename, max_examples=0):
    examples = []
    id = 0
    with open(filename, encoding='utf8') as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            sample = json.loads(line)
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid,
                                         texts=[sample['text']],
                                         label=1))
            if 0 < max_examples <= len(examples):
                break
    return examples


def get_qnli_examples(filename, max_examples=0, no_contradictions=False, fever_only=False):
    examples = []
    id = 0
    with open(filename, encoding='utf8') as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            sample = json.loads(line)
            label = sample['label']
            if label == 'contradiction' and no_contradictions:
                continue
            if sample['evidence'] == '':
                continue
            if fever_only and sample['source'] != 'fever':
                continue
            guid = "%s-%d" % (filename, id)
            id += 1

            examples.append(InputExample(guid=guid,
                                         texts=[sample['statement'].strip(), sample['evidence'].strip()],
                                         label=1.0))
            if 0 < max_examples <= len(examples):
                break
    return examples


def get_retrieval_examples(filename, negative_corpus='data/msmarco/collection.tsv', max_examples=0, no_statements=True,
                           encoder_model=None, negative_samples_num=4):
    examples = []
    queries = []
    passages = []
    negative_passages = []
    id = 0
    with open(filename, encoding='utf8') as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            sample = json.loads(line)

            if 'evidence' in sample and sample['evidence'] == '':
                continue

            guid = "%s-%d" % (filename, id)
            id += 1

            if sample['type'] == 'question':
                query = sample['question']
                passage = sample['answer']
            else:
                query = sample['statement']
                passage = sample['evidence']

            query = query.strip()
            passage = passage.strip()

            if sample['type'] == 'statement' and no_statements:
                continue

            queries.append(query)
            passages.append(passage)

            if sample['source'] == 'natural-questions':
                negative_passages.append(passage)

            if max_examples == len(passages):
                break

    if encoder_model is not None:
        # Load MSMARCO passages
        logging.info('Loading MSM passages...')
        with open(negative_corpus) as file:
            for line in file:
                p = line.rstrip('\n').split('\t')[1]
                negative_passages.append(p)

        logging.info('Building ANN index...')
        dense_retriever = DenseRetriever(model=encoder_model, batch_size=1024, use_gpu=True)
        dense_retriever.create_index_from_documents(negative_passages)
        results = dense_retriever.search(queries=queries, limit=100, probes=256)
        negative_samples = [
            [negative_passages[p[0]] for p in r if negative_passages[p[0]] != passages[i]][:negative_samples_num]
            for i, r in enumerate(results)
        ]
        # print(queries[0])
        # print(negative_samples[0][0])

        for i in range(len(queries)):
            texts = [queries[i], passages[i]] + negative_samples[i]
            examples.append(InputExample(guid=guid,
                                         texts=texts,
                                         label=1.0))

    else:
        for i in range(len(queries)):
            texts = [queries[i], passages[i]]
            examples.append(InputExample(guid=guid,
                                         texts=texts,
                                         label=1.0))

    return examples


def get_ict_examples(filename, max_examples=0):
    examples = []
    id = 0
    with open(filename, encoding='utf8') as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            sample = json.loads(line)
            # label = sample['label']
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid,
                                         texts=[sample['s1'].strip(), sample['s2'].strip()],
                                         label=1.0))
            if 0 < max_examples <= len(examples):
                break
    return examples


def preprocess_fever_dataset(path):
    def replace_symbols(text):
        return text.replace('-LRB-', '(') \
            .replace('-RRB-', ')') \
            .replace('-LSB-', '[') \
            .replace('-RSB-', ']'). \
            replace('-LCB-', '{'). \
            replace('-RCB-', '}')

    print('Loading wiki articles...')
    articles = {}
    for i in range(109):
        wid = str(i + 1).zfill(3)
        file = open(path + '/wiki-pages/wiki-' + wid + '.jsonl', 'r', encoding='utf8')
        for line in file:
            data = json.loads(line.rstrip('\n'))
            lines = data['lines'].split('\n')
            plines = []
            for line in lines:
                slines = line.split('\t')
                if len(slines) > 1 and slines[0].isnumeric():
                    plines.append(slines[1])
                else:
                    plines.append('')

            lines = plines
            data['id'] = unidecode(data['id'])
            articles[data['id']] = lines

    print('Preprocessing dataset...')
    files = ['train', 'dev']
    for file in files:
        fo = open(path + '/facts_{}.jsonl'.format(file), 'w+', encoding='utf8')
        with open(path + '/{}.jsonl'.format(file), encoding='utf8') as f:
            for line in f:
                data = json.loads(line.rstrip('\n'))
                claim = data['claim']
                label = data['label']
                evidence = []
                sents = []
                if label != 'NOT ENOUGH INFO':
                    for evidence_list in data['evidence']:
                        evidence_sents = []
                        extra_left = []
                        extra_right = []
                        evidence_words = 0
                        for e in evidence_list:
                            e[2] = unidecode(e[2])
                            if e[2] in articles and len(articles[e[2]]) > e[3]:
                                eid = e[3]
                                article = articles[e[2]]
                                evidence_sents.append(replace_symbols(article[eid].replace('  ', ' ').strip()))
                                evidence_words += len(article[eid].split(' '))
                                left = []
                                for i in range(0, eid):
                                    left.append(replace_symbols(article[i]))
                                extra_left.append(left)
                                right = []
                                for i in range(eid + 1, len(article)):
                                    right.append(replace_symbols(article[i]))
                                extra_right.append(right)
                            else:
                                evidence_sents = []
                                break
                        evidence_text = []
                        for i in range(len(evidence_sents)):
                            for j in range(len(extra_left[i])):
                                xlen = len(extra_left[i][j].split(' '))
                                if evidence_words + xlen < 254:
                                    evidence_text.append(extra_left[i][j])
                                    evidence_words += xlen
                            evidence_text.append(evidence_sents[i])
                            for j in range(len(extra_right[i])):
                                xlen = len(extra_right[i][j].split(' '))
                                if evidence_words + xlen < 254:
                                    evidence_text.append(extra_right[i][j])
                                    evidence_words += xlen

                        if len(evidence_text) > 0:
                            evidence_text = unidecode(' '.join(evidence_text))
                            evidence_text = ' '.join(evidence_text.split()).strip()
                            evidence.append(evidence_text)
                            sents.append(' '.join(evidence_sents))

                if label == 'NOT ENOUGH INFO' or len(evidence) > 0:
                    fo.write(json.dumps({
                        'statement': unidecode(replace_symbols(claim)),
                        'label': label,
                        'evidence_long': list(set(evidence)),
                        'evidence': list(set(sents))
                    }) + '\n')
        fo.close()


def preprocess_squad_dataset(path, output):
    # Read dataset
    with open(path) as f:
        dataset = json.load(f)

    # Iterate and write question-answer pairs
    with open(output, 'w+') as f:
        for article in dataset['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = [a['text'] for a in qa['answers']]
                    min_answer = ''
                    min_length = 1000
                    for answer in answers:
                        if len(answer) < min_length:
                            min_answer = answer
                            min_length = len(answer)
                    if min_answer == '':
                        continue
                    f.write(json.dumps({'question': question, 'answer': min_answer, 'evidence': context}))
                    f.write('\n')


def preprocess_nq_dataset():
    def _clean_token(token):
        return re.sub(u" ", "_", token["token"])

    def get_text_blocks(html):
        w = WebParser()
        html = '<div>{}</div>'.format(html)
        w.feed(html)
        blocks = w.get_blocks()
        return blocks

    files = ['train']
    for file in files:
        of = open('../data/qa/nq/{}_all_h.jsonl'.format(file), 'w+')
        with open('../data/qa/nq/{}.jsonl'.format(file), encoding='utf8') as f:
            for line in tqdm(f):
                data = json.loads(line.rstrip('\n'))
                question = data['question_text']

                if file == 'dev':
                    data['document_text'] = " ".join([_clean_token(t) for t in data["document_tokens"]])
                doc_tokens = data['document_text'].replace('*', '').split(' ')

                short_answer = ''
                if len(data['annotations'][0]['short_answers']) > 0:
                    short_answer_info = data['annotations'][0]['short_answers'][0]
                    doc_tokens.insert(short_answer_info['start_token'], '*')
                    doc_tokens.insert(short_answer_info['end_token'] + 1, '*')
                    short_answer = ' '.join(
                        doc_tokens[short_answer_info['start_token'] + 1:short_answer_info['end_token'] + 1])
                    short_answer = ' '.join(get_text_blocks(short_answer))

                long_answer_info = data['annotations'][0]['long_answer']
                long_answer = ' '.join(doc_tokens[long_answer_info['start_token']:long_answer_info['end_token']])
                long_answer = ' '.join(get_text_blocks(long_answer))

                if long_answer == '':
                    continue

                example = {
                    'question': question,
                    'answer': short_answer,
                    'evidence': long_answer
                }

                of.write(json.dumps(example) + '\n')


def preprocess_msmarco():
    files = ['train', 'dev']
    for file in files:
        fo = open('../data/qa/msmarco/{}_all.jsonl'.format(file), 'w+')
        with open('../data/qa/msmarco/{}.jsonl'.format(file), 'r') as f:
            for line in f:
                data = json.loads(line.rstrip('\n'))
                q = data['query']
                a = data['answers'][0]
                evidence = ''
                for passage in data['passages']:
                    if passage['is_selected'] == 1:
                        evidence = passage['passage_text']
                        break

                if '_' in q:
                    continue

                example = {
                    'question': q,
                    'answer': a,
                    'evidence': evidence
                }

                fo.write(json.dumps(example) + '\n')


def create_qa_ranking_dataset():
    files = ['train', 'dev']
    for file in files:
        relevant = 0
        irrelevant = 0
        max_irrelevant = 10 if file == 'train' else 1
        fo = open('../data/qa_ranking_large/{}.jsonl'.format(file), 'w+')
        with open('../data/qa/msmarco/{}.jsonl'.format(file), 'r') as f:
            for line in f:
                data = json.loads(line.rstrip('\n'))
                added_irrelevant = 0
                for passage in data['passages']:
                    example = {
                        'question': data['query'],
                        'answer': passage['passage_text'],
                        'relevant': passage['is_selected']
                    }
                    if passage['is_selected'] == 1 or added_irrelevant < max_irrelevant:
                        fo.write(json.dumps(example) + '\n')
                        if passage['is_selected'] == 1:
                            relevant += 1
                        else:
                            added_irrelevant += 1
                            irrelevant += 1

        print(relevant, irrelevant)


def preprocess_qa_nli_dataset(path):
    files = ['dev', 'train']
    for file in files:
        of = open(path + '/' + file + '_.tsv', 'w+')
        with open(path + '/' + file + '.tsv', 'r') as f:
            for line in f:
                line = line.rstrip()
                s = line.split('\t')
                if s[2].endswith('.'):
                    s[2] = s[2][:len(s[2]) - 1] + '?'
                if not s[2].endswith('?'):
                    s[2] += ' ?'
                input_str = s[2] + ' ' + s[3]
                output_str = s[4]
                of.write(input_str + '\t' + output_str + '\n')
        of.close()


def get_pair_input(tokenizer, sent1, sent2, max_len=256):
    text = "[CLS] {} [SEP] {} [SEP]".format(sent1, sent2)

    tokenized_text = tokenizer.tokenize(text)[:max_len]
    indexed_tokens = tokenizer.encode(text)[:max_len]

    segments_ids = []
    sep_flag = False
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == '[SEP]' and not sep_flag:
            segments_ids.append(0)
            sep_flag = True
        elif sep_flag:
            segments_ids.append(1)
        else:
            segments_ids.append(0)
    return indexed_tokens, segments_ids


def build_batch(tokenizer, text_list, max_len=256):
    token_id_list = []
    segment_list = []
    attention_masks = []
    longest = -1

    for pair in text_list:
        sent1, sent2 = pair
        ids, segs = get_pair_input(tokenizer, sent1, sent2, max_len=max_len)
        if ids is None or segs is None:
            continue
        token_id_list.append(ids)
        segment_list.append(segs)
        attention_masks.append([1] * len(ids))
        if len(ids) > longest:
            longest = len(ids)

    if len(token_id_list) == 0:
        return None, None, None

    # padding
    assert (len(token_id_list) == len(segment_list))
    for ii in range(len(token_id_list)):
        token_id_list[ii] += [0] * (longest - len(token_id_list[ii]))
        attention_masks[ii] += [1] * (longest - len(attention_masks[ii]))
        segment_list[ii] += [1] * (longest - len(segment_list[ii]))

    return token_id_list, segment_list, attention_masks


"""
class QAEncoderBatchGenerator(Sequence):
    def __init__(self, ids_to_paragraphs: List, ids_to_queries: Dict, qrels_train: List, tokenizer,
                 max_question_length=64, max_answer_length=512, batch_size=64):
        np.random.seed(42)
        self.ids_to_paragraphs = ids_to_paragraphs
        self.ids_to_queries = ids_to_queries
        self.qrels_train = qrels_train
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.qrels_train) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        question_texts = []
        answer_texts = []

        for index in indexes:
            query_id = self.qrels_train[index][0]
            answer_id = self.qrels_train[index][1]
            question_texts.append(self.ids_to_queries[query_id])
            answer_texts.append(self.ids_to_paragraphs[answer_id])

        question_inputs = get_inputs_text(texts=question_texts, tokenizer=self.tokenizer,
                                          max_length=self.max_question_length)
        answer_inputs = get_inputs_text(texts=answer_texts, tokenizer=self.tokenizer,
                                        max_length=self.max_answer_length)

        x = {
            'question_input_word_ids': question_inputs[0],
            'question_input_masks': question_inputs[1],
            'question_input_segments': question_inputs[2],
            'answer_input_word_ids': answer_inputs[0],
            'answer_input_masks': answer_inputs[1],
            'answer_input_segments': answer_inputs[2]
        }
        y = np.zeros((self.batch_size, self.batch_size))

        return x, y

    def on_epoch_end(self, logs=None):
        self.indexes = np.arange(len(self.qrels_train))
        np.random.shuffle(self.indexes)
"""


def create_qa_encoder_pretraining_dataset(dataset_file, output_file, total_samples):
    print('Loading paragraphs in memory...')

    articles = []
    file = open(dataset_file, 'r', encoding='utf8')
    while True:
        line = file.readline()
        if not line:
            break
        line = line.rstrip('\n')
        data = json.loads(line)
        articles.append(data['paragraphs'])

    print('Loaded paragraphs in memory')

    o = open(output_file, 'w+')

    added = 0
    while added < total_samples:
        index = randint(0, len(articles) - 1)
        article = articles[index]
        if len(article) < 2:
            continue
        if added % 2 == 0:  # Body-First-Selection
            paragraph = article[0]
            random_question_sentence_id = randint(0, len(paragraph) - 1)
            question = paragraph[random_question_sentence_id]
            random_answer_paragraph_id = randint(1, len(article) - 1)
            answer = ' '.join(article[random_answer_paragraph_id])
        else:  # Inverse-Cloze-Task
            random_paragraph_id = randint(0, len(article) - 1)
            paragraph = article[random_paragraph_id]
            random_question_sentence_id = randint(0, len(paragraph) - 1)
            question = paragraph[random_question_sentence_id]
            choice = random()
            if choice < 0.9:
                answer = ' '.join([paragraph[j] for j in range(len(paragraph)) if j != random_question_sentence_id])
            else:
                answer = ' '.join(paragraph)

        o.write(json.dumps({'s1': question, 's2': answer}) + '\n')

        added += 1
        if added % 10000 == 0:
            print(added)

    o.close()


def create_random_wiki_paragraphs(dataset_file, output_file, total_samples, max_length=510):
    print('Loading paragraphs in memory...')

    articles = []
    file = open(dataset_file, 'r', encoding='utf8')
    while True:
        line = file.readline()
        if not line:
            break
        line = line.rstrip('\n')
        data = json.loads(line)
        articles.append(data['paragraphs'])

    print('Loaded paragraphs in memory')

    o = open(output_file, 'w+')
    added = 0
    while added < total_samples:
        index = randint(0, len(articles) - 1)
        article = articles[index]
        if len(article) < 1:
            continue
        random_paragraph_id = randint(0, len(article) - 1)
        paragraph = article[random_paragraph_id]

        p = []
        words = 0
        i = 0
        while i < len(paragraph):
            nwords = len(paragraph[i].split(' '))
            if nwords + words >= max_length:
                break
            p.append(paragraph[i])
            i += 1

        o.write(json.dumps({'text': ' '.join(p)}) + '\n')
        added += 1
        if added % 10000 == 0:
            print(added)
    o.close()


def load_unsupervised_dataset(dataset_file):
    print('Loading dataset...')
    x = pickle.load(open(dataset_file, "rb"))
    print('Done')
    return x, len(x[0])


def create_entailment_dataset(dataset_file, tokenizer, output_file):
    x = [[] for _ in range(6)]
    y = []

    with open(dataset_file, encoding='utf8') as file:
        for j, line in enumerate(file):
            if j % 10000 == 0:
                print(j)
            line = line.rstrip('\n')
            sample = json.loads(line)
            label = sample['label']
            s1 = tokenize(text=sample['s1'],
                          max_length=256,
                          tokenizer=tokenizer)
            s2 = tokenize(text=sample['s2'],
                          max_length=256,
                          tokenizer=tokenizer)
            example = s1 + s2
            for i in range(6):
                x[i].append(example[i])
            if label == 'entailment':
                label = [1.0, 0.0, 0.0]
            elif label == 'contradiction':
                label = [0.0, 1.0, 0.0]
            else:
                label = [0.0, 0.0, 1.0]
            y.append(np.asarray(label, dtype='float32'))

    for i in range(6):
        x[i] = np.asarray(x[i])

    y = np.asarray(y)
    data = [x, y]

    pickle.dump(data, open(output_file, "wb"), protocol=4)


def load_supervised_dataset(dataset_file):
    print('Loading dataset...')
    d = pickle.load(open(dataset_file, "rb"))
    print('Done')
    return d[0], d[1]


def preprocess_sg():
    fo = open('../data/qa/squad/dev_sg_sq.txt', 'w+')
    f = open('../data/qa/squad/dev.jsonl', 'r')
    for line in f:
        d = json.loads(line.rstrip('\n'))
        q = d['question']
        a = d['answer']

        if q.endswith('.'):
            q = q[:len(q) - 1] + '?'
        if not q.endswith('?'):
            q += ' ?'

        fo.write(q + ' ' + a + '\n')


def preprocess_msmarco_():
    fo = open('../data/qa/msmarco/dev_sg.tsv', 'w+')
    with open('../data/qa/msmarco/dev.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line.rstrip('\n'))
            if data['wellFormedAnswers'] == '[]' or data['query_type'] == 'DESCRIPTION':
                continue
            q = data['query']
            a = data['answers'][0]
            fa = data['wellFormedAnswers'][0]

            if len(a) > len(q):
                continue

            if q.endswith('.'):
                q = q[:len(q) - 1] + '?'
            if not q.endswith('?'):
                q += ' ?'

            if '_' in q:
                continue

            fo.write('{} {}\t{}\n'.format(q, a, fa))


def create_qnli_nq():
    files = ['train', 'dev']
    for file in files:
        o = open('../data/qnli/{}_nq.jsonl'.format(file), 'w+')

        fs = open('../data/qnli/{}.txt'.format(file))
        statements = [s.rstrip('\n') for s in fs]

        fd = open('../data/qa/nq/{}_nli.jsonl'.format(file))
        for i, line in enumerate(fd):
            d = json.loads(line.rstrip('\n'))
            o.write(json.dumps({
                'statement': statements[i],
                'question': d['question'],
                'answer': d['answer'],
                'evidence': d['evidence']
            }) + '\n')


def create_qnli_ms():
    files = ['train', 'dev']
    for file in files:
        o = open('../data/qnli/{}_ms.jsonl'.format(file), 'w+')
        with open('../data/qa/msmarco/{}.jsonl'.format(file), 'r') as f:
            for line in f:
                data = json.loads(line.rstrip('\n'))
                if data['wellFormedAnswers'] == '[]' or data['query_type'] == 'DESCRIPTION':
                    continue
                q = data['query']
                a = data['answers'][0]
                fa = data['wellFormedAnswers'][0]

                evidence = ''
                for p in data['passages']:
                    if p['is_selected'] == 1:
                        evidence = p['passage_text']
                if evidence == '' or '_' in q:
                    continue

                o.write(json.dumps({
                    'statement': fa,
                    'question': q,
                    'answer': a,
                    'evidence': evidence
                }) + '\n')


def create_qnli_ms_2():
    files = ['train', 'dev']
    for file in files:
        o = open('../data/qnli/{}_ms_nw.jsonl'.format(file), 'w+')
        fs = open('../data/qa/msmarco/statements_{}_ms.txt'.format(file))
        statements = [s.rstrip('\n') for s in fs]
        with open('../data/qa/msmarco/{}.jsonl'.format(file), 'r') as f:
            n = 0
            for line in f:
                data = json.loads(line.rstrip('\n'))
                if data['wellFormedAnswers'] != '[]' or data['query_type'] == 'DESCRIPTION':
                    continue
                q = data['query']
                a = data['answers'][0]

                if a == 'No Answer Present.' or '_' in q:
                    continue

                evidence = ''
                for p in data['passages']:
                    if p['is_selected'] == 1:
                        evidence = p['passage_text']

                if '\n' in q or '\n' in a:
                    nl = (q + a).count('\n') + 1
                    print(nl)
                    n += nl
                    continue

                o.write(json.dumps({
                    'statement': statements[n],
                    'question': q,
                    'answer': a,
                    'evidence': evidence
                }) + '\n')

                n += 1


def create_msmarco_sg():
    files = ['train', 'dev']
    for file in files:
        o = open('../data/qa/msmarco/{}_ms_sg.txt'.format(file), 'w+')
        with open('../data/qa/msmarco/{}.jsonl'.format(file), 'r') as f:
            for line in f:
                data = json.loads(line.rstrip('\n'))
                if data['wellFormedAnswers'] != '[]' or data['query_type'] == 'DESCRIPTION':
                    continue
                q = data['query']
                a = data['answers'][0]

                if a == 'No Answer Present.' or '_' in q:
                    continue

                if q.endswith('.'):
                    q = q[:len(q) - 1] + '?'
                if not q.endswith('?'):
                    q += ' ?'

                o.write(q + ' ' + a + '\n')
                """
                o.write(json.dumps({
                    'statement': fa,
                    'question': q,
                    'answer': a,
                    'evidence': evidence
                }) + '\n')
                """


def create_qar():
    files = ['train', 'dev']
    for file in files:
        o = open('../data/qar/{}2.jsonl'.format(file), 'w+')

        f = open('../data/qa/nq/{}_all.jsonl'.format(file))
        for line in f:
            d = json.loads(line.rstrip('\n'))
            if d['evidence'] == '':
                continue
            o.write(json.dumps({
                'question': d['question'],
                'answer': d['evidence'],
                'source': 'natural-questions'
            }) + '\n')

        f = open('../data/qa/msmarco/{}_all.jsonl'.format(file))
        for line in f:
            d = json.loads(line.rstrip('\n'))
            if d['evidence'] == '':
                continue
            o.write(json.dumps({
                'question': d['question'],
                'answer': d['evidence'],
                'source': 'msmarco'
            }) + '\n')


"""
question_inputs = get_inputs_text(texts=question_texts, tokenizer=tokenizer,
                                      max_length=max_question_length)
    answer_inputs = get_inputs_text(texts=answer_texts, tokenizer=tokenizer,
                                    max_length=max_answer_length)

    x = {
        'question_input_word_ids': question_inputs[0],
        'question_input_masks': question_inputs[1],
        'question_input_segments': question_inputs[2],
        'answer_input_word_ids': answer_inputs[0],
        'answer_input_masks': answer_inputs[1],
        'answer_input_segments': answer_inputs[2]
    }
    y = np.zeros((self.batch_size, self.batch_size))
"""

"""
create_random_wiki_paragraphs('../data/wikipedia/wiki_paragraphs.jsonl',
                              '../data/random_wiki_paragraphs.jsonl',
                              total_samples=1000000,
                              max_length=510)
"""

# preprocess_nq_dataset()
