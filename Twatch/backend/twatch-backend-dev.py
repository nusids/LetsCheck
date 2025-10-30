import json
import logging
import math
import re as regex
import requests
import sqlite3
from time import sleep

API_URLS = ['http://localhost:8080/api/']
logging.getLogger().setLevel(logging.INFO)
MAX_PARAMS = 500
FILTERED_WORDS = ['covid-19', 'covid19', 'covid']


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def split_into_batches(ls):
    return [ls[s * MAX_PARAMS:(s + 1) * MAX_PARAMS]
            for s in range(math.ceil(len(ls) / MAX_PARAMS))]


def generate_param_str(batch):
    return '(' + ','.join(['?' for _ in batch]) + ')'


class Twatch:
    def __init__(self):
        self.con = sqlite3.connect('twatch.db')
        self.con.row_factory = dict_factory
        logging.info('Initialized!')

    def query(self, q, params=None):
        cur = self.con.cursor()
        if params:
            re = cur.execute(q, params).fetchall()
        else:
            re = cur.execute(q).fetchall()
        cur.close()
        return re

    def reload_trend(self):
        cur = self.con.cursor()
        cur.execute('DROP TABLE IF EXISTS trend')
        cur.execute('CREATE TABLE trend AS '
                    "SELECT strftime('%s', date(unix, 'unixepoch')) AS x, COUNT(*) AS y "
                    "FROM tweets GROUP BY date(unix, 'unixepoch')"
                    )

    def process_request(self, req):
        if req['type'] == 'trend':
            return self.query('SELECT * FROM trend WHERE x IS NOT NULL AND y IS NOT NULL;')
        elif req['type'] == 'users_by_id':
            param_sub_str = generate_param_str(req['ids'])
            return self.query(f'SELECT * FROM users WHERE user_id IN {param_sub_str}', req['ids'])
        elif req['type'] == 'tweets_by_id':
            tweet_ids_batches = split_into_batches(req['ids'])
            re = []
            for batch in tweet_ids_batches:
                param_sub_str = generate_param_str(batch)
                if req['full']:
                    re += self.query('SELECT * FROM tweets '
                                     'INNER JOIN users ON tweets.user_id = users.user_id '
                                     f'WHERE tweet_id IN {param_sub_str}', batch)
                else:
                    re += self.query('SELECT '
                                     'tweet_id, unix, reply_parent_id, num_retweets, users.user_id, user_name, '
                                     '(num_favourites + num_retweets) AS influence_score'
                                     ' FROM tweets '
                                     'INNER JOIN users ON tweets.user_id = users.user_id '
                                     f'WHERE tweet_id IN {param_sub_str}', batch)
            return re
        elif req['type'] == 'reply_tweets_by_root_id':
            tweet_ids_batches = split_into_batches(req['ids'])
            re = []
            for batch in tweet_ids_batches:
                param_sub_str = generate_param_str(batch)
                re += self.query('SELECT * FROM tweets '
                                 'INNER JOIN users ON tweets.user_id = users.user_id '
                                 f'WHERE root_tweet_id IN {param_sub_str}', batch)
            return re

        elif req['type'] == 'tweets_matching_text':
            query = req['queryRaw'].lower().split()
            tokens = []
            for q in query:
                if q not in FILTERED_WORDS:
                    tokens.append(regex.sub('[^a-zA-Z0-9]', '', q))
            query_str = ' AND '.join(tokens)
            return self.query('SELECT t.tweet_id, unix, reply_parent_id, num_retweets, '
                              'users.user_id, user_name, users.user_followers_count, '
                              '(num_favourites + num_retweets) AS influence_score '
                              'FROM tweets_fts AS tf '
                              'INNER JOIN tweets AS t on tf.tweet_id = t.tweet_id '
                              'INNER JOIN users ON t.user_id = users.user_id '
                              f"WHERE tf.text MATCH '{query_str}'")
        elif req['type'] == 'trending_news':
          f = open('trending_news', 'r')
          lines = f.readlines()
          return [line.strip() for line in lines]
        else:
            logging.warning(f"Invalid request {req}")
            return []


def process_api_requests(twatch: Twatch) -> None:
    sleep_time = 1
    while True:
        sleep(sleep_time)
        for API_URL in API_URLS:
            try:
                res = requests.post('{}/get_pending_requests?secret=123'.format(API_URL), data={}, timeout=10)
                pending_requests = json.loads(res.text)
                if len(pending_requests) == 0:
                    sleep_time = min(sleep_time * 2, 1)
                else:
                    sleep_time = 0.001
                for req in pending_requests:
                    try:
                        if not req['type'] or not req['id']:
                            continue
                        logging.info('Processing request {}'.format(req['type']))
                        re = twatch.process_request(req)
                        r = requests.post(
                            '{}/save_pending_result'.format(API_URL),
                            json={'result': re, 'request_id': req['id'], 'isValid': True},
                            timeout=10)
                        if r.status_code != 200:
                            logging.debug(str(r))
                            logging.info(f'Error saving result for {req["id"]}')
                        else:
                            logging.info(f'Sent results for {req["id"]}')
                    except Exception:
                        logging.exception(f'Processing error for {req["id"]}')
                    finally:
                        r = requests.post('{}/remove_pending_request?secret=123&request_id={}'.format(API_URL,
                                                                                                      req['id']),
                                          timeout=10)
                        if r.status_code != 200:
                            logging.info(f'Error removing pending request for {req["id"]}')
            except requests.exceptions.ConnectionError:
                logging.warning('Frontend not responding, waiting for 5s...')
                sleep(5)
            except Exception as e:
                logging.exception(e)
                logging.info('Encountered error, restarting in 5s...')
                sleep(5)


if __name__ == '__main__':
    t = Twatch()
    t.reload_trend()
    process_api_requests(t)
