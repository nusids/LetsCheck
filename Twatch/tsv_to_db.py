from datetime import datetime, date, timedelta
from enum import Enum
import urllib.request
import gzip
import os

import pandas as pd
from twarc import Twarc
# from postgres import Postgres
import sqlite3
# from psycopg2.errors import UniqueViolation

from twitter_auth_info import Auth

import pdb

twarc = Twarc(Auth.api['key'], Auth.api['secret'], Auth.access['key'], Auth.access['secret'])
max_count = -1  # For debugging
# https://developer.twitter.com/en/docs/tweets/post-and-engage/api-reference/get-statuses-lookup
LOOKUP_TWEETS_LIMIT = 100
runtime_log = open("tsv_to_postgres_log", "w")
error_log = open("tsv_to_postgres_errors_log", "w")
username_whitelist = open("whitelist", "r").read().splitlines()
db = sqlite3.connect('twatch.db')
chunksize = 10 ** 6


def get_time_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def init_db():
    with open('init.sql', 'r') as script:
        cur = db.cursor()
        queries = script.read().split(';')
        for q in queries:
            cur.execute(q)
    
    
class ChunkStatus(Enum):
    NOT_STARTED = 0
    INCOMPLETE = 1
    COMPLETE = 2


class ChunkManager():
    def __init__(self):
        self.chunks = []
        self.incomplete_chunks = []
        self.load_chunks()

    def load_chunks(self):
        with open('processed_chunks', 'r') as f:
            lines = f.readlines()
            for l in lines:
                if l[0] != '~':
                    self.chunks.append(int(l))
                else:
                    self.incomplete_chunks.append(int(l[1:]))

    def save_chunks(self):
        with open('processed_chunks', 'w') as chunk_file:
            for c in self.chunks:
                chunk_file.write("{}\n".format(c))
            for c in self.incomplete_chunks:
                chunk_file.write("~{}\n".format(c))

    def add_completed_chunk(self, chunk_id):
        if chunk_id not in self.chunks:
            self.chunks.append(chunk_id)
        if chunk_id in self.incomplete_chunks:
            self.incomplete_chunks.remove(chunk_id)
        self.save_chunks()

    def add_incomplete_chunk(self, chunk_id):
        if chunk_id in self.chunks:
            raise Exception("Chunck already completed")
        if chunk_id not in self.incomplete_chunks:
            self.incomplete_chunks.append(chunk_id)
        self.save_chunks()

    def check_chunk(self, chunk_id):
        if chunk_id in self.chunks:
            return ChunkStatus.COMPLETE
        elif chunk_id in self.incomplete_chunks:
            return ChunkStatus.INCOMPLETE
        else:
            return ChunkStatus.NOT_STARTED

    def print_chunks(self):
        print("Completed chunks: {}".format(str(self.chunks)))
        print("Incomplete chunks: {}".format(str(self.incomplete_chunks)))

    def reset_chunks(self):
        self.chunks = []
        self.incomplete_chunks = []
        self.save_chunks()

    def __del__(self):
        self.save_chunks()


def get_daily_tweets_df(date):
    try:
        url = f"https://raw.githubusercontent.com/thepanacealab/covid19_twitter/master/dailies/{date}/{date}-dataset.tsv"
        filename = f'{date}-dataset.tsv'
        print(f'Downloading tweets from {date}')
        urllib.request.urlretrieve(url, filename)
        f = open(filename, 'r')
    except Exception as e:
        print(f"Unable to download {filename}. Trying gzip version instead.")
        url = f"https://raw.githubusercontent.com/thepanacealab/covid19_twitter/master/dailies/{date}/{date}-dataset.tsv.gz"
        filename = f'{date}-dataset.tsv.gz'
        print(f'Downloading tweets from {date}')
        urllib.request.urlretrieve(url, filename)
        f = gzip.open(filename)
    finally:
        try:
            df = pd.read_csv(f, sep="\t", iterator=True, chunksize=chunksize)
            return df, filename
        except Exception as e:
            raise e


def crawl_tsv(df):
    buffer = []
    num_tweets_saved = 0
    total = 0
    chunk_manager = ChunkManager()
    cur = db.cursor()

    for chunk_id, in_df in enumerate(df):
        last_stop = 0
        if chunk_manager.check_chunk(chunk_id) == ChunkStatus.COMPLETE:
            print("Chunk #{} is completed.".format(chunk_id), end='\r', flush=True)
            continue
        if chunk_manager.check_chunk(chunk_id) == ChunkStatus.INCOMPLETE:
            print("Chunk #{} is partially completed.".format(chunk_id))
            chunk_test_distance = int(in_df.shape[0] / LOOKUP_TWEETS_LIMIT)
            test_row_ids = [i * chunk_test_distance for i in range(LOOKUP_TWEETS_LIMIT)]
            test_results = [False for i in range(LOOKUP_TWEETS_LIMIT)]

            for i, test_row_id in enumerate(test_row_ids):
                row = in_df.iloc[test_row_id]
                q = cur.execute(''' SELECT * FROM tweets WHERE tweet_id = ?''', [str(row.tweet_id)])
                test_tweet = q.fetchone()
                if test_tweet is not None:
                    test_results[i] = True
            last_stop_id = len(test_results) - 1
            while last_stop_id > 0:
                if test_results[last_stop_id]:
                    break
                else:
                    last_stop_id -= 1
            last_stop = test_row_ids[last_stop_id]
            print("Resuming from {}".format(last_stop))

        try:
            print("Reading chunk {}.".format(chunk_id), end="\r", flush=True)
            in_df = in_df[in_df['lang'] == 'en']
            for row in in_df.itertuples():
                if row.Index < last_stop:
                    continue
                if max_count > 0 and row.Index > max_count:
                    break

                q = cur.execute(''' SELECT * FROM tweets WHERE tweet_id = ?''', [str(row.tweet_id)])
                test_tweet = q.fetchone()
                if test_tweet is not None:
                    continue

                buffer.append(row.tweet_id)

                if len(buffer) < LOOKUP_TWEETS_LIMIT:
                    continue
                else:
                    for tweet in twarc.hydrate(buffer):
                        try:
                            if tweet['retweet_count'] + tweet['favorite_count'] < 10 and tweet['user'][
                                'screen_name'] not in username_whitelist:
                                continue
                            user = tweet['user']
                            unix = int(datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y').timestamp())
                            if tweet.get('retweeted_status') is not None:
                                continue
                            else:
                                cur.execute(''' INSERT INTO tweets(tweet_id,unix,user_id,num_retweets,num_favourites,
                                           reply_parent_id,text,location)
                                          VALUES (?,?,?,?,?,?,?,?)''',
                                            (tweet["id_str"], unix, user["id_str"], tweet["retweet_count"],
                                             tweet["favorite_count"],
                                             tweet["in_reply_to_status_id"] if tweet[
                                                                                   "in_reply_to_status_id"] is not None else -1,
                                             tweet["full_text"], str(tweet["coordinates"])
                                             ))
                            try:
                                cur.execute(''' INSERT INTO users(user_id,user_name,user_screen_name,user_location,user_followers_count,user_friends_count,user_is_verified,user_profile_image_url)
                                            VALUES (?,?,?,?,?,?,?,?) ON CONFLICT (user_id) DO UPDATE
                                            SET user_name = excluded.user_name,
                                                user_screen_name = excluded.user_screen_name,
                                                user_followers_count = excluded.user_followers_count,
                                                user_friends_count = excluded.user_friends_count,
                                                user_is_verified = excluded.user_is_verified,
                                                user_profile_image_url = excluded.user_profile_image_url;
                                        ''', (user['id_str'], user['name'], user['screen_name'], user['location'],
                                              user['followers_count'], user['friends_count'], user['verified'],
                                              user['profile_image_url_https']))
                            except sqlite3.IntegrityError:
                                pass
                            except Exception as e:
                                raise e
                            finally:
                                num_tweets_saved += 1
                        except sqlite3.IntegrityError:
                            pass
                        except Exception as e:
                            print("Encountered error at line {}".format(row.Index))
                            error_log.write("====={}=====\n".format(get_time_now()))
                            error_log.write("Encountered error when processing line {}\n".format(row.Index))
                            error_log.write("Number of tweets saved: {}\n".format(num_tweets_saved))
                            error_log.write(str(e) + "\n")
                            error_log.write("\n=============================\n")
                            error_log.flush()
                        finally:
                            db.commit()
                            runtime_log.write("===== {} =====\n".format(get_time_now()))
                            runtime_log.write(
                                "Stored tweets until line {} to DB, {} tweets saved so far.\n".format(row.Index,
                                                                                                      num_tweets_saved))
                            runtime_log.write("=============================\n")
                            runtime_log.flush()
                            print("Chunk #{}: {}/{} tweets processed, {} new tweets saved".format(chunk_id,
                                                                                                  row.Index % chunksize,
                                                                                                  in_df.shape[0],
                                                                                                  num_tweets_saved),
                                  end="\r", flush=True)
                            buffer.clear()
        except:
            print("Chunk {} interrupted.".format(chunk_id))
            db.commit()
            chunk_manager.add_incomplete_chunk(chunk_id)
            raise
        db.commit()
        chunk_manager.add_completed_chunk(chunk_id)


if __name__ == '__main__':
    while True:
        try:
            f = open('last_processed_date', 'r')
            w = f.readline()
            last_date = w.strip()
            try:
                df, filename = get_daily_tweets_df(last_date)
            except:
                print(f'Error downloading daily tweets for {last_date}')
                break
            crawl_tsv(df)
            os.remove(filename)
            f.close()
            last_date_d = date.fromisoformat(last_date)
            next_date_d = last_date_d + timedelta(days=1)
            f = open('last_processed_date', 'w')
            f.write(next_date_d.isoformat())
            f.close()
            f = open('processed_chunks', 'w')
            f.write('')
            f.close()
            out = open('new_tweets_available', 'w')
            out.close()
        except Exception as e:
            print(f"Encountered error when processing tweets")
            error_log.write("====={}=====\n".format(get_time_now()))
            error_log.write(str(e) + "\n")
            error_log.write("\n=============================\n")
            error_log.flush()
