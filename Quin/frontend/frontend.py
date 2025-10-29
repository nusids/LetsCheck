import argparse
import glob
import hashlib
import json
import logging
import os
from time import sleep

from flask import Flask, request, send_from_directory, redirect
from flask_cors import CORS

from waitress import serve


class Frontend:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.secret = '123'
        self.pending_requests = set()
        self.counter_fname = {
            'covid': './counter_1',
            'general': './counter_2'
        }
        self.counter = {
            'covid': int(open(self.counter_fname['covid'], 'r').readline()),
            'general': int(open(self.counter_fname['general'], 'r').readline())
        }

    def increment_counter(self, n='covid'):
        self.counter[n] += 1
        with open(self.counter_fname[n], 'w') as f:
            logging.info(f"Recording numbers of queries processed for api ({n}): {self.counter[n]}")
            f.write(f'{self.counter[n]}')

    @staticmethod
    def response(data=None, status='ok'):
        return json.dumps({'status': status, 'data': data})

    def build_endpoints(self):
        """
        @self.app.before_request
        def before_request():
            if request.url.startswith('http://'):
                url = request.url.replace('http://', 'https://', 1)
                code = 301
                return redirect(url, code=code)
        """

        @self.app.route('/clear_cache', methods=['POST'])
        def clear_cache():
            secret = request.args.get('secret')
            if secret != self.secret:
                return self.response(status='invalid secret')

            files = glob.glob('cache/*')
            for f in files:
                os.remove(f)

            logging.info('Cleared API cache')

            return self.response()

        @self.app.route('/save_cache', methods=['POST'])
        def save_cache():
            secret = request.args.get('secret')
            if secret != self.secret:
                return self.response(status='invalid secret')

            request_url = request.args.get('request_url')
            hash = hashlib.sha1(request_url.encode('utf-8')).hexdigest()
            data = request.form['data']
            f = open('cache/{}.json'.format(hash), 'w+')
            f.write(data)
            f.close()

            logging.info('Saved response for {}'.format(request_url))

            return self.response()

        @self.app.route('/get_pending_requests', methods=['POST'])
        def get_pending_requests():
            try:
                secret = request.args.get('secret')
                if secret != self.secret:
                    return self.response(status='invalid secret')

                return self.response(data=list(self.pending_requests))
            except Exception as e:
                logging.warning(e)
                return self.response(data=[])

        @self.app.route('/remove_pending_request', methods=['POST'])
        def remove_pending_request():
            try:
                secret = request.args.get('secret')
                if secret != self.secret:
                    return self.response(status='invalid secret')

                request_url = request.args.get('request_url')
                self.pending_requests.remove(request_url)
            except:
                pass

            return self.response()

        @self.app.route('/api', methods=['GET'])
        def api_call():
            query = request.args.get('query').lower()
            source = request.args.get('source').lower()
            request_url = 'api?source={}&query={}'.format(source, query)
            hash = hashlib.sha1(request_url.encode('utf-8')).hexdigest()
            cache_file = 'cache/{}.json'.format(hash)
            if os.path.exists(cache_file):
                return self.response(data=json.loads(open(cache_file).read()))
            else:
                self.pending_requests.add(request_url)
                waited = 0
                while waited < 60:
                    sleep(1)
                    waited += 1
                    if os.path.exists(cache_file):
                        self.increment_counter(n='covid')
                        return self.response(data=json.loads(open(cache_file).read()))
                return self.response(status='pending')

        @self.app.route('/api2', methods=['GET'])
        def api_call_2():
            query = request.args.get('query').lower()
            request_url = 'api2?query={}'.format(query)
            print(f"Received request: {query}")
            hash = hashlib.sha1(request_url.encode('utf-8')).hexdigest()
            cache_file = 'cache/{}.json'.format(hash)
            if os.path.exists(cache_file):
                return self.response(data=json.loads(open(cache_file).read()))
            else:
                self.pending_requests.add(request_url)
                waited = 0
                while waited < 60:
                    sleep(1)
                    waited += 1
                    if os.path.exists(cache_file):
                        self.increment_counter(n='general')
                        return self.response(data=json.loads(open(cache_file).read()))
                return self.response(status='pending')

        @self.app.route('/stats')
        def get_stats():
            return self.response(data=self.counter)

        @self.app.route('/<path:path>')
        def send_static_file(path):
            return send_from_directory('static', path)

        @self.app.route('/')
        def send_index_page():
            return send_from_directory('static', 'general.html')

        @self.app.route('/covid')
        def send_covid_page():
            return send_from_directory('static', 'covid.html')

        @self.app.after_request
        def remove_header(response):
            del response.headers['Server']
            # response.headers.add('Access-Control-Allow-Origin', 'http://quin-api.ids-exalit/*')
            # response.headers['X-Frame-Options'] = 'SAMEORIGIN'
            return response

    def serve(self):
        self.build_endpoints()
        print('ok')
        return self.app


q = Frontend()
app = q.serve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quin Frontend and middleware.')
    parser.add_argument('--port', default=80)
    args = parser.parse_args()
    serve(app, host='0.0.0.0', port=args.port, ident=None)

# ssl_context=('/etc/letsencrypt/live/quin.algoprog.com/fullchain.pem','/etc/letsencrypt/live/quin.algoprog.com/privkey.pem')
