import json
import math
import requests

from multiprocessing.pool import ThreadPool
from bs4 import BeautifulSoup
from boilerpipe.extract import Extractor


def get_html(url: str, html=False) -> str:
    """
    Downloads the html source code of a webpage
    :param url:
    :return: html source code
    """
    try:
        data = requests.get(url, stream=True, timeout=10, allow_redirects=True)
    except:
        return ''
    if 'content-type' not in data.headers or (html and not data.headers['content-type'].startswith('text/html')):
        return ''
    try:
        data = data.text
    except:
        data = ''

    return data


added_urls = set()

with open('data/covid-news.jsonl', encoding='utf8') as f:
    for line in f:
        try:
            d = json.loads(line.rstrip('\n'))
            added_urls.add(d['url'])
        except:
            pass

print(len(added_urls))

for i in range(1000):
    api_url = 'https://newslookup.com/results?l=2&q=covid-19+coronavirus&dp=&mt=-1&mkt=0&mtx=0&mktx=0&s=&groupby=no&cat=-1&from=&fmt=&tp=720&ps=50&ovs=&page={}'.format(i+1)
    data = get_html(api_url)
    print(api_url)
    soup = BeautifulSoup(data, 'lxml')
    links = soup.findAll('a', {'class': 'title'})
    urls = []
    titles = {}
    dates = {}
    for link in links:
        url = link['href']
        if url in added_urls:
            continue
        urls.append(url)
        titles[url] = link.text
        dates[url] = ''
        added_urls.add(url)
        if len(urls) % 100 == 0:
            print(len(urls))
    dates_e = soup.findAll('span', {'class': 'stime'})
    for i, date in enumerate(dates_e):
        if i < len(urls):
            dates[urls[i]] = date.text

    results = {}
    def process(url):
        global results
        extractor = Extractor(extractor='ArticleExtractor', url=url)
        results[url] = extractor.getText()

    p = ThreadPool(50)
    pool_output = p.map_async(process, urls)
    p.close()
    p.join()

    f = open('data/covid-news.jsonl', 'a')
    for url, article in results.items():
        print(titles[url])
        article = str(article)
        if article != '':
            f.write(json.dumps({
                'title': titles[url],
                'text': article,
                'url': url,
                'date': dates[url]
            })+'\n')
    f.close()
