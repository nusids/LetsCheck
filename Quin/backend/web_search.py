import random
import re

from pprint import pprint
from urllib.request import Request, urlopen
from html.parser import HTMLParser
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, parse_qs

import requests, lxml, json
import logging

import pdb

def USER_AGENT():
    uastrings = [
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/600.8.9 (KHTML, like Gecko) Version/8.0.8 Safari/600.8.9',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:53.0) Gecko/20100101 Firefox/53.0',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1',
    ]
    ua = random.choice(uastrings)
    logging.info(f'Using user agent: {ua}')
    return ua


def brave_search(query: str, pages=1) -> list:
    params = {
        'q': quote_plus(query),
        'source': 'web',
        'tf': 'at',
    }
    headers = {
        'User-Agent': USER_AGENT()
    }

    urls = []
    for page in range(pages):
        params['offset'] = page
        html = requests.get('https://search.brave.com/search', headers=headers, params=params)
        soup = BeautifulSoup(html.text, 'lxml')

        for result in soup.select('.snippet'):
            link = result.select_one('.result-header').get('href')
            """
            sitelinks_container = result.select('.deep-results-buttons .deep-link')
            sitelinks = None

            if sitelinks_container:
                sitelinks = []
                for sitelink in sitelinks_container:
                    sitelinks.append({
                        'title': sitelink.get_text().strip(),
                        'link': sitelink.get('href')
                    })
            """
            urls.append(link)
    return urls


def bing_search(query: str, pages=1) -> list:
    """
    Gets web results from Bing
    :param query: query to search
    :param pages_number: number of search pages to scrape
    :return: a list of links in ranked order
    """
    urls = []
    for page in range(pages):
        first = page * 10 + 1
        address = "https://www.bing.com/search?q=" + quote_plus(query) + '&first=' + str(first)
        data = get_html(address)
        soup = BeautifulSoup(data, 'lxml')
        links = soup.findAll('li', {'class': 'b_algo'})
        urls.extend([link.find('h2').find('a')['href'] for link in links])

    return urls


def duckduckgo_search(query: str, pages=1):
    """
    NOT WORKING; LIKELY BLOCKED
    """
    urls = []
    start_index = 0
    for page in range(pages):
        address = "https://duckduckgo.com/html/?kl=en-us&q={}&s={}".format(quote_plus(query), start_index)
        data = get_html(address)
        soup = BeautifulSoup(data, 'lxml')
        links = soup.findAll('a', {'class': 'result__a'})
        urls.extend([link['href'] for link in links])
        start_index = len(urls)
    try:
        urls = [parse_qs(l.split('/')[-1][5:])[''][0] for l in urls]
    except:
        logging.warn(f'Parsing failed for {len(urls)} urls')
    return urls


def news_search(query: str, pages=1):
    urls = []
    for page in range(pages):
        api_url = f'https://newslookup.com/results?l=2&q={quote_plus(query)}&dp=&mt=-1&mkt=0&mtx=0&mktx=0&s=&groupby=no&cat=-1&from=&fmt=&tp=720&ps=50&ovs=&page={page}'
        data = get_html(api_url)
        soup = BeautifulSoup(data, 'lxml')
        links = soup.findAll('a', {'class': 'title'})
        urls.extend([link['href'] for link in links])
    return urls


def get_html(url: str) -> str:
    """
    Downloads the html source code of a webpage
    :param url:
    :return: html source code
    """
    try:
        headers = {
            'User-Agent': USER_AGENT()
        }
        req = Request(url, headers=headers)
        page = urlopen(req, timeout=3)
        return str(page.read())
    except:
        return ''


class WebParser(HTMLParser):
    """
    A class for converting the tagged html to formats that can be used by a ML model
    """

    def __init__(self):
        super().__init__()
        self.block_tags = {
            'div', 'p'
        }
        self.inline_tags = {
            '', 'a', 'b', 'tr', 'main', 'span', 'time', 'td',
            'sup', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'em', 'strong', 'br'
        }
        self.allowed_tags = self.block_tags.union(self.inline_tags)
        self.opened_tags = []
        self.block_content = ''
        self.blocks = []

    def get_last_opened_tag(self):
        """
        Gets the last visited tag
        :return:
        """
        if len(self.opened_tags) > 0:
            return self.opened_tags[len(self.opened_tags) - 1]
        return ''

    def error(self, message):
        pass

    def handle_starttag(self, tag, attrs):
        """
        Handles the start tag of an HTML node in the tree
        :param tag: the HTML tag
        :param attrs: the tag attributes
        :return:
        """
        self.opened_tags.append(tag)
        if tag in self.block_tags:
            self.block_content = self.block_content.strip()
            if len(self.block_content) > 0:
                if not self.block_content.endswith('.'):
                    self.block_content += '.'
                self.block_content = self.block_content.replace('\\n', ' ').replace('\\r', ' ')
                self.block_content = re.sub("\s\s+", " ", self.block_content)
                self.blocks.append(self.block_content)
            self.block_content = ''

    def handle_endtag(self, tag):
        """
        Handles the end tag of an HTML node in the tree
        :param tag: the HTML tag
        :return:
        """
        if len(self.opened_tags) > 0:
            self.opened_tags.pop()

    def handle_data(self, data):
        """
        Handles a text HTML node in the tree
        :param data: the text node
        :return:
        """
        last_opened_tag = self.get_last_opened_tag()
        if last_opened_tag in self.allowed_tags:
            data = data.replace('  ', ' ').strip()
            if data != '':
                self.block_content += data + ' '

    def get_text(self):
        return "\n\n".join(self.blocks)

if __name__ == '__main__':
    pdb.set_trace()
