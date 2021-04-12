import requests
from bs4 import BeautifulSoup
from time import sleep
from src.util.constants import *
import logging
import string
import time

def get_gutenberg(url):
    logging.info('Starting project gutenberg data prep.')

    file_name = url.rsplit('/', 1)[-1]
    gutenberg_folder = os.sep.join([CORPRA_FOLDER, 'gutenberg'])
    file_path = os.sep.join([gutenberg_folder, file_name])

    if not os.path.exists(gutenberg_folder):
        os.makedirs(gutenberg_folder)

    if os.path.exists(file_path):
        logging.info(f'Found existing book txt data at {file_path}.')
        return read_book_utf(file_path)

    else:
        logging.info(f'Downloading book data from Project Gutenberg from {url}.')
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')

        with open(file_path, 'w') as f:
            f.write("%s \n" % soup)
        logging.info(f'Wrote book to {file_path}.')

        logging.info('Waiting 5 seconds between URL get next book.')
        time.sleep(5)
        return read_book_utf(file_path)


def simple_vocabulary(book):
    book_vocab = book['bookdata'].split(' ')
    # reference: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    book_vocab = [i.translate(str.maketrans('', '', string.punctuation)).lower() for i in book_vocab]
    book_vocab = list(set(book_vocab))
    return book_vocab


def read_book_utf(file_path):
    f = open(file_path, "r")

    preamble = []
    preamble_flag = True
    bookdata = []
    bookdata_flag = False
    disclaimers = []
    disclaimers_flag = False

    for line in f:
        if not line.startswith("**") and preamble_flag:
            preamble.append(line)
        else:
            preamble_flag = False

        if not preamble_flag:
            bookdata_flag = True
            if line.startswith("***"):
                bookdata_flag = False
                disclaimers_flag = True
            if bookdata_flag:
                bookdata.append(line)
            if disclaimers_flag:
                disclaimers.append(line)
    f.close()

    preamble = [i.strip() for i in preamble]
    bookdata = ' '.join([i.strip() for i in bookdata])
    disclaimers = [i.strip() for i in disclaimers]

    print(bookdata[:50])


    book = {'preamble': preamble
        , 'bookdata': bookdata
        , 'disclaimers': disclaimers}

    return book
