import requests
from bs4 import BeautifulSoup
from time import sleep
from src.util.constants import *
import logging
import string


def get_gutenberg(url):
    logging.info('Starting project gutenberg data prep.')

    file_name = url.rsplit('/', 1)[-1]
    file_path = os.sep.join([CORPRA_FOLDER, file_name])

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
        return read_book_utf(file_path)


def simple_vocabulary(book):
    book_vocab = book['bookdata'].split(' ')
    # reference: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    book_vocab = [i.translate(str.maketrans('', '', string.punctuation)).lower() for i in book_vocab]
    book_vocab = list(set(book_vocab))
    return book_vocab


def read_book_utf(file_path):
    f = open(file_path, "r")

    metadata = []
    metadata_flag = True
    bookdata = []
    bookdata_flag = False
    disclaimers = []
    disclaimers_flag = False

    for line in f:
        if not line.startswith("**") and metadata_flag:
            metadata.append(line)
        else:
            metadata_flag = False

        if not metadata_flag and "Chapter" not in line and len(line) > 1:
            if 'CONTENTS' in line:
                bookdata_flag = True
            if line.startswith("***"):
                bookdata_flag = False
                disclaimers_flag = True
            if bookdata_flag:
                bookdata.append(line)
            if disclaimers_flag:
                disclaimers.append(line)
    f.close()

    metadata = [i.strip() for i in metadata]
    bookdata = ' '.join([i.strip() for i in bookdata][1:])
    disclaimers = [i.strip() for i in disclaimers]

    book = {'metadata': metadata
        , 'bookdata': bookdata
        , 'disclaimers': disclaimers}

    return book
