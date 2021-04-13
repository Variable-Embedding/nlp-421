import requests
from bs4 import BeautifulSoup
from time import sleep
from src.util.constants import *
import logging
import string
import time
import re

from copy import deepcopy


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
    # book_vocab = [i.translate(str.maketrans('', '', string.punctuation)).lower() for i in book_vocab]
    book_corpus = [i.lower() for i in book_vocab if i != ""]
    book_vocab = list(set(book_corpus))
    return book_vocab, book_corpus


def read_book_utf(file_path):
    f = open(file_path, "r")

    preamble = []
    preamble_flag = True
    bookdata = []
    bookdata.extend(["<book_start>", "<title>"])
    bookdata_flag = False
    disclaimers = []
    disclaimers_flag = False

    for line in f:
        if preamble_flag:
            preamble.append(line.strip())

        if bookdata_flag:
            bookdata.append(line.strip())

        if line.startswith("*** "):
            preamble_flag = False
            bookdata_flag = True

    preamble = ' '.join(preamble)
    bookdata_string = ' '.join(bookdata)

    meta_info_brackets = re.findall(r'\[.*?]', bookdata_string)
    meta_info_brackets.extend(["", "CHAPTER"])
    if len(meta_info_brackets) > 0:
        bookdata = [item for item in bookdata if item not in meta_info_brackets]

    story_flag = False
    remove_flags = ["<meta_content>"]
    bookdata_temp = deepcopy(bookdata)
    content_flags = ["Gutenberg", "Edited by ", "From The Quarto of", "(or any other work associate", "FOUND AMONG THE PAPERS OF THE", "The Walter Scott Publishing", "distribution of electronic works", "00", "PROLOGUE", "To protect the Project Gutenberg-tm mission", "With an Introduction by ", "PLEASE READ THIS BEFORE YOU DISTRIBUTE", "The following is a reprint", "PART ", "Table of Contents", "THE FULL PROJECT GUTENBERG LICENSE", "Transcribed by", "Distributed Proofreading Team", "TRANSCRIBE", "Transcriber's Note:", "Book the ", "THE PREFACE", "Original Transcriber’s", "Epilogue", "THE MILLENNIUM FULCRUM EDITION 3.0", "PRODUCED BY", "Produced by", "CONTENT", "Content", "THERE IS", "ETYMOLOGY", "EXTRACTS", "Translated by", "CHAPTER ", "Chapter ", 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']

    for idx, section in enumerate(bookdata):

        if section.startswith(tuple(content_flags)):
            contents_flag = True
        else:
            contents_flag = False

        if contents_flag and not story_flag:
            bookdata_temp[idx] = "<meta_content>"

        if section == 'cover' and not story_flag:
            bookdata_temp[idx] = "<meta_content>"

        if section == "I" and bookdata[idx-1] == "Table of Contents":
            bookdata_temp[idx] = "<meta_content>"

        if section.startswith("by") or section.startswith("By") or section.startswith("_by_") or section.startswith("BY"):
            bookdata[idx] = "<author>"

        if section.startswith("***") or section.startswith("End of the Project Gutenberg EBook"):
            disclaimers_flag = True

        if disclaimers_flag:
            bookdata_temp[idx] = "<meta_content>"
            disclaimers.append(line)

    title_flag = True

    for idx, section in enumerate(bookdata):

        if title_flag and section.startswith("<author>"):
            bookdata_temp[idx] = "<author>"
            bookdata_temp.insert(idx, "</title>")
            title_flag = False

    bookdata_temp = [i for i in bookdata_temp if i not in remove_flags]
    bookdata_temp.append("<book_end>")
    bookdata_string = ' '.join(bookdata_temp)

    punct_replacement = {".": " <punctuation> <period> "
                       , ",": " <punctuation> <comma> "
                       , ";": " <punctuation> <semicolon> "
                       , "!": " <punctuation> <exclamation> "
                       , "?": " <punctuation> <question> "
                       , "\"": " <punctuation> <double_quote> "
                       , "\'": " <punctuation> <single_quote> "
                       , ":": " <punctuation> <colon> "
                       , "--": " <punctuation> <double_dash> "
                       , "-": " <punctuation> <single_dash> "
                       , "...": " <punctuation> <ellipsis> "
                       ,  "#": " <punctuation> <hash_tag> "
                       , "(": " <punctuation> <left_paren> "
                       , ")": " <punctuation> <right_paren> "
                       , "'t": " <punctuation> <apostrophe> t "
                       , "'s": " <punctuation> <apostrophe> s "
                         }

    rep = dict((re.escape(k), v) for k, v in punct_replacement.items())
    pattern = re.compile("|".join(rep.keys()))
    bookdata = pattern.sub(lambda m: rep[re.escape(m.group(0))], bookdata_string)

    disclaimers = ' '.join(disclaimers)

    book = {'preamble': preamble
        , 'bookdata': bookdata
        , 'disclaimers': disclaimers}

    return book
