from src.util.get_gutenberg import get_gutenberg, simple_vocabulary
import string
import time

def stage_prep_corpus(is_gutenberg=True):
    """
    Manage all functions to process a target corpus, return a vocabulary for now.
    :return: A list of words representing all the unique words in a corpus, i.e. the vocabulary.
    """

    if is_gutenberg:
        #TODO: Set up method to get all book links here
        book_urls = [
                     # the top 20
                     "https://www.gutenberg.org/files/1342/1342-0.txt"
                   , "https://www.gutenberg.org/files/64317/64317-0.txt"
                   , "https://www.gutenberg.org/files/11/11-0.txt"
                   , "https://www.gutenberg.org/files/98/98-0.txt"
                   , "https://www.gutenberg.org/files/1661/1661-0.txt"
                   , "https://www.gutenberg.org/files/1080/1080-0.txt"
                   , "https://www.gutenberg.org/files/65053/65053-0.txt"
                   , "https://www.gutenberg.org/files/2701/2701-0.txt"
                   , "https://www.gutenberg.org/files/844/844-0.txt"
                   , "https://www.gutenberg.org/files/5200/5200-0.txt"
                   , "https://www.gutenberg.org/files/174/174-0.txt"
                   , "https://www.gutenberg.org/files/2542/2542-0.txt"
                   , "https://www.gutenberg.org/files/1260/1260-0.txt"
                   , "https://www.gutenberg.org/files/345/345-0.txt"
                   , "https://www.gutenberg.org/files/76/76-0.txt"
                   , "https://www.gutenberg.org/files/1952/1952-0.txt"
                   , "https://www.gutenberg.org/files/219/219-0.txt"
                   , "https://www.gutenberg.org/files/1400/1400-0.txt"
                   , "https://www.gutenberg.org/files/46/46-0.txt"
                   # the bottom 20
                   , "https://www.gutenberg.org/cache/epub/34901/pg34901.txt"
                   , "https://www.gutenberg.org/files/2097/2097-0.txt"
                   , "https://www.gutenberg.org/files/786/786-0.txt"
                   , "https://www.gutenberg.org/cache/epub/14838/pg14838.txt"
                   , "https://www.gutenberg.org/files/863/863-0.txt"
                   , "https://www.gutenberg.org/files/4517/4517-0.txt"
                   , "https://www.gutenberg.org/cache/epub/61/pg61.txt"
                   , "https://www.gutenberg.org/cache/epub/42324/pg42324.txt"
                   , "https://www.gutenberg.org/cache/epub/779/pg779.txt"
                   , "https://www.gutenberg.org/files/41/41-0.txt"
                   , "https://www.gutenberg.org/files/6133/6133-0.txt"
                   , "https://www.gutenberg.org/cache/epub/22120/pg22120.txt"
                   , "https://www.gutenberg.org/cache/epub/105/pg105.txt"
                   , "https://www.gutenberg.org/cache/epub/10007/pg10007.txt"
                   , "https://www.gutenberg.org/files/1399/1399-0.txt"
                   , "https://www.gutenberg.org/cache/epub/4363/pg4363.txt"
                   , "https://www.gutenberg.org/files/521/521-0.txt"
                   ,  "https://www.gutenberg.org/cache/epub/852/pg852.txt"
                    # some others, arbitrary selection
                   , "https://www.gutenberg.org/cache/epub/514/pg514.txt"
                   , "https://www.gutenberg.org/files/1184/1184-0.txt"
                     ]

        vocabulary = []

        for book_url in book_urls:
            book = get_gutenberg(book_url)
            book_vocab = simple_vocabulary(book)
            book_vocab.sort()
            vocabulary.extend(book_vocab)

        # return only the unique instances of words in the corpus
        vocabulary = list(set(vocabulary))
        # include all forms of punctuation as part of the vocabulary
        vocabulary.extend(string.punctuation)

        return vocabulary
