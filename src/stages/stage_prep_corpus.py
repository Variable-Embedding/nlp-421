from src.util.get_gutenberg import get_gutenberg, simple_vocabulary


def stage_prep_corpus(is_gutenberg=True):
    """
    Manage all functions to process a target corpus, return a vocabulary for now.
    :return: A list of words representing all the unique words in a corpus, i.e. the vocabulary.
    """

    book_urls = ["https://www.gutenberg.org/files/1342/1342-0.txt", "https://www.gutenberg.org/files/64317/64317-0.txt"]
    books = []
    vocabulary = []

    for book_url in book_urls:
        book = get_gutenberg(book_url)
        book_vocab = simple_vocabulary(book)
        vocabulary.extend(book_vocab)

    vocabulary = list(set(vocabulary))

    return vocabulary
