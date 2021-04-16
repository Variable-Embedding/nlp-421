import requests, zipfile, io
import os
import logging
from pathlib import Path
from src.util.constants import *
import pickle
import time
import numpy as np
import gc
from torchnlp.word_to_vector import GloVe
import torch
import mmap
from tqdm import tqdm
import torch.nn as nn
from src.util.spinning_cursor import Spinner

def random_embedding_vector(embedding_dim, scale=0.6):
    """A helper function to return a randomized embedding space of dimension embedding_dim

    :param embedding_dim: integer, usually 300 or one of 200, 100, 50, 25, depending on embedding space.
    :param scale: stdev of distribution for np.random.normal function
    :return: a randumized numpy array to fill an embedding vector
    """
    return np.random.normal(scale=scale, size=(embedding_dim,))


def prep_nn_embeddings(vectors, non_trainable=False):
    """A helper function to return pytorch nn embedding layer.

    :param vectors: weight matrix of pre-trained or randomized vectors
    :param non_trainable: bool, default to False. If False, keep static.
    :return: torch sparse embedding layer, number of embeddings, and number of embedding dims

    source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    """

    num_embeddings, embedding_dim = vectors.size()

    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': vectors})

    if non_trainable:
        emb_layer.weight.requires_grad = False
    else:
        emb_layer.weight.requires_grad = True

    logging.info(f'Prepared embedding layer for pytorch nn, set trainable to {non_trainable}, '
                 f'to switch this, set prep_nn_embeddings(non_trainable=True).'
                 f'\n returning emb_layer ({type(emb_layer)}), num_embeddings ({num_embeddings})'
                 f', and embedding_dim ({embedding_dim}).')

    return emb_layer, num_embeddings, embedding_dim


def prep_corpus_embeddings(word2idx, vectors, target_vocab, **kwargs):
    """Consume pre-trained embedding with vocab of target corpus, return embedding layer for training.

    :return: an embedding space t hat has words from the target vocabulary, if they exist, or
    initialize random embedding for new words from the target corpus

    source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    """

    matrix_len = len(target_vocab)
    embedding_dim = vectors.size()[1]

    vectors_new = torch.zeros((matrix_len, embedding_dim))
    words_found = 0
    word2idx_new = {}
    idx2word_new = {}

    for i, word in enumerate(target_vocab):
        try:
            vectors_new[i] = vectors[word2idx[word]]
            words_found += 1
        except KeyError:
            vectors_new[i] = torch.from_numpy(random_embedding_vector(embedding_dim=embedding_dim))

        word2idx_new.update({word: i})
        idx2word_new.update({i: word})

    remainder = len(target_vocab) - words_found

    logging.info('Out of {} total words, found {} words in the pre-trained model,'
                 ' {} words initialized randomly.'.format(len(target_vocab), words_found, remainder))
    logging.info('Returning {} vocabulary embeddings for nn training.'.format(len(word2idx_new)))

    # embeddings = {"word2idx": word2idx_new, "idx2word": idx2word_new, "vectors": vectors_new}

    return word2idx_new, idx2word_new, vectors_new


def get_embeddings(glove_embeddings):
    """Return pre-trained embedding data to the neural network pipeline.

    :param glove_embeddings: A dict of GloVe embeddings
    :return: Return a 3-Tuple of GloVe embeddings :
             word2idx, dict, {word:int}
             idx2word, dict, {int:word}
             vectors, torch tensor of torch tensors, [[tensor],[tensor]]
    """
    # Initialize data structs
    words = []
    idx = 0
    word2idx = {}
    idx2word = {}
    # TODO: vectors will need to become array/torch object of array/torch objects.
    vectors = []

    for word, vector in glove_embeddings.items():
        words.append(word)
        word2idx[word] = idx
        idx2word[idx] = word
        vectors.append(vector)
        idx += 1

    vectors = torch.stack([torch.from_numpy(item).float() for item in vectors])

    return word2idx, idx2word, vectors


def get_torch_glove(torch_glove_type):
    """A helper function to user torchnlp built-in glove getter.

    :param torch_glove_type: a string, name of GloVe embedding
    :return: bool, whether the get operation workd.
    """
    logging.info(f'Downloading GloVe vectors from TorchNLP for {torch_glove_type}')
    # set path for download (cache in torchnlp)
    period = "."
    underscore = "_"
    if period in torch_glove_type:
        torch_glove_path = torch_glove_type.replace(period, underscore)
    else:
        torch_glove_path = torch_glove_type

    torch_glove_folder = os.sep.join([EMBEDDING_FOLDER, f'torch_glove_{torch_glove_path}'])
    # run torchnlp method for GloVe download
    directories = []

    if os.path.exists(torch_glove_folder):
        directories = os.listdir(torch_glove_folder)

    if len(directories) == 0:
        GloVe(name=torch_glove_type, cache=torch_glove_folder)
        directories = os.listdir(torch_glove_folder)

    if len(directories) > 0:
        directories = [x for x in directories if not x.endswith('.pt') and not x.endswith('.zip')]
        write_pickle(directories, torch_glove_folder)

    if directories:
        return True
    else:
        return False


def download_embeddings(url, unzip_path):
    """Get and process GloVe embeddings.

    :param url: The target URL, see constants.py for urls to GloVe downloads.
    :param unzip_path: The full path to unzip GloVe downloads to.
    :return: bool, return True if everything runs
    """

    directory = os.listdir(unzip_path)

    if len(directory) > 0 and any(".txt" in i for i in directory):
        logging.info(f'Directory Not Empty {unzip_path}.')
        logging.info(f'Current files {directory}')
        logging.info(f'Skipping download from URL: {url} - to force download, delete or rename files in this directory.')

        write_pickle(directory, unzip_path)

    else:
        # source: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
        if not any(".zip" in i for i in directory):
            logging.info(f'Starting download from {url}. This may take a while.')

            r = requests.get(url, stream=True, allow_redirects=True)
            total_size = int(r.headers.get('content-length'), 0)
            block_size = 1024
            file = url.split('/')[-1]
            file_path = os.sep.join([unzip_path, file])
            pbar = tqdm(total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {file}")

            with open(file_path, 'wb') as f:
                for data in r.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)

            pbar.close()

            if total_size != 0 and pbar.n != total_size:
                logging.warning(f'ERROR in Download for {url}, byte size does not match expected size.')

            directory = os.listdir(unzip_path)

        logging.info(f'Found embedding files on local, unzipping {directory}.')

        for embedding_file in directory:
            if embedding_file.endswith('.zip'):
                z = zipfile.ZipFile(os.sep.join([unzip_path, embedding_file]))
                z.extractall(unzip_path)
        # prior method (simpler but no progress)
        # r = requests.get(url)
        # z = zipfile.ZipFile(io.BytesIO(r.content))
        # z.extractall(unzip_path)
        directory = os.listdir(unzip_path)
        logging.info(f'Downloaded {directory} to {unzip_path}')

        write_pickle(directory, unzip_path)

    logging.info('download_embeddings() Complete')

    return True


def write_pickle(directory, unzip_path):
    """Write dictionaries to pickled files if the provided directory does not already have pickle files.

    :param directory: A dictionary of word embeddings.
    :param unzip_path: Full path to file locations.
    :return: None, writes data to disk.
    """

    if any([".pickle" in i for i in directory]):
        logging.info(f'Pickle files detected, skipping conversion from .txt to .pickle format.')

    else:
        logging.info(f'Converting .txt embedding files to .pickle objects for faster IO.')

        for text_file in directory:
            text_file_path = os.sep.join([unzip_path, text_file])

            if text_file_path.endswith(".txt"):

                embedding_dict = parse_embedding_txt(text_file_path)

                logging.info(f'Parsing embedding files as pickles. This may take a while.')


                pickle_file = f'{Path(text_file).stem}.pickle'
                pickle_file_path = os.sep.join([unzip_path, pickle_file])

                # TODO: Add a spinning pbar or something here
                f = open(pickle_file_path, "wb")
                pickle.dump(embedding_dict, f, protocol=2)
                f.close()

                # test read pickle
                parse_embedding_pickle(pickle_file_path)

                logging.info(f'Wrote embedding pickle to {pickle_file_path}.')


def parse_embedding_pickle(embedding_file_path):
    """Read pickled embedding files from disk.

    :param embedding_file_path: string, full path to the embedding txt file
    :return: a dictionary of embeddings k: word (string), v: embedding vector of float32s (numpy array)
    """
    logging.info(f'Reading embedding file from {embedding_file_path}.')

    if ".pickle" in embedding_file_path:
        start_time = time.time()

        with open(embedding_file_path, "rb") as handle:
            embedding_dict = pickle.load(handle)

        end_time = time.time()
        logging.info(f'PICKLE Read Time is {end_time - start_time}')
    else:
        logging.info(f'Did not read file, please indicate a file ending in .pickle.')

    return embedding_dict


def parse_embedding_txt(embedding_file_path):
    """Read text embedding files from disk.

    :param embedding_file_path: string, full path to the embedding txt file
    :return: a dictionary of embeddings k: word (string), v: embedding vector of float32s (numpy array)
    """
    logging.info(f'Parsing word embeddings from {embedding_file_path}')
    embedding_dict = {}
    embedding_dim = 300 if "300d" in embedding_file_path else 200 if "200d" in embedding_file_path else 100 if "100d" in embedding_file_path else 50 if "50d" in embedding_file_path else 25

    start_time = time.time()
    with open(embedding_file_path, 'r', encoding="utf-8") as f:
        # for line in f:
        for line in tqdm(f, total=get_num_lines(embedding_file_path), desc='Opening Text File'):
            word, vector = read_line(line, embedding_dim=embedding_dim)
            embedding_dict.update({word: vector})
    end_time = time.time()
    logging.info(f'TEXT Read Time is {end_time-start_time}')

    return embedding_dict


def get_num_lines(file_path):
    """A helper function to count number of lines in a given text file.

    :param file_path: full path to some .txt file.
    :return: integer, count of lines in a .txt file.

    reference: https://blog.nelsonliu.me/2016/07/30/progress-bars-for-python-file-reading-with-tqdm/
    """

    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def read_line(line, embedding_dim):
    """Read lines in a text file from embeddings.

    :param line: Each line of a text open object.
    :param embedding_dim: the expected vector dimension
    :return: 2-tuple of word (string) and numpy array (vector).
    """
    first = 0
    rest = 1
    values = line.split()
    # the first element is assumed to be the word
    word = values[first]

    # catch cases where first n strings are repeating as the word
    try:
        # the rest of list is usually the vector but sometimes it is not
        vector = np.asarray(values[rest:], "float32")

    except ValueError:
        #TODO: words such as ... or -0.0033421 are truncating the vector space from 300 to less than 300,
        # impacting about a dozen entries, provide random vector for now, fix later
        word, rest = return_repeating_word(values)
        # vector = np.asarray(values[rest:], "float32")
        vector = random_embedding_vector(embedding_dim=embedding_dim)

    if len(vector) != embedding_dim:
        vector = random_embedding_vector(embedding_dim=embedding_dim)

    return tuple((word, vector))


def return_repeating_word(values):
    """A helper function for read_line().

    Address issues where word has repeating chargers, return them as a single word.

    :param values: values, a line of embedding text data
    :return: A string of repeating characters.
    """
    word = []
    first_char = values[0]
    counter = 1
    word.append(first_char)

    # while values:
    for idx, char in enumerate(values[1:]):
        counter += 1
        curr_char = char
        if curr_char == first_char:
            word.append(curr_char)
        else:
            break

    word = ''.join(map(str, word))

    return word, counter




