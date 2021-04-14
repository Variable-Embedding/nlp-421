
from src.util.constants import *
import logging

from torchtext.datasets.wikitext2 import WikiText2


def get_torch_text(corpus_type):
    """
    A function wrapper to get corpra with torch text.
    :param corpus_type: string, required. Any of available torch text dataset types.
    :return:
    """
    torch_text_name = 'torch_text'
    torch_text_folder = f'{torch_text_name}_{corpus_type}'
    torch_text_path = os.sep.join([CORPRA_FOLDER, torch_text_folder])

    if corpus_type == "WikiText2":
        logging.info(f'Using TorchText to download {corpus_type} to {torch_text_path}')
        corpus = WikiText2(root=torch_text_path)
        print(len(corpus))
