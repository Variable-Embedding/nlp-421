"""A list of constants used for the workflow.
"""
import os
from os.path import dirname

WORKFLOW_ROOT = dirname(dirname(dirname(__file__)))
DATA_FOLDER = os.sep.join([WORKFLOW_ROOT, 'data'])
EMBEDDING_FOLDER = os.sep.join([DATA_FOLDER, 'embeddings'])
CORPRA_FOLDER = os.sep.join([DATA_FOLDER, 'corpra'])
LOGGING_PATH = os.sep.join([WORKFLOW_ROOT, 'logs'])

ALL_FOLDERS = [DATA_FOLDER, EMBEDDING_FOLDER, CORPRA_FOLDER, LOGGING_PATH]

GLOVE_TWITTER = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
GLOVE_COMMON_CRAWL_840B = "http://nlp.stanford.edu/data/glove.840B.300d.zip"

ALL_GLOVE = [GLOVE_TWITTER, GLOVE_COMMON_CRAWL_840B]
