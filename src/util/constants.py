"""A list of constants used for the workflow.
"""
import os
from os.path import dirname

WORKFLOW_ROOT = dirname(dirname(dirname(__file__)))
DATA_FOLDER = os.sep.join([os.environ['PWD'], 'data'])
EMBEDDING_FOLDER = os.sep.join([DATA_FOLDER, 'embeddings'])
LOGGING_PATH = os.sep.join([os.environ['PWD'], 'logs'])

GLOVE_TWITTER = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"


