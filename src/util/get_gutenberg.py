import requests
from bs4 import BeautifulSoup
from time import sleep
from src.util.constants import *

def get_gutenberg():
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    file_name = "1342"
    file_path = os.sep.join([CORPRA_FOLDER, file_name])


    # with open(file_path, 'w') as f:
    #     f.write("%s \n" % story)
