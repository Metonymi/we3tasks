#!/usr/bin/python

'''

This downloads and manipulates the sentiment analysis
dataset found here

http://ai.stanford.edu/~amaas/data/sentiment/

for use in a sentiment analysis task.

'''

import os
import tarfile
import requests

def download_and_untar_file(url, path):
    request = requests.get(url)
    with open(path, 'wb') as new_file:
        new_file.write(request.content)

    tar = tarfile.open(path, 'r:gz')
    tar.extractall()
    tar.close()

    return True

if __name__ == '__main__':
    download_and_untar_file('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                            os.getcwd() + '/aclImdb_v1.tar.gz')
