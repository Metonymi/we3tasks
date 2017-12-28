#!/usr/bin/python

'''
Import and run assembly scripts for all three datasets.

'''

import os
import nltk
from src import newsgroup_assembly as nga
from src import aclimdb_assembly as imdba
from src import reuters_assembly as ra
from src import util as util

def main():
    # gets reuters from nltk
    nltk.download('reuters')
    categories = ra.categorize_reuters()
    ra.write_data('reuters', categories)

    nga.categorize_20newsgroup('20newsgroup')

    imdba.download_and_untar_file('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                                  os.getcwd() + '/aclImdb_v1.tar.gz')
    return True

if __name__ == '__main__':
    main()
