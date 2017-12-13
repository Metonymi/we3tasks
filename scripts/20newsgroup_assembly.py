#!/usr/bin/python

__author__ = 'Thomas Stearns'

'''

Manipulates the 20newsgroup dataset for use
in a document classification task.

'''

import os
from sklearn.datasets import fetch_20newsgroups

def categorize_20newsgroup(base_dir):
    corpus = \
    fetch_20newsgroups(subset='train',
                       remove=('headers', 'footers', 'quotes'))

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    for cat in range(20):
        if not os.path.exists('{}/{}'.format(base_dir, 'c'+ str(cat))):
            os.mkdir('{}/{}'.format(base_dir, 'c' + str(cat)))

    for index in range(len(corpus.data)):
        new = '{}/{}/{}'.format(base_dir,
                                'c' + str(list(corpus.target)[index]),
                                str(index) + '.txt')

        with open(new, 'wb') as new_file:
            new_file.write(corpus.data[index].encode())

    return True

if __name__ == '__main__':
    categorize_20newsgroup('20newsgroup')
