#!/usr/bin/python

'''

Manipulates the 20newsgroup dataset for use
in a document classification task.

'''

import util as util
from sklearn.datasets import fetch_20newsgroups

def categorize_20newsgroup(base_dir):
    corpus = \
    fetch_20newsgroups(subset='train',
                       remove=('headers', 'footers', 'quotes'))

    util.make_dir(base_dir)

    for cat in range(20):
        util.make_dir('{}/{}'.format(base_dir, 'c' + str(cat)))

    for index in range(len(corpus.data)):
        new = '{}/{}/{}'.format(base_dir,
                                'c' + str(list(corpus.target)[index]),
                                str(index) + '.txt')

        util.write_file(new, corpus.data[index].encode())

    return True

if __name__ == '__main__':
    categorize_20newsgroup('20newsgroup')
