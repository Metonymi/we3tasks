#!/usr/bin/python

'''

Like the 20 newsgroup script, this manipulates the Reuters
dataset for use in a simple classification task.

Unlike 20NG, this dataset will be purposefully imbalanced.

'''

import util as util
from nltk.corpus import reuters

def categorize_reuters():
    '''
    Parses dataset to only examine documents associated
    with a single category.
    '''
    categories = {}
    for file_id in reuters.fileids():
        if len(reuters.categories(file_id)) == 1:
            cat = reuters.categories(file_id)[0]
            if cat not in categories.keys():
                categories[cat] = {}

        text = reuters.raw(file_id)
        categories[cat][file_id.replace('/', '_')] = text

    return categories

def write_data(base_dir, categories):
    '''
    Writes the text data in according to the
    above defined category document structure.
    '''
    util.make_dir(base_dir)
    for cat in categories:
        util.make_dir(base_dir + '/' + cat)
        for doc in categories[cat]:
            new = '{}/{}/{}'.format(base_dir, cat, doc + '.txt')
            util.write_file(new, categories[cat][doc].encode())

    return True

if __name__ == '__main__':
    categories = categorize_reuters()
    write_data('reuters', categories)
