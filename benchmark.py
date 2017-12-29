#!/usr/bin/python

import os
import warnings
import torchfile
from sklearn.linear_model import LogisticRegression as lr
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
import numpy as np

warnings.simplefilter('ignore')

class Benchmark(object):
    '''
    This class assumes the data elements (.txt files)
    exist in subfolders of the data_folder argument, each
    of which corresponds to a single class label.

    Those folders can be created by running the main_assembly.py script.
    '''
    def __init__(self, data_folder, feature_file, label_folders=False):
        self.data_folder = data_folder
        self.features = torchfile.load(feature_file)
        if label_folders:
            self.label_folders = label_folders
        else:
            self.label_folders = os.listdir(data_folder)

        self.corpus = {
            label_folder: {
                'reps': {},
                'text_files': {}
            } for label_folder in self.label_folders
        }

        for label_folder in self.label_folders:
            self.read_label_folder(label_folder)
            self.add_reps(label_folder)

        self.flatten()
        self.train_model()
        self.evaluate_model()

    def make_labels(self, index, folder):
        '''
        Creates a label vecots for a given folder.
        '''
        if index == 0:
            self.labels = \
            np.zeros((len(self.features[folder.encode()]), 1))
        else:
            self.labels = \
            np.vstack((self.labels,
                       index * np.ones((len(self.features[folder.encode()]), 1))))
        return True

    def add_reps(self, folder):
        '''
        Adds Metonymi representations to the corpus from the downloaded file.
        '''
        for text_file in list(self.corpus[folder]['text_files'].keys()):
            if text_file.encode() in list(self.features[folder.encode()].keys()):
                self.corpus[folder]['reps'][text_file] = \
                self.features[folder.encode()][text_file.encode()] \
                .reshape((1, 2048)) #Each Metonymi representation is 2048 dimensional

        return True

    def read_label_folder(self, label_folder):
        '''
        Reads text files from a given label folder to make
        human readable dict to corpus.
        '''
        path = '{}/{}'.format(self.data_folder, label_folder)
        for text_file in os.listdir(path):
            my_file = open('{}/{}'.format(path, text_file), 'rb')
            text = my_file.read()
            self.corpus[label_folder]['text_files'][text_file] = text
            my_file.close()

        return True

    def flatten(self):
        '''
        Flattens the dictionaries into a single array for
        logistic regression training.
        '''
        print('FLATTENING %s DATA...' % self.data_folder)
        label_counter = 0
        for label_folder in list(self.corpus.keys()):
            if label_counter == 0:
                self.frame = \
                np.vstack(list(self.corpus[label_folder]['reps'].values()))
            else:
                self.frame = \
                np.vstack((self.frame,
                           np.vstack(list(self.corpus[label_folder]['reps'].values()))))

            self.make_labels(label_counter, label_folder)
            label_counter += 1

        self.examples = len(self.frame)
        self.frame = np.hstack((self.frame, self.labels))
        print('DONE!')

        return True

    def train_model(self):
        '''
        Trains simple logistic regression using the class labels.
        No regularization. The Metonymi features do all of the heavy lifting!
        '''
        print('TRAINING MODEL...')
        labels = self.frame[:, -1]
        frame = scale(self.frame[:, :-1])
        self.train, self.test, self.train_labels, self.test_labels = \
        tts(frame, labels, random_state=26, test_size=.15)
        self.model = lr(max_iter=200)
        self.model.fit(self.train, self.train_labels)
        print('DONE!\n')

        return True

    def evaluate_model(self):
        '''
        Evaluates trained model on test set.
        '''
        preds = self.model.predict(self.test)
        print('CONFUSION MATRIX FOR %s:' % self.data_folder)
        print(metrics.confusion_matrix(self.test_labels, preds))
        print('\n')
        print('CLASSIFICATION RATE FOR %s:' % self.data_folder)
        print(str(100 * round(metrics.accuracy_score(self.test_labels, preds), 3)) + '%')
        print('\n')

        return True

def main():
    Benchmark('aclImdb', 'we3tasks_features/imdb_data_features', ['train/pos', 'train/neg'])
    Benchmark('20newsgroup', 'we3tasks_features/20newsgroup_data_features')
    Benchmark('reuters', 'we3tasks_features/reuters_data_features')

    return True

if __name__ == '__main__':
    main()
