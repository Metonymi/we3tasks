#!/usr/bin/python

__author__ = 'Thomas Stearns'

import os
import torchfile
from sklearn.linear_model import LogisticRegression as lr
from sklearn.preprocessing import scale
from sklearn import metrics
import numpy as np

class Benchmark(object):
    '''
    This class assumes the data elements (.txt files)
    exist in subfolders of the data_folder argument, each
    of which corresponds to a single class label.

    Those folders can be created by running the *_assembly.py scripts.
    '''
    def __init__(self, data_folder, feature_file):
        self.data_folder = data_folder
        self.features = torchfile.load(feature_file)
        self.label_folders = os.listdir(data_folder)

        self.corpus = {
            label_folder: {
                'representations': {},
                'text_files': {}
            } for label_folder in self.label_folders
        }

        for label_folder in self.label_folders:
            self.read_label_folder(label_folder)
            self.add_representations(label_folder)

    def make_labels(self, index, folder):
        if index == 0:
            self.labels = \
            np.zeros((len(self.corpus[folder]['text_files']), 1))
        else:
            self.labels = \
            np.vstack((self.labels,
                       index * np.ones((len(self.corpus[folder]['text_files']), 1))))
        return True

    def add_representations(self, folder):
        folder = folder.encode()
        for text_file in list(self.corpus[folder]['text_files'].keys()):
            text_file = text_file.encode()
            if text_file in list(self.features[folder].keys()):
                self.corpus[folder]['representations'][text_file] = \
                np.hstack((self.corpus[folder]['representations'][text_file],
                           self.features[folder][text_file].reshape((1, 2048))))
        return True

    def flatten(self):
        label_counter = 0
        for label_folder in list(self.corpus.keys()):
            if label_counter == 0:
                self.frame = \
                np.vstack(list(self.corpus[label_folder]['representations'].values()))
            else:
                self.frame = \
                np.vstack((self.frame,
                           np.vstack(list(self.corpus[label_folder]['representations'].values()))))

            self.make_labels(label_counter, label_folder)
            label_counter += 1

        self.examples = len(self.frame)
        self.frame = np.hstack((self.frame, self.labels))
        np.random.shuffle(self.frame)

        return True

    def read_label_folder(self, label_folder):
        path = '{}/{}'.format(self.data_folder, label_folder)
        for text_file in os.listdir(path):
            my_file = open('{}/{}'.format(path, text_file), 'rb')
            text = my_file.read()
            self.corpus[label_folder]['text_files'][text_file] = text
            my_file.close()

        return True

    def train_model(self):
        self.labels = self.frame[:, -1]
        self.frame = self.frame[:, :-1]
        self.frame = scale(self.frame)
        self.model = lr()

        self.model.fit(self.frame[:int(self.examples * .85), :],
                       self.labels[:int(self.examples * .85)])

        return True

    def evaluate_model(self):
        labels = self.labels[int(self.examples * .85):]
        data = self.model.predict(self.frame[int(self.examples * .85):, :])

        print(metrics.confusion_matrix(labels, data))
        print(metrics.accuracy_score(labels, data))

        return True
'''
work = Benchmark('reuters/', 'reuters_data_features10')
work.flatten()
work.train_model('logistic_regression')
work.evaluate_model()
'''
benchmarking_work = Benchmark('20newsgroup/', '20newsgroup_data_features10')
benchmarking_work.flatten()
benchmarking_work.train_model()
benchmarking_work.evaluate_model()
'''
work = Benchmark('aclImdb/', 'imdb_data_features10')
work.vectorize()
work.flatten()
work.train_model('logistic_regression')
work.evaluate_model()

'''
