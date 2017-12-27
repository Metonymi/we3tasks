#!/usr/bin/python

'''
Utils for we3tasks scripts

'''

import os

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

    return True

def write_file(path, string):
    with open(path, 'wb') as new_file:
        new_file.write(string)

    return True
