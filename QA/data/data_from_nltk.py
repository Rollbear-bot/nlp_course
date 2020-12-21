# -*- coding: utf-8 -*-
# @Time: 2020/12/14 8:24
# @Author: Rollbear
# @Filename: data_from_nltk.py

from nltk.corpus import brown, gutenberg, webtext, reuters

from util import timer


@timer(True)
def get_data():
    data = list(brown.sents()) + list(gutenberg.sents()) + \
           list(webtext.sents()) + list(reuters.sents())
    # data = list(brown.sents())
    return data


if __name__ == '__main__':
    get_data()
