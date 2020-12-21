# -*- coding: utf-8 -*-
# @Time: 2020/12/14 8:14
# @Author: Rollbear
# @Filename: hello_QA.py

from QA.data.data_from_nltk import get_data
from QA.alg.retrieve_methods import RetrieveHandler


def main():
    data = get_data()
    handler = RetrieveHandler(data)

    # handler.naive_retrieve("hello world")
    # handler.tf_idf_weight_retrieve("hello world")

    handler.interactive_mode()


if __name__ == '__main__':
    main()
