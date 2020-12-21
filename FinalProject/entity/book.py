# -*- coding: utf-8 -*-
# @Time: 2020/12/20 9:22
# @Author: Rollbear
# @Filename: book.py

class Book:
    def __init__(self, name, author, publish_house, key_words, abstract, book_clf_label, publish_date):
        # 书名,作者,出版社,关键词,摘要,中国图书分类号,出版年月
        self.__name = name
        self.__author = author
        self.__publish_house = publish_house
        self.__key_words = key_words
        self.__abstract = abstract
        self.__book_clf_label = book_clf_label
        self.__publish_date = publish_date

    @property
    def abstract(self):
        return self.__abstract
