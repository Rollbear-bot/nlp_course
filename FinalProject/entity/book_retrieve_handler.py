# -*- coding: utf-8 -*-
# @Time: 2020/12/20 10:20
# @Author: Rollbear
# @Filename: book_retrieve_handler.py


DEBUG = True


from FinalProject.entity.book import Book
from FinalProject.service.vec_for_book_obj import Vec4Book
from sklearn.cluster import MeanShift
from util.timer import timer


class BookRetrieveHandler:
    def __init__(self, data):
        # member fields
        self.__book_objs = data
        self.docs_vec = None
        self.clusters = None

        # calling init methods
        self.doc_vectorize()
        self.clustering()

    @timer(DEBUG)
    def doc_vectorize(self):
        vec = Vec4Book(self.__book_objs)
        self.docs_vec = vec.get_profile_doc_vec()

    @timer(DEBUG)
    def clustering(self):
        ms_model = MeanShift()
        self.clusters = ms_model.fit(self.docs_vec)
