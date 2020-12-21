# -*- coding: utf-8 -*-
# @Time: 2020/12/20 9:26
# @Author: Rollbear
# @Filename: vec_for_book_obj.py

from FinalProject.service.vectorized import Vec


class Vec4Book(Vec):
    def __init__(self, lt_book_objs: iter):
        raw_data = [book.abstract for book in lt_book_objs]
        super().__init__(raw_data)

