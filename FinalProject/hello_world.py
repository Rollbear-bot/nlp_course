# -*- coding: utf-8 -*-
# @Time: 2020/12/19 22:07
# @Author: Rollbear
# @Filename: hello_world.py

from FinalProject.data.data_interface import get_data
from clustering import mean_shift_clustering
from FinalProject.entity.book import Book
from FinalProject.service.vec_for_book_obj import Vec4Book
from FinalProject.entity.book_retrieve_handler import BookRetrieveHandler


def main():
    data = get_data(drop_header=True)
    book_lt = [Book(*line) for line in data]

    handler = BookRetrieveHandler(book_lt)
    print(handler.clusters[0])


if __name__ == '__main__':
    main()
