# -*- coding: utf-8 -*-
# @Time: 2020/12/19 22:17
# @Author: Rollbear
# @Filename: data_interface.py

import csv


def get_data(drop_header=True):
    path = "./data/books.csv"

    with open(path, "r", encoding="utf8") as rf:
        reader = csv.reader(rf)

        # 表头：['书名', '作者', '出版社', '关键词', '摘要', '中国图书分类号', '出版年月']
        data = list(reader)
        if drop_header:
            del data[0]  # 去掉表头
    return data
