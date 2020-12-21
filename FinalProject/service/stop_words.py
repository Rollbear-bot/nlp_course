# -*- coding: utf-8 -*-
# @Time: 2020/12/20 9:55
# @Author: Rollbear
# @Filename: stop_words.py

STOP_WORDS_PATH = "./data/stop_words.txt"


def get_stop_words():
    with open(STOP_WORDS_PATH, "r", encoding="utf8") as rf:
        stop_words = [word.rstrip().strip() for word in rf.readlines()]
        return stop_words
