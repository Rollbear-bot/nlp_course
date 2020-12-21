# -*- coding: utf-8 -*-
# @Time: 2020/12/19 22:23
# @Author: Rollbear
# @Filename: vectorized.py

from gensim.models import Word2Vec
import jieba
from util.timer import timer
from FinalProject.service.stop_words import get_stop_words


DEBUG = False


@timer(DEBUG)
def drop_stop_words(cut_sents: iter):
    res = []
    stop_words = get_stop_words()
    for sent in cut_sents:
        cur_sent = [word for word in sent if word not in stop_words]
        res.append(cur_sent)

    return res


class Vec:
    def __init__(self, raw_data):
        self.wv_model = None
        self.cut_profile_data = None
        self.raw_data = raw_data

        self.preprocessing()
        self.init_wv_model()

    @timer(DEBUG)
    def preprocessing(self):
        # 去停用词
        self.cut_profile_data = drop_stop_words(
            [jieba.lcut(book_profile) for book_profile in self.raw_data
             if len(book_profile) != 0]
        )

    @timer(DEBUG)
    def init_wv_model(self):
        model = Word2Vec(self.cut_profile_data)
        self.wv_model = model

    # @timer(DEBUG)
    def get_wv_word_vec(self, word: str):
        return self.wv_model[word]

    # @timer(DEBUG)
    def get_wv_doc_vec(self, doc: iter):
        skip_count = 0  # 记录计算文档向量时跳过的词
        vec_sum = 0

        for word in doc:
            try:
                vec_sum += self.get_wv_word_vec(word)
            except KeyError:
                skip_count += 1
        # if DEBUG:
        #     print(f"in get_wv_doc_vec: skip {skip_count} in {len(doc)} words.")
        return vec_sum / len(doc) - skip_count

    def get_profile_doc_vec(self):
        return [self.get_wv_doc_vec(doc) for doc in self.cut_profile_data]
