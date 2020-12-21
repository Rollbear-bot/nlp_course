# -*- coding: utf-8 -*-
# @Time: 2020/12/14 8:41
# @Author: Rollbear
# @Filename: retrieve_methods.py

import nltk
from util import timer
from nltk import word_tokenize
import math
from gensim.models import Word2Vec
import numpy as np

DEBUG = True


def intersection_rank(score_sets):
    """
    计算交集
    实现方式：哈希分桶
    计算交集，并将doc_id相同的元素的score字段相加，并按照score排序
    :param score_sets: list of set of (doc_id, score)
    :return: 交集
    """
    vol = len(score_sets)
    hash_bucket = {}
    for s in score_sets:
        for word in s:
            if word[1] != 0:
                if word[0] not in hash_bucket:
                    hash_bucket[word[0]] = [word[1]]
                else:
                    hash_bucket[word[0]].append(word[1])

    return sorted([(key, sum(elem)) for key, elem in hash_bucket.items() if len(elem) >= vol],
                  key=lambda elem: elem[1],
                  reverse=True)


class RetrieveHandler:
    def __init__(self, docs: iter):
        self.__docs = docs
        self.__num_docs = len(docs)
        self.__porter = nltk.PorterStemmer()

        self.__word_frequency = None
        self.__doc_frequency = None
        self.__tf_idf_weight = None
        self.__word_doc_distance = None

        self.__init_word_freq()
        self.__init_doc_freq()
        self.__init_tf_idf()
        # self.__init_wv_distance()

    @timer(DEBUG)
    def __init_word_freq(self):
        """初始化词频表"""
        wf = {}
        for doc_id, doc in enumerate(self.__docs):
            cur_wf = {}  # 单个文档的词频计数
            for word in doc:
                word_stem = self.__porter.stem(word)
                cur_wf[word_stem] = cur_wf.get(word_stem, 0) + 1
            # 将单个文档的词频计数更新到全局表
            for word_s, count in cur_wf.items():
                if word_s in wf:
                    wf[word_s].append((doc_id, count))
                else:
                    wf[word_s] = [(doc_id, count)]

        self.__word_frequency = wf

    @timer(DEBUG)
    def __init_doc_freq(self):
        """初始化文档词频表"""
        df = {}
        for doc_id, doc in enumerate(self.__docs):
            for word in doc:
                word_stem = self.__porter.stem(word)
                df[word_stem] = df.get(word_stem, 0) + 1

        self.__doc_frequency = df

    @timer(DEBUG)
    def __init_tf_idf(self):
        """初始化每个词的tf-idf权重"""
        tf = self.__word_frequency
        df = self.__doc_frequency
        tf_idf = {}
        for word, data in tf.items():
            for w_freq in data:
                # tf-idf weight计算公式
                tf_idf_score = (1 + math.log(w_freq[1], 10)) * math.log(self.__num_docs / df[word], 10)

                if word not in tf_idf:
                    tf_idf[word] = [(w_freq[0], tf_idf_score)]
                else:
                    tf_idf[word].append((w_freq[0], tf_idf_score))
        self.__tf_idf_weight = tf_idf

    @timer(DEBUG)
    def naive_retrieve(self, target: str):
        target_tokens = word_tokenize(target)
        token_map = {}

        for token in target_tokens:
            docs_rank = []
            token_stem = self.__porter.stem(token)
            for doc_id, doc in enumerate(self.__docs):
                count = 0
                for word in doc:
                    if self.__porter.stem(word) == token_stem:
                        count += 1
                docs_rank.append((doc_id, count))

            token_map[token] = docs_rank

        r = intersection_rank(token_map.values())

        if DEBUG:
            self.__print_res(r)
        return r

    @timer(DEBUG)
    def tf_idf_weight_retrieve(self, target: str):
        target_tokens = word_tokenize(target)
        tokens_retrieve = []
        for token in target_tokens:
            token_stem = self.__porter.stem(token)
            try:
                tokens_retrieve.append(self.__tf_idf_weight[token_stem])
            except KeyError:
                if DEBUG:
                    print("No Found!")

        r = intersection_rank(tokens_retrieve)

        if DEBUG:
            self.__print_res(r)
        return r

    @timer(DEBUG)
    def __init_wv_distance(self):
        if DEBUG:
            print("training word2vec...")
        model = Word2Vec(self.__docs, min_count=1)

        if DEBUG:
            print("calc doc vec...")
        doc_vec = []
        for doc in self.__docs:
            vec = sum([model[word] for word in doc]) / len(doc)
            doc_vec.append(vec)

        if DEBUG:
            print("calc distance...")
        word_doc_distance = {}
        words = set([word for doc in self.__docs for word in doc])

        for word in list(words)[:3]:
            for d_id, _ in enumerate(self.__docs):
                # 欧氏距离
                distance = np.sqrt(np.sum((np.array(doc_vec[d_id]) - np.array(model[word])) ** 2))
                if word not in word_doc_distance:
                    word_doc_distance[word] = [(d_id, distance)]
                else:
                    word_doc_distance[word].append((d_id, distance))
            word_doc_distance[word] = sorted(word_doc_distance[word], key=lambda elem: elem[1])
        self.__word_doc_distance = word_doc_distance

    @timer(DEBUG)
    def vec_search(self, target, num_top=10):
        target_tokens = word_tokenize(target)
        sets = []
        for token in target_tokens:
            if len(self.__word_doc_distance[token]) < num_top:
                sets.append(self.__word_doc_distance[token])
            else:
                sets.append(self.__word_doc_distance[token][:num_top])
        r = intersection_rank(sets)
        if DEBUG:
            self.__print_res(r)
        return r

    def __print_res(self, res):
        if DEBUG:
            if len(res) > 0:
                print("top10:")
                for elem in res[:10]:
                    print(f"score: {elem[1]}\tdoc: {' '.join(self.__docs[elem[0]])}")
            else:
                print("No Found!")

    def interactive_mode(self):
        while True:
            q = input("Search Something: ")
            self.tf_idf_weight_retrieve(target=q)


if __name__ == '__main__':
    # just for test
    set_1 = [(1, 1), (2, 3)]
    set_2 = [(3, 2), (2, 4)]
    sets = [set_1, set_2]
    tmp = intersection_rank(sets)
    print(tmp)
