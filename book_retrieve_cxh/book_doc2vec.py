# decoding=GBK
"""
流程总结：
1. getdata，获取数据，只取“简介”这一列
2.
"""


import gensim
import numpy as np
import pandas as pd
import jieba
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import time

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def getData():
    p = "../FinalProject/data/books.csv"
    data = pd.read_csv(p, sep=",", dtype=str, encoding="utf-8")
    data = data.drop(columns=['作者', '出版社', "关键词", "中国图书分类号", "出版年月"])
    data = data.astype(str)
    return data


# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='UTF-8').readlines()]
    return stopwords


stopwords = stopwordslist('../FinalProject/data/stop_words.txt')  # 这里加载停用词的路径


# 对句子去除停用词
def movestopwords(sentence):
    santi_words = []
    for x in sentence:
        if len(x) > 1 and x not in stopwords:
            santi_words.append(x)
    return santi_words


def get_corpus():
    data = getData()
    sen = []
    n = len(data)
    for i in range(n):
        sentence = data['摘要'][i]
        tag = data['书名'][i]
        words = jieba.cut(sentence)
        word_list = movestopwords(words)  # 去除停用词
        document = TaggededDocument(word_list, tags=[tag])
        sen.append(document)
    return sen


def get_train(x_train, size=200, epoch_num=1):
    """训练doc2vec模型（文档向量化）"""
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model_doc2vec')
    return model_dm


def get_test():
    """测试训练的doc2vec模型"""
    model_dm = Doc2Vec.load("./model_doc2vec")
    text_test = u'本书共分为疾病与疾病观、医疗技术的发展、医疗服务体系的完善三部分，主要内容包括疾病的变迁、疾病观念的演进、诊断、药物、外科、针灸、医事制度的建立等'
    text_cut = jieba.cut(text_test)
    word_list = movestopwords(text_cut)
    text_raw = []
    for i in list(word_list):
        text_raw.append(i)
    inferred_vector_dm = model_dm.infer_vector(text_raw)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)

    return sims


def train():
    x_train = get_corpus()
    model_dm = get_train(x_train)
    return model_dm


def test():
    sims = get_test()
    Top10 = sorted(sims, key=lambda x: x[1], reverse=True)
    for i in Top10:
        print(i[0])


if __name__ == '__main__':
    start = time.time()
    train()
    # test()

    end = time.time()
    print(end - start)
