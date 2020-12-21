# -*- coding: utf-8 -*-
# @Time: 2020/12/19 22:16
# @Author: Rollbear
# @Filename: clustering.py

from sklearn.cluster import MeanShift


def mean_shift_clustering(data):
    ms_model = MeanShift()
    ms_model.fit(data)
    return ms_model
