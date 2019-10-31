#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from aip import AipNlp

APP_ID = '17617866'
API_KEY = 'kHBH6lCy423KOR72KkOxtbMS'
SECRET_KEY = 'm3hwYS1ieBTZZdCUVR297cljVVx6b5gt'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

file = pd.read_csv("C:/Users/brjiang/Downloads/nba_weibo_2019_1024_pm.csv", encoding="utf-8")
print(file.shape)

# 去掉重复行
file.drop_duplicates(subset='Content', keep='first', inplace=True)
print(file.shape)

# 去掉两个关键字（来自于营销号）
index_list = []
for x in file.Content:
    if ("叶落" in x) | ("买球" in x):
        index_list.append(file.loc[file['Content'] == x, :].index[0])

file.drop(index_list, axis=0, inplace=True)
print(file.shape)
def get_score(x):
    x=client.sentimentClassify(str(x).encode(encoding='gbk', errors='ignore').decode('gbk'))
    if 'items' in x.keys():
        print(x['items'][0]['sentiment'])
        return x['items'][0]['sentiment']
    else:
        # print("error")
        # time.sleep(1.5)
        return get_score(x)
file['sentiment_score'] = file['Content'].apply(
    lambda x: get_score(x))

file.to_csv('a.csv')