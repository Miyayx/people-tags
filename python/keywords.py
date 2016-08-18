# -*- coding:utf-8 -*-

import pynlpir
pynlpir.open()

def get_keywords(text, max_words=10):
    return pynlpir.get_key_words(text, max_words)

if __name__=='__main__':
    keywords_count = {}
    url_keywords = {}
    for line in open('../data/xinjiang_profile_wx.dat'):
        url, content = line.strip('\n').split('\t')
        keys = get_keywords(content)
        url_keywords[url] = keys
        for k in keys:
            keywords_count[k] = keywords_count.get(k, 0) + 1

    keywords = [k for k, v in keywords_count.items() if v > 10]


