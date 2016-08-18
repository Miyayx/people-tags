# -*- coding:utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm

"""
Features:
    1. 平均点击次数         0
    2. 平均查看物品个数     1
    3. 产品中是否有标识字样 2,3
    4. 购物行为停留时长     4
"""    

def generate_matrix(fn):
    id_class = {}
    id_features = {}
    id_datetime =  {}
    id_pagestring = {}
    id_behaviorcount = {}
    id_avgduration = {}

    for line in open(fn):
        items = line.strip('\n').strip().split('\t')
        _id, clazz = items[:2]
        product = items[-4]
        first = items[2]
        second = items[3]
        third = items[4]
        datetime = items[-1]
        pagestring = items[-2]

        if clazz != '男' or clazz != '女':
            continue

        if not _id in id_features:
            id_features[_id] = [0] * 5
            id_class[_id] = clazz

        if not _id in id_datetime:
            id_datetime[_id] = set()
        id_datetime[_id].add(long(datetime))

        if not _id in id_pagestring:
            id_pagestring[_id] = list() 
        id_pagestring[_id].append(pagestring)

        if '男' in product and '女' in product:
            continue

        k = '男'
        if k in product or k in first or k in second or k in third:
            #id_keyword_count[_id][k] = id_keyword_count[_id].get(k, 0)+1
            pass
            id_features[_id][2] += 1

        k = '女'
        if k in product or k in first or k in second or k in third:
            pass
            id_features[_id][3] += 1

    for _id, datetimes in id_datetime.items():
        s_duration = 0
        s_count = 0

        datetimes = sorted(list(datetimes))
        prev = datetimes[0]
        durations = []
        dura = 0
        for d in datetimes[1:]:
            if d == prev:
                continue
            if d-prev <= 30*60:
                dura += (d-prev)
            else:
                s_duration += dura
                s_count += 1
                dura = 0
            prev = d
        if dura > 0:
            s_duration += dura
            s_count += 1
            
        if s_duration == 0:
            s_duration += 1 
            s_count += 1

        id_behaviorcount[_id] = s_count
        #id_avgduration[_id] = s_duration * 1.0/s_count
        id_features[_id][4] = s_duration * 1.0/s_count

    for _id, pagestrings in id_pagestring.items():
        id_features[_id][0] = len(pagestrings) * 1.0/s_count
        id_features[_id][1] = len(set(pagestrings)) * 1.0/s_count
        pass

    matrix = [v for k, v in sorted(id_features.items(), key=lambda x:x[0])]
    labels = [v for k, v in sorted(id_class.items(), key=lambda x:x[0])]
    for m in matrix[:10]:
        print m
    return matrix, labels

def train_test(matrix, labels):

    print "\nLogistic Regression..."
    
    classifier = LogisticRegression(C=1.0)
    #classifier = svm.LinearSVC()
    classifier.fit(matrix, labels)
    prediction =  classifier.predict(matrix)
        
    print "Presicion:", classifier.score(matrix, labels)
    return classifier

def main():
    matrix, labels = generate_matrix('xj_phone_gender_product_balance.dat')
    train_test(matrix, labels)

if __name__ == '__main__':
    main()
