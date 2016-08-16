# -*- coding:utf-8 -*-

import numpy as np
import sklearn.feature_extraction
import sklearn.naive_bayes as nb
import sklearn.externals.joblib as jl
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
import sys
import copy
import jieba

def clipper(txt):
    return jieba.cut(txt)

def word():
    catedict = { "男":0, "女":1 }

    N_Features = 800

    fh = sklearn.feature_extraction.FeatureHasher(n_features=N_Features, non_negative=True, input_type='string')

    kvlist = []
    targetlist = []
    # use partial fitting because of big data frame.
    c = 0
    for line in open('xj_phone_gender_product_balance.dat'):
        items = line.strip('\n').split('\t')
        clazz, text = items[1], items[6]
        if not clazz in catedict:
            continue
        kvlist += [  [ i for i in clipper(text) ] ]
        targetlist += [int(catedict[clazz])]
        c+=1
        sys.stdout.write('\r' + repr(c) + ' rows has been read   ')
        sys.stdout.flush()

    print("\npartial fitting...")
    X = fh.fit_transform(kvlist)

def merge_similar_brand(id_matrix, features):
    print "Features:", len(features)

    feature_index = dict((k, i) for i, k in enumerate(features))

    features.reverse()

    brand_split = {}
    for f in features:
        if u"（" in f:
            items = f.replace(u"（", "\t").replace(u"）", "\t").split("\t")
            items = [i.strip().lower().replace(" ","") for i in items if len(i.strip()) > 0]
            brand_split[f] = items

    for i in range(len(features)-1):
        fi = features[i]
        if not fi in brand_split:
            continue
        for j in range(i+1, len(features)):
            fj = features[j]
            itemi = brand_split[fi]
            if fj in brand_split:
                itemj = brand_split[fj]
                if len(set(itemi) & set(itemj)) > 0:
                    feature_index[fj] = feature_index[fi]
                    #print "merge", fi, fj
            elif fj.replace(" ","").lower() in itemi:
                feature_index[fj] = feature_index[fi] 
                #print "merge", fi, fj
    features.reverse()

    featureN = len(set(feature_index.values()))
    vector = [0] * featureN
    for f, i in feature_index.iteritems():
        if i < featureN:
            vector[i] = 1

    print "vector:",vector.count(0)
    j = vector.index(0)
    for f, i in copy.copy(feature_index).iteritems():
        if i >= featureN:
            vector[j] = 1
            feature_index[f] = j
            for k in range(j+1, featureN):
                if vector[k] == 0:
                    j = k
                    break
    print "vector:",vector.count(0)

    return feature_index

def convert_matrix(id_matrix, features, feature_index, frequency=True):
    new_matrix = {}

    featureN = len(set(feature_index.values()))
    print "Old feature Count", len(features)
    print "New feature Count", featureN

    for _id, matrix in id_matrix.iteritems():
        vector = [0] * featureN
        for i, feature in enumerate(features):
            if frequency:
                vector[feature_index[feature]] += matrix[i]
            else:
                if matrix[i] == 1:
                    vector[feature_index[feature]] = 1
        new_matrix[_id] = vector
    return new_matrix

def information_gain(X, y):
    X = np.array(X)

    def _calIg():
        entropy_x_set = 0
        entropy_x_not_set = 0
        for c in classCnt:
            probs = classCnt[c] / float(featureTot)
            entropy_x_set = entropy_x_set - probs * np.log(probs)
            probs = (classTotCnt[c] - classCnt[c]) / (float(tot - featureTot) + 0.0001)
            entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        for c in classTotCnt:
            if c not in classCnt:
                probs = classTotCnt[c] / float(tot - featureTot)
                entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                             +  ((tot - featureTot) / float(tot)) * entropy_x_not_set)

    tot = X.shape[0]
    classTotCnt = {}
    entropy_before = 0
    for i in y:
        if i not in classTotCnt:
            classTotCnt[i] = 1
        else:
            classTotCnt[i] = classTotCnt[i] + 1
    for c in classTotCnt:
        probs = classTotCnt[c] / float(tot)
        entropy_before = entropy_before - probs * np.log(probs)

    nz = X.T.nonzero()
    pre = 0
    classCnt = {}
    featureTot = 0
    information_gain = []
    print "\n" ,len(y)
    for i in range(0, len(nz[0])):
        if (i != 0 and nz[0][i] != pre):
            for notappear in range(pre+1, nz[0][i]):
                information_gain.append(0)
            ig = _calIg()
            information_gain.append(ig)
            pre = nz[0][i]
            classCnt = {}
            featureTot = 0
        featureTot = featureTot + 1
        sys.stdout.write('\r' + repr(nz[1][i]) + ' label row ')
        sys.stdout.flush()
        yclass = y[nz[1][i]]
        if yclass not in classCnt:
            classCnt[yclass] = 1
        else:
            classCnt[yclass] = classCnt[yclass] + 1
    ig = _calIg()
    information_gain.append(ig)

    return np.asarray(information_gain)


