# -*- coding:utf-8 -*-

import numpy as np
from fileio import *
from train import *
from feature import clipper
from feature import information_gain
from sklearn.svm import SVR
from sklearn.svm import SVC
import sklearn.feature_extraction
import math
import sys
from collections import Counter

def model(trainX, trainY, testX):
    clf = SVC(C=1.0, probability=True)
    clf.fit(trainX, trainY) 
    predict = clf.predict(testX)
    predict_proba = clf.predict_proba(testX)

    label_index = {}
    labelN = len(Counter(predict))
    for i in range(len(predict)):
        if len(label_index) >= labelN:
            break
        m = np.argmax(predict_proba[i])
        print predict_proba[i]
        label_index[predict[i]] = m

    print '\nlabel index:'
    for l, i in label_index.iteritems():
        print l,i

    with open('tmp', 'w') as f:
        for ps in predict_proba:
            f.write("%s\n"%(",".join([str(p) for p in ps])))
    return predict_proba, label_index

def DG(prob_m, prob_f):
    g = (prob_m + prob_f) * 1.0 / 2
    dg = math.sqrt(((prob_m - g)**2 + (prob_f - g)**2)/2)/g
    return dg

def get_words(urls, fn, id_col=0, content_col=1, seperator='\t'):
    import pynlpir
    pynlpir.open()

    url_set = set(urls)
    url_words = {}
    stop_words = set([ line.strip('\n') for line in open('stop_words.txt')])

    c = 0
    for line in open(fn):
        items = line.strip('\n').split(seperator)
        url, text = items[id_col], items[content_col]
        if not url in urls:
            continue
        if url in url_words:
            continue
        words = [ i for i in clipper(text) if not i.isdigit() ]
        url_words[url] = words
        c+=1
        sys.stdout.write('\r' + repr(c) + ' rows has been read ')
        sys.stdout.flush()

    doc_words = []
    for url in urls:
        doc_words.append(url_words[url])
        #doc_words += [  " ".join([ i for i in pynlpir.get_key_words(text) if not i.isdigit() and not i in stop_words ]) ]
    return doc_words

def content_feature(doc_words, featureN=1000 ):
    """
    直接从分词中通过tfidf找到keyword
    好像不管用？
    """

    fe = sklearn.feature_extraction.text.TfidfVectorizer(max_features=featureN)
    X = fe.fit_transform(doc_words)
    keywords = fe.get_feature_names()
    for k in keywords[:20]:
        print "keywords",k
    return X, keywords

def select_feature_by_ig(doc_words, labels, featureN=8000):
    """
    用information gain
    """

    keywords = []
    for words in doc_words:
        if type(words) == str:
            for w in words.split():
                keywords.append(w)
        else:
            keywords += words
    keywords = list(set(keywords))
    #fw = open('jd_keywords.dat', 'w')
    #for w in keywords:
    #    fw.write(("%s\n"%(w)).encode('utf-8'))
    #fw.close()

    keyword_index = dict((k, i) for i, k in enumerate(keywords))

    X = np.zeros((len(doc_words), len(keywords)))
    for i in range(len(doc_words)):
        for w in doc_words[i]:
            X[i][keyword_index[w]] = 1

    infos = information_gain(X, labels)

    indices = infos.argsort()[-featureN:][::-1]

    with open('information_gain.dat', 'w') as f:
        for i in indices:
            try:
                f.write(("%f\t%s\n"%(infos[i], keywords[i])).encode('utf-8'))
            except:
                f.write(("%f\t%s\n"%(infos[i], keywords[i])))

    features = [keywords[i] for i in indices]

    return features

def generateX(features, doc_words):
    feature_index = dict((k, i) for i, k in enumerate(features))
    X = np.zeros((len(doc_words), len(features)))
    for i in range(len(doc_words)):
        for w in doc_words[i]:
            if w in feature_index:
                index = feature_index[w]
                X[i][index] += 1
    return X

def demographic_predict(id_url_matrix, url_tend_matrix):
    id_predict_proba = []
    id_predict = []
    for u_vector in id_url_matrix:
        c_predict = np.array([1.0] * len(url_tend_matrix[0]))
        for i in range(len(urls)):
            if u_vector[i] > 0:
                c_predict *= url_tend_matrix[i]
        id_predict_proba.append(c_predict)
        #id_predict.append(1 if c_predict[1] > 0.4 else np.argmax(c_predict))
        id_predict.append(np.argmax(c_predict))

    print "\nPredict Distribution:", Counter(id_predict)

    with open('predict_result', 'w') as f:
        for ps in id_predict_proba:
            f.write("%s\n"%(",".join([str(p) for p in ps])))

    return id_predict

if __name__ == '__main__':
    import time

    start_time = time.time()

    id_matrix, urls, id_class = read_matrix('../data/xinjiang_profile_wx_balance.dat', feature_col=3, times=30, frequency=True, norm=False, _class=True, class_col=2, class_type='age' )
    #id_matrix, urls, id_class = read_matrix('../data/xj_phone_gender_product_balance.dat', feature_col=-3, frequency=True, norm=False )
    print "Urls:",len(urls)

    print "Actual Distribution:"
    for k, v in Counter(id_class.values()).items():
        print k,v

    matrix, labels = convert_matrix(id_matrix, id_class)
    matrix = np.array(matrix)
    w_sum = matrix.sum(axis=0)

    w_c = {}
    for c in set(labels):
        w_c[c] = np.array([0] * len(w_sum))

    url_tendency = {}
    for i, c in enumerate(labels):
        for j in range(len(urls)):
            w_c[c][j] += matrix[i][j]

    train_urls = []
    trainY = []

    f_web_tend_examples = open('web_tend_examples', 'w')

    for c, array in w_c.iteritems():
        for i, prob in enumerate(w_c[c] * 1.0 / w_sum):
            if prob > 0.75:
                f_web_tend_examples.write("%s\t%s\n"%(c, urls[i]))
                train_urls.append(urls[i])
                trainY.append(c)

    f_web_tend_examples.close()

    print 'train num:', len(train_urls)
    print 'train stat:'
    for k, v in Counter(trainY).items():
        print k,v
    #DGs = []
    #for p1, p2 in zip(w_c.values()[0], w_c.values()[1]):
    #    DGs.append(DG(p1, p2))
    #print sorted(DGs, reverse=True)[:10]

    train_doc_words = get_words(train_urls, './wx_result.dat', id_col=-1, content_col=1, seperator='\0')
    #train_doc_words = get_words(train_urls, '../data/xj_phone_gender_product_balance.dat', id_col=-3, content_col=6)
    features = select_feature_by_ig(train_doc_words, trainY, featureN=10000)
    trainX = generateX(features, train_doc_words)

    all_doc_words = get_words(urls, './wx_result.dat', id_col=-1, content_col=1, seperator='\0')
    #all_doc_words = get_words(urls, '../data/xj_phone_gender_product_balance.dat', id_col=-3, content_col=6)
    allX = generateX(features, all_doc_words)
    #testX, test_keywords = content_feature(urls, '/Users/Miyayx/Documents/workspace/pageparser-weixin/wx_result.dat')
    url_tend_matrix, label_index = model(trainX, trainY, allX)

    predict = demographic_predict(matrix, url_tend_matrix)

    correct = 0
    
    print "Predict Number:", len(predict)
    print "Total Number:", len(labels)
    for i in range(len(predict)):
        if not labels[i] in label_index:
            continue
        if label_index[labels[i]] == predict[i]:
            correct += 1
    print "Accuracy = %f"%(correct*1.0/len(labels))
    

    print 'Time Consuming:', time.time() - start_time

