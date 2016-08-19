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
    print("model")
    #clf = SVC(C=1.0, probability=True)
    clf = LogisticRegression(C=1.0)
    clf.fit(trainX, trainY) 
    predict = clf.predict(testX)
    predict_proba = clf.predict_proba(testX)

    label_index = {}
    labelN = len(Counter(predict))
    for i in range(len(predict)):
        if len(label_index) >= labelN:
            break
        m = np.argmax(predict_proba[i])
        print(predict_proba[i])
        label_index[predict[i]] = m

    print('\nlabel index:')
    for l, i in label_index.items():
        print(str(l) + " "+ str(i))

    with open('tmp', 'w') as f:
        for ps in predict_proba:
            f.write("%s\n"%(",".join([str(p) for p in ps])))
    return predict_proba, label_index

def DG(prob_m, prob_f):
    g = (prob_m + prob_f) * 1.0 / 2
    dg = math.sqrt(((prob_m - g)**2 + (prob_f - g)**2)/2)/g
    return dg

def get_words(urls, fn, id_col=0, content_col=1, seperator='\t'):
    #import pynlpir
    #pynlpir.open()

    url_set = set(urls)
    print("Get words, %d articles"%(len(url_set)))
    url_words = {}
    stop_words = set([ line.strip('\n') for line in open('stop_words.txt')])

    c = 0
    for line in open(fn):
        items = line.strip('\n').split(seperator)
        url, text = items[id_col], items[content_col]
        if not url in url_set:
            continue
        if url in url_words:
            continue
        words = [ i for i in clipper(text) if not i.isdigit() ]
        url_words[url] = words
        c+=1
        sys.stdout.write('\r' + repr(c) + ' rows has been read ')
        sys.stdout.flush()
    print("\n")

    doc_words = []
    for url in urls:
        if not url in url_words:
            #print url
            continue
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
        print("keywords %s"%str(k))
    return X, keywords

def select_feature_by_ig(doc_words, labels, featureN=8000):
    """
    用information gain
    """

    print("select feature by ig")
    keywords = []
    for words in doc_words:
        if type(words) == str:
            for w in words.split():
                keywords.append(w)
        else:
            keywords += words
    keywords = list(set(keywords))
    print("keywords num:%d"%len(keywords))
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

def generateX(features, doc_words, norm=False):
    print("generateX")
    feature_index = dict((k, i) for i, k in enumerate(features))
    X = np.zeros((len(doc_words), len(features)))
    for i in range(len(doc_words)):
        for w in doc_words[i]:
            if w in feature_index:
                index = feature_index[w]
                X[i][index] += 1
        sys.stdout.write('\r' + repr(i) + ' has been generated ')
        sys.stdout.flush()
    print("\n")
    if norm:
        X = sklearn.preprocessing.normalize(X)
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
        maxi = np.argmax(c_predict)
        if c_predict[maxi] > 0.0:
            id_predict.append(maxi)
        else:
            id_predict.append(None)

    print("\nPredict Distribution: %s"%str(Counter(id_predict)))

    with open('predict_result', 'w') as f:
        for ps in id_predict_proba:
            f.write("%s\n"%(",".join([str(p) for p in ps])))

    return id_predict

def smooth(X):
    X[X < 0.00001] = 0.1
    return X

def norm(X):
    return X/X.sum(axis=1)[:,None]

def get_separate_R(R):
    from sklearn.utils.extmath import randomized_svd
    U, Sigma, VT = randomized_svd(X, n_components=30, n_iter=5, random_state=None)
    return U*np.sqrt(Sigma), VT*np.sqrt(Sigma)

if __name__ == '__main__':
    import time

    start_time = time.time()

    #id_matrix, urls, id_class = read_matrix('../data/xinjiang_profile_wx_balance.dat', feature_col=3, times=10, frequency=True, norm=False)
    #id_matrix, urls, id_class = read_matrix('../data/xj_gender_product_balance.dat', feature_col=8, frequency=True, norm=False, times=2)
    id_matrix, urls, id_class = read_matrix('../data/xj_phone_gender_product_balance.dat', feature_col=7, frequency=True, norm=False, times=2)
    print("Urls: %d"%len(urls))

    print("Actual Distribution:")
    for k, v in Counter(id_class.values()).items():
        print(k+" "+v)

    matrix, labels = convert_matrix(id_matrix, id_class)
    matrix = np.array(matrix)


    w_sum = matrix.sum(axis=0) #每个页面被点击的总次数

    w_c = {} #每个页面的点击数在类别上的分布
    for c in set(labels):
        w_c[c] = np.array([0] * len(w_sum))

    for i, c in enumerate(labels):
        for j in range(len(urls)):
            w_c[c][j] += matrix[i][j]

    train_urls = []
    trainY = []

    f_web_tend_examples = open('web_tend_examples', 'w')

    for c, array in w_c.items():
        for i, prob in enumerate(w_c[c] * 1.0 / w_sum):
            if prob > 0.75:
                f_web_tend_examples.write("%s\t%s\n"%(c, urls[i]))
                train_urls.append(urls[i])
                trainY.append(c)

    f_web_tend_examples.close()

    print('train num: %d'%len(train_urls))
    print('train stat:')
    for k, v in Counter(trainY).items():
        print(k+" "+v)
    #DGs = []
    #for p1, p2 in zip(w_c.values()[0], w_c.values()[1]):
    #    DGs.append(DG(p1, p2))
    #print sorted(DGs, reverse=True)[:10]

    ####  content features
    ## 用文本内容
    #train_doc_words1 = get_words(train_urls, './wx_result.dat', id_col=-1, content_col=1, seperator='\0')
    ## 用标题
    #train_doc_words1 = get_words(train_urls, '../data/xinjiang_profile_wx_balance.dat', id_col=3, content_col=8, seperator='\t')
    #train_doc_words1 = get_words(train_urls, '../data/xj_gender_product_balance.dat', id_col=8, content_col=7, seperator='\t')
    train_doc_words1 = get_words(train_urls, '../data/xj_phone_gender_product_balance.dat', id_col=7, content_col=6, seperator='\t')
    features1 = select_feature_by_ig(train_doc_words1, trainY, featureN=5000)
    trainX1 = generateX(features1, train_doc_words1)

    #### category features
    #train_doc_words2 = get_words(train_urls, '../data/xinjiang_profile_wx_balance.dat', id_col=3, content_col=7, seperator='\t')
    train_doc_words2 = get_words(train_urls, '../data/xj_phone_gender_product_balance.dat', id_col=7, content_col=5, seperator='\t')
    features2 = select_feature_by_ig(train_doc_words2, trainY, featureN=3000)
    trainX2 = generateX(features2, train_doc_words2)

    trainX = np.concatenate((trainX1,trainX2),axis=1)
    #trainX = trainX1

    #train(trainX, trainY)

    #all_doc_words1 = get_words(urls, './wx_result.dat', id_col=-1, content_col=1, seperator='\0')
    #all_doc_words1 = get_words(urls, '../data/xj_gender_product_balance.dat', id_col=8, content_col=7, seperator='\t')
    all_doc_words1 = get_words(urls, '../data/xj_phone_gender_product_balance.dat', id_col=7, content_col=6, seperator='\t')
    allX1 = generateX(features1, all_doc_words1)
    #allX = allX1
    #all_doc_words2 = get_words(urls, '../data/xinjiang_profile_wx_balance.dat', id_col=3, content_col=7, seperator='\t')
    all_doc_words2 = get_words(urls, '../data/xj_phone_gender_product_balance.dat', id_col=7, content_col=5, seperator='\t')
    allX2 = generateX(features2, all_doc_words2)
    allX = np.concatenate((allX1,allX2),axis=1)
    #testX, test_keywords = content_feature(urls, '/Users/Miyayx/Documents/workspace/pageparser-weixin/wx_result.dat')

    url_tend_matrix, label_index = model(trainX, trainY, allX) #计算所有url的类别趋势

    predict = demographic_predict(matrix, url_tend_matrix)
    predict = [ p for p in predict if p != None]

    correct = 0
    
    print("Predict Number: %d"%len(predict))
    print("Total Number: %d"%len(labels))
    for i in range(len(predict)):
        if not labels[i] in label_index:
            continue
        if predict[i] == None:
            continue
        if label_index[labels[i]] == predict[i]:
            correct += 1
    print("Accuracy = %f"%(correct*1.0/len(predict)))
    

    print('Time Consuming: %f'%(time.time() - start_time))

