# -*- coding:utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def train2():

    classifier = nb.MultinomialNB(alpha = 0.01)
    classifier = LogisticRegression()

    scores = cross_validation.cross_val_score(classifier, X, targetlist, cv=5)
    print "Matrix N:",N_Features
    print classifier
    print scores
    targetlist = []
    kvlist = []

    #finally , save the model
    #jl.dumps(gnb,'final.pkl')
    #you can use the model and feature hasher another place to predict text category.

def train(X, labels, models = []):

    # Create classifiers
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, n_jobs=-1)
    lr2 = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, n_jobs=-1)
    lr3 = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, n_jobs=-1)
    gnb = GaussianNB()
    bnb = BernoulliNB()
    mnb = MultinomialNB()
    svc = svm.LinearSVC(C=1.0)
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    classifiers = [(lr, 'Logistic'), (gnb, 'Naive Bayes'),(bnb, 'BernoulliNB'), (mnb, 'MultinomialNB'), (svc, 'Support Vector Classification'), (rfc, 'Random Forest')]

    if not models:
        models = range(len(classifiers))

    for mi in models:
        classifier, name = classifiers[mi]
    
        print str(classifier)

        scores = cross_validation.cross_val_score(classifier, X, labels, cv=5)
        #classifier.fit(matrix, labels)
        #prediction =  classifier.predict(matrix)
            
        #print "Presicion:", classifier.score(matrix, labels)
        print scores
    #return classifier

def convert_matrix(id_matrix, id_class):
    matrix = [v for k, v in sorted(id_matrix.iteritems(), key=lambda x:x[0])]
    labels = [v for k, v in sorted(id_class.iteritems(), key=lambda x:x[0])]
    return matrix, labels

def decomposition(X, n=100):
    pca = TruncatedSVD(n_components=n)
    return pca.fit_transform(X)

def feature_selection(X, y):
    print np.array(X).shape
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X = model.transform(X)
    print X.shape
    return X

