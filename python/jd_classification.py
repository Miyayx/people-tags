# -*- coding:utf-8 -*-

from fileio import *
from feature import *
from train import *
from webpage_tendency import *

def main1():
    id_matrix, origin_features, id_class = read_matrix('../data/xj_gender_product_balance.dat', feature_col=6, times=2, frequency=False, norm=False, _class=True, class_col=2, class_type='age' )
    matrix, labels = convert_matrix(id_matrix, id_class)

    words_matrix = []
    for vector in matrix:
        words_matrix.append([origin_features[i] for i in range(len(vector)) if vector[i] > 0 ])
    
    featureNs = [6000, 7000, 8000, 9000, 10000, 11000, 12000]
    # age 7000
    # gender 10700?
    #featureNs = [10500, 10600, 10700, 10800, 10900, 11000, 11100, 11200, 11300, 11400, 11500]
    for featureN in featureNs:
        print "features:", featureN
        features = select_feature_by_ig(words_matrix, labels, featureN=featureN )
        allX = generateX(features, words_matrix)

        train(allX, labels, [3])

def main2():
    id_matrix, page_strings, id_class = read_matrix('../data/xj_gender_product_8000.dat', feature_col=-1, times=1, frequency=False, norm=False)
    matrix, labels = convert_matrix(id_matrix, id_class)

    words_matrix = get_words(page_strings, '../data/xj_gender_product_8000.dat', id_col=-1, content_col=7, seperator='\t')
    words_matrix2 = []
    for i,vector in enumerate(matrix):
        words = []
        for j, v in enumerate(vector):
            if v > 0:
                words += words_matrix[j]
        if not words:
            words.append("")
        words_matrix2.append(words)


    featureNs = [9000, 10000, 11000, 12000, 13000, 14000, 15000]
    #featureNs = [10500, 10600, 10700, 10800, 10900, 11000, 11100, 11200, 11300, 11400, 11500]
    # age 7000
    # gender 10700?
    for featureN in featureNs:
        print "features:", featureN
        features = select_feature_by_ig(words_matrix2, labels, featureN=featureN )
        allX = generateX(features, words_matrix2)

        train(allX, labels, [3])

if __name__ == '__main__':
    import time

    start_time = time.time()

    main2()

    print 'Time Consuming:', time.time() - start_time
    
