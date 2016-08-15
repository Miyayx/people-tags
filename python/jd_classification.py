# -*- coding:utf-8 -*-

from fileio import *
from feature import *
from train import *
from webpage_tendency import *

if __name__ == '__main__':
    import time

    start_time = time.time()

    id_matrix, origin_features, id_class = read_matrix('../data/xj_phone_gender_product_balance.dat', feature_col=5, times=1, frequency=False, norm=False )
    matrix, labels = convert_matrix(id_matrix, id_class)

    words_matrix = []
    for vector in matrix:
        words_matrix.append([origin_features[i] for i in range(len(vector)) if vector[i] > 0 ])
    features = select_feature_by_ig(words_matrix, labels, featureN=4000 )
    allX = generateX(features, words_matrix)
    #binarizer = preprocessing.Binarizer().fit(allX)
    #allX = binarizer.transform(matrix)

    train(allX, labels)
    print 'Time Consuming:', time.time() - start_time
    
