# -*- coding:utf-8 -*-

from fileio import *
from feature import *
from train import *
import time

def main1():
    id_matrix, features, id_class = read_matrix("../data/xinjiang_profile_wx_balance.dat", 7, frequency=True, norm=True, _class=True, class_col=2, class_type='age' )
    print "Samples:",len(id_matrix)

    #new_feature_index = merge_similar_brand(id_matrix, features)

    #id_matrix = convert_matrix(id_matrix, features, new_feature_index, False)

    matrix, labels = convert_matrix(id_matrix, id_class)
    #matrix = decomposition(matrix)
    #matrix = feature_selection(matrix, labels)
    #print information_gain(matrix, labels)
    
    train(matrix, labels)

    #output = '../data/xj_matrix_gender_brand_balance.csv'

    #write_matrix(output, id_matrix, id_class)

def main2():
    features = [line.strip('\n') for line in open('../data/xj_matrix_gender_v5_balance_selected_features.csv')]
    matrix, labels = read_csv_matrix("../data/xj_matrix_gender_v5_balance.csv", header=True, filter_features = features)
    print len(matrix)
    print len(labels)

    train(matrix, labels)

if __name__=="__main__":
    start_time = time.time()

    #main1()
    main2()

    print "Time Consuming:", time.time()-start_time

