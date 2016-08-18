# -*- coding:utf-8 -*-

import codecs
import copy
import numpy as np

def read_matrix(fn, feature_col=3, times = 1, frequency=True, norm=True, _class=True, class_col=1, class_type='gender'):
    feature_count = {}
    for line in open(fn):
        f = line.strip('\n').split('\t')[feature_col]
        feature_count[f] = feature_count.get(f, 0) + 1
    #feature_list = [f for f in feature_count if feature_count[f] >= 1 ]
    feature_list = [f for f in feature_count if feature_count[f] >= times ]
    #feature_list = list(set([line.split('\t')[feature_col].strip() for line in codecs.open(fn, 'r', 'utf-8')]))
    if '-1' in feature_list:
        feature_list.remove('-1')
    print("Feature Number: %d"%len(feature_list))
    feature_index = dict((k, i) for i, k in enumerate(feature_list))

    id_matrix = {}
    id_class = {}
    
    for line in open(fn):
        items = line.split('\t')
        _id = items[0]
        clazz = items[class_col].strip()
        feature = items[feature_col].strip()

        if not feature in feature_index:
            continue

        if feature == '-1':
            continue
        
        if _class:
            if class_type == 'gender':
                if clazz.isdigit():
                    id_class[_id] = int(clazz)
                else:
                    id_class[_id] = clazz
            elif class_type == 'age':
                try:
                    clazz = int(float(clazz))
                    if clazz < 18:
                        id_class[_id] = '<18'
                    elif clazz >= 18 and clazz < 25:
                        id_class[_id] = '18~24'
                    elif clazz >= 25 and clazz < 35:
                        id_class[_id] = '25~35'
                    elif clazz >= 35 and clazz < 45:
                        id_class[_id] = '35~45'
                    elif clazz >= 45 and clazz < 55:
                        id_class[_id] = '45~55'
                    else:
                        id_class[_id] = '>=55'
                except Exception as e:
                    print(e)
                    continue

        if not _id in id_matrix:
            id_matrix[_id] = [0 for i in range(len(feature_list))]

        if frequency:
            id_matrix[_id][feature_index[feature]] += 1
        else:
            id_matrix[_id][feature_index[feature]] = 1

    if norm:
        for _id, matrix in id_matrix.items():
            s = sum(matrix)
            id_matrix[_id] = [1.0*i/s for i in matrix]

    if _class:
        return id_matrix, feature_list, id_class
    else:
        return id_matrix, feature_list

def write_matrix(output, id_matrix, id_class, boolean=True, features=[], class_type="gender"):

    featureN = len(id_matrix.values()[0])

    print("Writing to%s"%output)
    fw = open(output, 'w')
    if features:
        fw.write(','.join(features)+',Class\n')
    else:
        fw.write(','.join(['A'+str(i) for i in range(featureN)])+',Class\n')
    for _id, features in id_matrix.items():
        if boolean:
            features = ["no" if f == 0 else "yes" for f in features ]
        fw.write(','.join([str(f) for f in features]))
        
        if class_type == "gender":
            if id_class[_id] == 1 or id_class[_id] == '女':
                fw.write(',female\n')
            else:
                fw.write(',male\n')
        elif class_type == "age":
                fw.write(','+id_class[_id]+'\n')
    fw.close()

def read_csv_matrix(fn, header=True, filter_features=[]):
    print('Reading %s'%fn)
    delimiter = ','
    #from numpy import genfromtxt
    #data = genfromtxt(fn, delimiter=delimiter)
    #import pandas as pd
    #df=pd.read_csv(fn, sep=',',header=1)
    #data = df.values
    #return data[:,:-1], data[:, -1]

    matrix = []
    labels = []
    filter_features = set(filter_features)
    feature_index = []

    for line in open(fn):
        if header:
            if filter_features:
                for i, f in enumerate(line.split(',')):
                    if f in filter_features:
                        feature_index.append(i)
            header=False
            continue
        items = line.strip('\n').split(',')
        row = [1 if items[i] == 'yes' else 0 for i in feature_index]
        matrix.append(row)
        labels.append(items[-1])
    return np.array(matrix), np.array(labels)

if __name__=='__main__':
    import time
    start_time = time.time()

    id_matrix, features, id_class = read_matrix("../data/xj_gender_product_8000.dat", 6, frequency=False, norm=False, _class=True, class_col=2, class_type='age', times=1 )
    print("Samples:%d"%len(id_matrix))

    output = '../data/xj_matrix_age_brand_8000.csv'
    write_matrix(output, id_matrix, id_class, class_type="age")

    with open('../data/xj_age_brand_8000_features.csv', 'w') as f:
        for i, feature in enumerate(features):
            f.write(("%d\t%s\n"%(i, feature)))

    print("Time Consuming:%f"%(time.time()-start_time))

