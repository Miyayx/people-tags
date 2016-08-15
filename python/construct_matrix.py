# -*- coding:utf-8 -*-


def generate_matrix(fn, output, class_col = 1, feature_col = 6):
    feature_list = list(set([line.split('\t')[feature_col] for line in open(fn)]))
    print "Feature Number:",len(feature_list)
    feature_index = dict((k, i) for i, k in enumerate(feature_list))

    id_matrix = {}
    
    for line in open(fn):
        items = line.split('\t')
        _id = items[0]
        clazz = items[class_col].strip()
        feature = items[feature_col].strip()
        
        if not _id in id_matrix:
            id_matrix[_id] = [0 for i in range(len(feature_list) + 1)]
        if clazz.isdigit():
            id_matrix[_id][-1] = int(clazz)
        elif clazz == '男':
            id_matrix[_id][-1] = 0
        elif clazz == '女':
            id_matrix[_id][-1] = 1
        else:
            continue

        id_matrix[_id][feature_index[feature]] += 1

    print "Writing to", output
    fw = open(output, 'w')
    #fw.write(','.join(['A'+str(i) for i in range(len(feature_list))])+',Class\n')
    fw.write(','.join(feature_list)+',Class\n')
    for features in id_matrix.values():
        Sum = sum(features[:-1])
        if Sum == 0:
            continue
        fs = features[:-1]
        #features = [f*1.0/Sum for f in fs] + [features[-1]]
        features = [f for f in fs] + [features[-1]]
        fw.write(','.join([str(f) for f in features[:-1]]))
        if features[-1] == 1:
            fw.write(',female\n')
        else:
            fw.write(',male\n')
    fw.close()


if __name__ == '__main__':
    generate_matrix('../data/xinjiang_profile_wx_balance.dat', '../data/xj_matrix_gender_wx_account_fre_balance.csv')
    #train_test('xj_matrix_v2_fre.csv')


