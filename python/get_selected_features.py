# -*- coding:utf-8 -*-

if __name__=="__main__":
    import sys

    selected_features_file, features_id_file = sys.argv[1], sys.argv[2]
    output = sys.argv[3]

    id_feature = dict((line.strip('\n').split('\t')) for line in open(features_id_file))
    filter_id_feature = {}

    fw = open(output, 'w')
    for line in open(selected_features_file):
        i = line.strip('\n').strip('A')
        if i in id_feature:
            fw.write(id_feature[i]+'\t'+i+'\n')
            #filter_id_feature[int(i)] = id_feature[i]
    #for k, v in sorted(filter_id_feature.iteritems(), key=lambda x:x[0]):
    #    fw.write(v+'\n')
    #fw.close()

