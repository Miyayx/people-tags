# -*- coding:utf-8 -*-
fw = open('xinjiang_profile_wx_balance.dat', 'w')
s = set()
girls = set()
for line in open('xinjiang_profile_wx.dat'):
    _id, g = line.split('\t')[:2]
    if g == '男' and len(s) < 32000 :
        fw.write(line)
        s.add(_id+g)
    elif g == '女' and len(girls) < 32000:
        fw.write(line)
        girls.add(_id+g)
fw.close()
