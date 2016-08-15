# -*- coding:utf-8 -*-

def stat(fn):
    d = {}
    s = set()
    for line in open(fn):
        _id, age = line.split('\t')[:2]
        if _id in s:
            continue
        s.add(_id)
        d[age] = d.get(age, 0) + 1

    for a, c in sorted(d.iteritems(), key=lambda x:x[0]):
        print a,c


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def convert_age(fn, output):
    fw = open(output, 'w')
    d = {}
    s = set()

    for line in open(fn):
        _id, age = line.split('\t')[:2]
        if not isfloat(age):
            continue
        _age = age
        age = int(float(age))
        if age == 0:
            continue
    
        Type = None
        if age < 20:
            Type = '<20'
        elif age >=20 and age < 30:
            Type = '20-30'
        elif age >=30 and age < 40:
            Type = '30-40'
        elif age >=40 and age < 50:
            Type = '40-50'
        elif age >=50 and age < 60:
            Type = '50-60'
        else:
            Type = '>60'
        
        if not _id in s:
            d[Type] = d.get(Type, 0) + 1
            s.add(_id)

        fw.write(line.replace(_age, Type))

    for a, c in sorted(d.iteritems(), key=lambda x:x[0]):
        print a,c

    fw.close()

if __name__=='__main__':
    stat('xj_phone_age_category.dat')
    convert_age('xj_phone_age_category.dat', 'xj_phone_ageinvertal_category.dat')



