# -*- coding:utf-8 -*-

user_brands = {}
user_gender = {}
for line in open('xj_phone_gender_product_balance.dat'):
    items = line.split('\t')
    user, brand, gender = items[0], items[5], items[1]
    if not user in user_brands:
        user_brands[user] = []
        user_gender[user] = gender
    if brand == '-1':
        continue
    user_brands[user].append(brand)

fw = open('xj_gender.examples', 'w')
for user, brands in user_brands.iteritems():
    fw.write("%s\t%s\t%s\n"%(user, ";;;".join(brands), user_gender[user]))
fw.close()




