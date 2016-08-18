# -*- coding:utf-8 -*-

def class_first_catalog(fn):
    d_catalog = {}
    d_class = {}
    s = set()
    for line in open(fn):
        _id, clazz = line.strip('\n').strip().split('\t')[:2]
        fc = line.strip('\n').strip().split('\t')[2]
        if (_id, fc) in s:
            continue
        s.add((_id, fc))
        if not fc in d_catalog:
            d_catalog[fc] = {}
        d_catalog[fc][clazz] = d_catalog[fc].get(clazz, 0) + 1

        if not clazz in d_class:
            d_class[clazz] = {}
        d_class[clazz][fc] = d_class[clazz].get(fc, 0) + 1

    for cata, clazz_d in d_catalog.items():
        print cata, clazz_d
    for clazz, cata_d in d_class.items():
        print clazz
        Sum = sum(cata_d.values())
        for cata, count in sorted(cata_d.items(), key=lambda x:x[1], reverse=True):
            print cata, count, count*1.0/Sum

#平均点击次数
def class_click_number(fn):
    class_click = {} #每个class的点击次数
    class_count = {} #每个class的购物行为次数
    tmp_id = None
    for line in open(fn):
        _id, clazz = line.strip('\n').strip().split('\t')[:2]
        if tmp_id != _id:
            class_count[clazz] = class_count.get(clazz, 0) + 1
            tmp_id = _id
        class_click[clazz] = class_click.get(clazz, 0) + 1

    for clazz, click in class_click.items():
        print clazz, click*1.0/class_count[clazz]

#物品个数
def class_page_number(fn):
    class_iddate_page = {}
    for line in open(fn):
        items = line.strip('\n').strip().split('\t')
        _id, clazz = items[:2]
        date = items[-1][:8]
        page_string = items[-2]
        if not clazz in class_iddate_page:
            class_iddate_page[clazz] = {}
        k = (_id, date)
        if not k in class_iddate_page[clazz]:
            class_iddate_page[clazz][k] = set()

        class_iddate_page[clazz][k].add(page_string)
    #for clazz, iddate_page in class_iddate_page.items():
    #    for iddate, page in sorted(iddate_page.items(), key=lambda x:len(x[1]), reverse=True):
    #        print clazz, iddate, len(page)
    for clazz, iddate_page in class_iddate_page.items():
        print clazz, sum([len(pages) for pages in iddate_page.values()]) *1.0/len(iddate_page)

def class_keyword(fn):
    keywords = ['男', '女']
    class_keyword_count = {}
    for line in open(fn):
        items = line.strip('\n').strip().split('\t')
        _id, clazz = items[:2]
        product = items[-4]
        first = items[2]
        second = items[3]
        third = items[4]
        if not clazz in class_keyword_count:
            class_keyword_count[clazz] = {}
        if '男' in product and '女' in product:
            continue
        for k in keywords:
            if k in product or k in first or k in second or k in third:
                class_keyword_count[clazz][k] = class_keyword_count[clazz].get(k, 0) + 1

    for clazz, keyword_count in class_keyword_count.items():
        for key, c in keyword_count.items():
            print clazz, key, c

#一次购物行为的停留时长
def class_duration(fn):
    class_id_datetime = {}
    for line in open(fn):
        items = line.strip('\n').strip().split('\t')
        _id, clazz = items[:2]
        datetime = items[-1]
        if not clazz in class_id_datetime:
            class_id_datetime[clazz] = {}
        if not _id in class_id_datetime[clazz]:
            class_id_datetime[clazz][_id] = set()
        class_id_datetime[clazz][_id].add(datetime)

    for clazz, id_datetime in class_id_datetime.items():
        s_duration = 0
        s_count = 0
        for _id, datetimes in id_datetime.items():
            #datetimes = [long(d)/100 for d in datetimes]
            datetimes = [long(d) for d in datetimes]
            datetimes.sort()
            prev = datetimes[0]
            durations = []
            dura = 0
            for d in datetimes[1:]:
                if d == prev:
                    continue
                if d-prev <= 30*60:
                    dura += (d-prev)
                else:
                    durations.append(dura)
                    dura = 0
                prev = d
            if dura > 0:
                durations.append(dura)
            if not durations:
                s_duration += 1 
                s_count += 1
            else:
                s_duration += sum(durations)
                s_count += len(durations)
        if s_count == 0:
            continue
        print clazz, s_duration*1.0/s_count


if __name__ == '__main__':
    #class_first_catalog('xj_phone_gender_category.dat')
    #class_click_number('xj_phone_gender_product.dat.uniq')
    class_page_number('xj_phone_gender_product.dat.uniq')
    #class_keyword('xj_phone_gender_product.dat.uniq')
    #class_duration('xj_phone_gender_product.dat.uniq')

