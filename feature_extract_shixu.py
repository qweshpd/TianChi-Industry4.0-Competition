#coding=utf-8
import numpy as np
import pandas as pd
import time
import os
from multiprocessing import Pool

params = ['tparam1', 'tparam18', 'tparam14', 'tparam2', 'tparam10', 'tparam7','tparam9', 'tparam3', 'tparam8', 'tparam11', 'tparam17', 'tparam4','tparam5', 'tparam6', 'tparam16', 'tparam15', 'tparam12', 'tparam13']
tv_train = pd.read_csv('/devdata2/qweshpd/Industrial data/zhongzhi_data/timevarying_param_train.csv',header=None)
tv_train.columns = ['product_no','key_index','param_name','param_value','add_time']
products = list(tv_train.product_no.unique())
tv_features_train = pd.DataFrame(columns=['product_no']+[p for p in params])
tv_features_test = pd.DataFrame(columns=['product_no']+[p for p in params])

def lenth_param(products):
    param_size_0 = 0
    for product in products:
        tv_product = tv_train[tv_train.product_no == product]
        for param in params:
            tv_product_param = tv_product[tv_product.param_name == param]
            param_size = tv_product_param.param_value.size
            if param_size > param_size_0:
                param_size_0 = param_size
    print(param_size_0)
    return param_size_0


def train_(product, param_size_0):
    tv_product = tv_train[tv_train.product_no==product]
    d = {'product_no': tv_product.product_no[:param_size_0]}
    d['product_no'] = pd.Series(list(tv_product.product_no[:param_size_0]))
    d['key_index'] = pd.Series(list(tv_product.key_index[:param_size_0]))
    tv_product.add_time = tv_product.add_time.apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
    tv_product = tv_product.sort_values(by=['add_time'])
    for param in params:
        tv_product_param = tv_product[tv_product.param_name==param]
        d[param] = pd.Series(list(tv_product_param.param_value))
    this_tv_features = pd.DataFrame(d)
    return this_tv_features

param_size_0 = lenth_param(products)

rst = []
pool = Pool(12)
for product in products[:5000]:
    #print(product)
    rst.append(pool.apply_async(train_, args=(product, param_size_0)))
pool.close()
pool.join()

rst = [i.get() for i in rst]
for i in rst:
    tv_features_train = pd.concat([tv_features_train, i], axis=0)

tv_features_train.to_csv('/devdata2/qweshpd/Industrial data/zhongzhi_data/feature/tv_features_train_shixu.csv',index=None)

rst = []
pool = Pool(12)
for product in products[5000:]:
    #print(product)
    rst.append(pool.apply_async(train_, args=(product, param_size_0)))
pool.close()
pool.join()

rst = [i.get() for i in rst]
for i in rst:
    tv_features_test = pd.concat([tv_features_test, i], axis=0)

tv_features_test.to_csv('/devdata2/qweshpd/Industrial data/zhongzhi_data/feature/tv_features_test_shixu.csv',index=None)

