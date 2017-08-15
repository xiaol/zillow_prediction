#!/usr/bin/python

import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from sklearn.preprocessing import OneHotEncoder


def prepare_data(df, columns):
    df = pd.get_dummies(df, columns=columns, prefix=columns, sparse=True)
    return df

print('Loading data ...')

train = pd.read_csv('../../data/train_2016_v2.csv')
prop = pd.read_csv('../../data/properties_2016.csv')
sample = pd.read_csv('../../data/sample_submission.csv')

print('Binding to float32')
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')
drop_cols = ['parcelid', 'logerror', 'transactiondate', 'latitude', 'longitude']

one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid','heatingorsystemtypeid', 'storytypeid', 'regionidcity', 'regionidcounty','regionidneighborhood', 'regionidzip', 'hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag',  'propertyzoningdesc', 'propertycountylandusecode']

df_train = train.merge(prop, how='left', on='parcelid')

# drop outliers
df_train = df_train[df_train.logerror > -4]
df_train = df_train[df_train.logerror < 4]

x_train = df_train.drop(drop_cols, axis=1)
train_columns = x_train.columns
x_train = prepare_data(x_train, one_hot_encode_cols)

y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)
print x_train.dtypes


del df_train; gc.collect()

split = 60000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.01
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['min_child_weight'] = 20
params['colsample_bytree'] = 0.2
params['max_depth'] = 12
params['lambda'] = 0.3
params['alpha'] = 0.6
params['silent'] = 1


watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

#xgb.plot_importance(clf)
del d_train, d_valid

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
print(df_test.shape)

del prop; gc.collect()

x_test = df_test[train_columns]
x_test = prepare_data(x_test, one_hot_encode_cols)


del df_test, sample; gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test; gc.collect()

sub = pd.read_csv('../../data/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

print('Writing csv ...')
sub.to_csv('../../data/xgb.csv', index=False, float_format='%.4f')




# Thanks to @inversion
