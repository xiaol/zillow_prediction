
import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from sklearn.preprocessing import OneHotEncoder

drop_cols = ['parcelid','logerror']  # 'latitude', 'longitude']
one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid','heatingorsystemtypeid', 'storytypeid', 'regionidcity', 'regionidcounty','regionidneighborhood', 'regionidzip', 'hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag', 'propertylandusetypeid', 'propertycountylandusecode', 'propertyzoningdesc']


def prepare_data(df, columns):
    df = pd.get_dummies(df, columns=columns, prefix=columns, sparse=True)
    return df


def get_features(df):
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])

    df['transaction_month'] = df['transactiondate'].dt.month.astype(np.int8)
    df['transaction_day'] = df['transactiondate'].dt.weekday.astype(np.int8)

    df = df.drop('transactiondate', axis=1)
    return df

def chunks(l, n):

    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

print('Loading data ...')

train = pd.read_csv('../../data/train_2016_v2.csv')
prop = pd.read_csv('../../data/properties_2016.csv')
sample = pd.read_csv('../../data/sample_submission.csv')

print('Binding to float32')
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')
train = train.sort_values('transactiondate')
train = train[train.transactiondate < '2017-01-01']
split = train[train.transactiondate < '2016-10-01'].shape[0]
print(split)
df_train = train.merge(prop, how='left', on='parcelid')

# drop outliers
# df_train = df_train[df_train.logerror > -0.4]
# df_train = df_train[df_train.logerror < 0.419]


x_train = df_train.drop(drop_cols, axis=1)
x_train = prepare_data(x_train, one_hot_encode_cols)
x_train = get_features(x_train)

train_columns = x_train.columns

y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)
print x_train.dtypes


del df_train; gc.collect()


x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['min_child_weight'] = 1
params['colsample_bytree'] = 0.2
params['max_depth'] = 5
params['lambda'] = 0.3
params['alpha'] = 0.6
params['silent'] = 1


watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

#xgb.plot_importance(clf)
del d_train, d_valid

print('Building test set ...')

print('Predicting on test ...')
sub = pd.read_csv('../../data/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
print(sample.shape)

for c in sub.columns[sub.columns != 'ParcelId']:
    if c > '201709':
        sub[c] = p_test
        continue
    p_test = np.array([])

    for fold in chunks(sample, 80000):
        df_test_fold = fold.merge(prop, on='parcelid', how='left')
        x_test_fold = prepare_data(df_test_fold, one_hot_encode_cols)
        transactiondate = c[:4] + '-' + c[4:] +'-01'
        # transactiondate = '2017-12-01'
        x_test_fold['transactiondate'] = transactiondate
        x_test_fold = get_features(x_test_fold)

        sub_cols = set(train_columns).intersection(set(x_test_fold.columns))
        x_test_fold = x_test_fold[list(sub_cols)]
        sLength = x_test_fold.shape[0]
        for train_col in train_columns:
            if train_col not in x_test_fold.columns:
                x_test_fold[train_col] = pd.Series(np.zeros((sLength)), index=x_test_fold.index)

        x_test_fold = x_test_fold[train_columns.tolist()]

        d_test_cks = xgb.DMatrix(x_test_fold)
        p_test_cks = clf.predict(d_test_cks)

        p_test = np.append(p_test, p_test_cks)

        del d_test_cks; gc.collect()
        del df_test_fold, x_test_fold; gc.collect()

    sub[c] = p_test

del prop, sample; gc.collect()

print('Writing csv ...')
sub.to_csv('../../data/xgb.csv', index=False, float_format='%.4f')

# #1 0.0645657  [270]	train-mae:0.052583	valid-mae:0.051849

# #2 0.0644769  [577]	train-mae:0.051888	valid-mae:0.051766

# [977]	train-mae:0.051399	valid-mae:0.051709

# 0.0644580 [670]	train-mae:0.051573	valid-mae:0.051681 ,
# add coordinate feature, remove weekday feature, just predict 20171201 for speed.

# [303]	train-mae:0.067442	valid-mae:0.066382

# Thanks to @inversion