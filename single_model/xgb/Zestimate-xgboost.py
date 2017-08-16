
import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from sklearn.preprocessing import OneHotEncoder

drop_cols = ['parcelid', 'logerror', 'transactiondate', 'latitude', 'longitude','propertyzoningdesc']
# [257]	train-mae:0.052669	valid-mae:0.051891
# [278]	train-mae:0.052569	valid-mae:0.051863
one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid','heatingorsystemtypeid', 'storytypeid', 'regionidcity', 'regionidcounty','regionidneighborhood', 'regionidzip', 'hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag', 'propertylandusetypeid', 'propertycountylandusecode']


def prepare_data(df, columns):
    df = pd.get_dummies(df, columns=columns, prefix=columns, sparse=True)
    return df


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

print('Loading data ...')

train = pd.read_csv('../../data/train_2016_v2.csv')
prop = pd.read_csv('../../data/properties_2016.csv')
sample = pd.read_csv('../../data/sample_submission.csv')

# prop = prop.head(1000)

prop = prepare_data(prop, one_hot_encode_cols)
print(prop.shape)

print('Binding to float32')
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid', copy=False)

# drop outliers
df_train = df_train[df_train.logerror > -0.4]
df_train = df_train[df_train.logerror < 0.419]

x_train = df_train.drop(drop_cols, axis=1)
train_columns = x_train.columns

y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)
print x_train.dtypes


del df_train; gc.collect()

split = 80000
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
params['min_child_weight'] = 20
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

sample['parcelid'] = sample['ParcelId']
p_test = np.array([])

print(sample.shape)

for fold in chunks(sample, 50000):
    df_test_fold = fold.merge(prop, on='parcelid', how='left', copy=False)
    x_test_fold = df_test_fold[train_columns]

    d_test_cks = xgb.DMatrix(x_test_fold)
    p_test_cks = clf.predict(d_test_cks)

    p_test = np.append(p_test, p_test_cks)

    del d_test_cks; gc.collect()
    del df_test_fold, x_test_fold; gc.collect()


del prop, sample; gc.collect()
print(p_test.shape)

sub = pd.read_csv('../../data/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

print('Writing csv ...')
sub.to_csv('../../data/xgb.csv', index=False, float_format='%.4f')


# Thanks to @inversion
