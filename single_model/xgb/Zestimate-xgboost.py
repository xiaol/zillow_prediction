#encoding=utf8
import gc
import os
import sys
from datetime import datetime

import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util import *
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

drop_cols = ['parcelid', 'logerror']
one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid','heatingorsystemtypeid','storytypeid', 'regionidcity', 'regionidcounty','regionidneighborhood', 'regionidzip','hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag', 'propertylandusetypeid', 'propertycountylandusecode', 'propertyzoningdesc', 'typeconstructiontypeid', 'fips']


def prepare_data(df, columns):
    df = pd.get_dummies(df, columns=columns, prefix=columns, sparse=True)
    return df


def get_features(df):
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])

    df['transaction_month'] = df['transactiondate'].dt.month.astype(np.int8)
    df['transaction_day'] = df['transactiondate'].dt.weekday.astype(np.int8)

    df = df.drop('transactiondate', axis=1)
    # df['tax_rt'] = df['taxamount'] / df['taxvaluedollarcnt']
    df['extra_bathroom_cnt'] = df['bathroomcnt'] - df['bedroomcnt']
    df['room_sqt'] = df['calculatedfinishedsquarefeet']/(df['roomcnt'] + 1)
    # df['structure_tax_rt'] = df['structuretaxvaluedollarcnt'] / df['taxvaluedollarcnt']
    '''
    df['land_tax_rt'] = df['landtaxvaluedollarcnt'] / df['taxvaluedollarcnt']
    '''

    # 商圈内待售房屋数量
    df = merge_nunique(df, ['loc_label'], 'parcelid', 'loc_building_num')
    df = merge_nunique(df, ['regionidzip'], 'parcelid', 'region_property_num')
    df = merge_nunique(df, ['regionidcity'], 'parcelid', 'city_property_num')
    # df = merge_nunique(df, ['regionidcounty'], 'parcelid', 'county_property_num')

    # df = merge_count(df, ['transaction_month','regionidcity'], 'parcelid', 'city_month_transaction_count')
    # 商圈房屋状况均值
    # df = merge_median(df, ['regionidcity'], 'buildingqualitytypeid', 'city_quality_median')
    for col in ['finishedsquarefeet12', 'garagetotalsqft', 'yearbuilt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet',
                'unitcnt', 'poolcnt']:
        df = merge_mean(df, ['loc_label'], col, 'loc_'+col+'_mean')

    return df


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

print('Loading data ...')

train = pd.read_csv('../../data/train_2016_v2.csv')
prop = pd.read_csv('../../data/properties_2016.csv').fillna(-0.001)  # , nrows=500)
sample = pd.read_csv('../../data/sample_submission.csv')
'''
print('Binding to float32')
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)
'''

print('Creating training set ...')
train = train.sort_values('transactiondate')
train = train[train.transactiondate < '2017-01-01']
split = train[train.transactiondate < '2016-10-01'].shape[0]
print(split)

train = train[train.logerror > -0.4]
train = train[train.logerror < 0.419]

prop['latitude'] = prop['latitude']*1e-6
prop['longitude'] = prop['longitude']*1e-6

df_coor = prop[['parcelid', 'latitude', 'longitude']]
df_coor = df_coor.round(3)
df_coor.rename(columns={'latitude':'lati', 'longitude':'longi'}, inplace=True)
prop = prop.merge(df_coor, how='left', on='parcelid')
df_coor = pd.DataFrame(df_coor.groupby(['lati', 'longi'])['parcelid'].count()).reset_index()

db = DBSCAN(eps=5/6371., min_samples=5, algorithm='ball_tree',
            metric='haversine').fit(np.radians(df_coor[['lati', 'longi']]), sample_weight=df_coor['parcelid'].values)
df_coor.loc[:, 'loc_label'] = db.labels_
print(np.sum(db.labels_ == -1))
num_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print('Number of clusters: {}'.format(num_clusters))

prop = prop.merge(df_coor, how='left', on=['lati', 'longi'])
prop.drop(['lati','longi'], axis=1)
prop[['parcelid', 'loc_label']].to_csv('../../data/loc_label.csv')

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train
x_train = get_features(x_train)
x_train = prepare_data(x_train, one_hot_encode_cols)
x_train = x_train.drop(drop_cols, axis=1)

train_columns = x_train.columns

y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)
print x_train.columns
pd.Series(list(x_train.columns)).to_csv('../../data/columns.csv')


del df_train; gc.collect()


# x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
# x_valid, y_valid = x_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
# d_valid = xgb.DMatrix(x_valid, label=y_valid)

# del x_train, x_valid; gc.collect()
del x_train; gc.collect()

print('Training ...')

params = {'eta': 0.015, 'objective': 'reg:linear', 'eval_metric': 'mae', 'min_child_weight': 1.5, 'colsample_bytree': 0.2, 'max_depth': 7, 'lambda': 0.3, 'alpha': 0.6, 'silent': 1}

print(params)

watchlist = [(d_train, 'train')]
# cross-validation
# TODO bad news 0.0644361 higher than previous CV set, interesting. 858
# remove cv. back to last point. and continue to test features.
# fold 2 , 0.0643877, overfitting is working. 620+-
'''
rint("Running XGBoost CV....")
res = xgb.cv(params, d_train, num_boost_round=2000, nfold=2,
                 early_stopping_rounds=100, verbose_eval=10, show_stdv=True)
num_best_rounds = len(res)
print("Number of best rounds: {}".format(num_best_rounds))
'''
num_best_rounds = 520
clf = xgb.train(params, d_train, num_best_rounds, watchlist, verbose_eval=10)  # watchlist,  early_stopping_rounds=100, verbose_eval=10)

fig, ax = plt.subplots(figsize=(20,40))
xgb.plot_importance(clf, max_num_features=200, height=0.8, ax=ax)
plt.savefig('../../data/importance.pdf')
# del d_train, d_valid
del d_train

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
        sys.stdout.write('.')
        sys.stdout.flush()

        df_test_fold = fold.merge(prop, on='parcelid', how='left')

        transactiondate = c[:4] + '-' + c[4:] +'-01'
        df_test_fold['transactiondate'] = transactiondate
        x_test_fold = get_features(df_test_fold)

        x_test_fold = prepare_data(x_test_fold, one_hot_encode_cols)
        # transactiondate = '2017-12-01'

        sub_cols = set(train_columns).intersection(set(x_test_fold.columns))
        x_test_fold = x_test_fold[list(sub_cols)]
        sLength = x_test_fold.shape[0]
        for train_col in train_columns:
            if train_col not in x_test_fold.columns:
                x_test_fold[train_col] = pd.Series(np.zeros((sLength)), index=x_test_fold.index)

        x_test_fold = x_test_fold[train_columns.tolist()]

        d_test_cks = xgb.DMatrix(x_test_fold)
        p_test_cks = clf.predict(d_test_cks) # , ntree_limit=clf.best_ntree_limit)

        p_test = np.append(p_test, p_test_cks)

        del d_test_cks; gc.collect()
        del df_test_fold, x_test_fold; gc.collect()

    print(c)

    sub[c] = p_test

del prop, sample; gc.collect()

print('Writing csv ...')
file_path = '../../data/xgb' + datetime.now().strftime("%m_%d_%H_%M_%S")+'.csv'
sub.to_csv(file_path, index=False, float_format='%.4f')

# #1 0.0645657  [270]	train-mae:0.052583	valid-mae:0.051849

# #2 0.0644769  [577]	train-mae:0.051888	valid-mae:0.051766

# [977]	train-mae:0.051399	valid-mae:0.051709

# 0.0644580 [670]	train-mae:0.051573	valid-mae:0.051681 ,
# add coordinate feature, remove weekday feature, just predict 20171201 for speed.

# [303]	train-mae:0.067442	valid-mae:0.066382

# 0.0644440 back to the start point, only change the validation set. [476]   train-mae:0.050504      valid-mae:0.052248

# 0.0644258 add loc_label , loc_quality_median, loc_building_num  [485]   train-mae:0.051334      valid-mae:0.052187

# Thanks to @inversion

