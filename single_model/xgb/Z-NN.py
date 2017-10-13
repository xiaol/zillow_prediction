#encoding=utf8
import gc
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util import *
from sklearn.cluster import DBSCAN, Birch
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import tensorflow as tf
import itertools

from sklearn import preprocessing

from sklearn.cluster import MiniBatchKMeans

import selu
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

drop_cols = ['logerror','parcelid'] # ,'parcelid']
one_hot_encode_cols = ['rawcensustractandblock','censustractandblock','regionidneighborhood', 'airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid','heatingorsystemtypeid','storytypeid', 'regionidcity', 'regionidcounty','regionidneighborhood', 'regionidzip','hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag', 'propertylandusetypeid', 'propertycountylandusecode', 'propertyzoningdesc', 'typeconstructiontypeid', 'fips', 'pooltypeid10','pooltypeid2', 'pooltypeid7','decktypeid']


def MAE(yhat, y):
    return np.sum([abs(yhat[i] - y[i]) for i in range(len(yhat))]) / len(yhat)


def prepare_data(df, columns):
    df = pd.get_dummies(df, columns=columns, prefix=columns, sparse=True)
    return df


def mae(y, y_pred):
    return np.sum([abs(y[i] - y_pred[i]) for i in range(len(y))]) / len(y)


def get_features(df):
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    hard_date = datetime(2016,1,1)

    df['transaction_month'] = df['transactiondate'].dt.month.astype(np.int8)
    df['transaction_day'] = df['transactiondate'].dt.weekday.astype(np.int8)
    df['transaction_month_day'] = df['transactiondate'].dt.day.astype(np.int8)
    df['transaction_quarter'] = df['transactiondate'].dt.quarter.astype(np.int8)
    df['transaction_date'] = df['transactiondate'] - hard_date
    df['transaction_date'] = df['transaction_date'].dt.days

    df['transactiondate_quarter'] = df['transactiondate'].dt.quarter

    df = df.drop('transactiondate', axis=1)

    return df


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

print('Loading data ...')

train = pd.read_csv('../../data/train_2017.csv')
prop = pd.read_csv('../../data/properties_2017.csv').fillna(0)  # , nrows=500)
sample = pd.read_csv('../../data/sample_submission.csv')

string_cols = []
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == object or c in one_hot_encode_cols:
        # prop[c] = prop[c].astype(np.int32)  # categorical_column_with_hash_bucket only support string and int
        string_cols.append(c)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))

# string_cols.extend(one_hot_encode_cols)
string_cols = set(string_cols)

print('Creating training set ...')
# train = train.sort_values('transactiondate')
# train = train[train.transactiondate < '2017-01-01']
# split = train[train.transactiondate < '2016-10-01'].shape[0]
#print(split)

train = train[train.logerror > -0.4]
train = train[train.logerror < 0.419]

# prop['latitude'] = prop['latitude']*1e-6
# prop['longitude'] = prop['longitude']*1e-6

# prop['censustractandblock'] /= 1e12



# brc = Birch(branching_factor=5, n_clusters=None, threshold=0.02, compute_labels=True)
# prop['loc_label'] = brc.fit_predict(prop[['latitude', 'longitude']])


df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train
x_train = get_features(x_train)
# x_train = prepare_data(x_train, one_hot_encode_cols)
x_train = x_train.drop(drop_cols, axis=1)


# x_train = x_train.drop('censustractandblock', axis=1)

train_columns = x_train.columns
numeric_cols = set(train_columns)-set(string_cols)

where_are_nan = np.isnan(x_train)
where_are_inf = np.isinf(x_train)
x_train[where_are_nan] = 0
x_train[where_are_inf] = 0

assert not np.any(np.isnan(x_train))
assert not np.any(np.isinf(x_train))

scaler_dict = {}
for n_col in numeric_cols:
    scaler = preprocessing.StandardScaler()#MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(x_train[n_col].values.reshape(-1,1))
    x_train[n_col] = scaler.transform(x_train[n_col].values.reshape(-1,1))
    scaler_dict[n_col] = scaler

y_train = df_train['logerror'].values

'''
y_scaler = preprocessing.StandardScaler()
y_scaler.fit(y_train.reshape(-1,1))
y_train = y_scaler.transform(y_train.reshape(-1, 1))
'''
print(x_train.shape, y_train.shape)
print x_train.columns
pd.Series(list(x_train.columns)).to_csv('../../data/columns.csv')
'''
select_qtr4 = df_train["transactiondate_quarter"] == 4

y_train = y_train[~select_qtr4]
x_train = x_train[~select_qtr4]
x_valid = x_train[select_qtr4]
y_valid = y_train[select_qtr4]
'''
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, stratify=x_train['transaction_month'].values, train_size=0.98, random_state=1)
# x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
del df_train; gc.collect()

print('Training ...')

model_dir = "../../data/model7/"


feature_cols = [tf.feature_column.numeric_column(k) for k in numeric_cols]



for string_col in string_cols:
    voca_list = map(int,list(prop[string_col].unique()))
    feature_category_col = tf.feature_column.categorical_column_with_vocabulary_list(key=string_col, vocabulary_list=voca_list, dtype=tf.as_dtype(prop[string_col].dtype))
    if len(voca_list) < 3:
        emb_dim = len(voca_list)
    else:
        emb_dim = max(int(np.log(len(voca_list))),1)
    feature_category_col_emb = tf.feature_column.embedding_column(feature_category_col, dimension=emb_dim)
    feature_cols.append(feature_category_col_emb)


print(len(feature_cols))
hidden_units = []
hidden_units.extend([1024, 512])
hidden_units.extend([])
print(hidden_units)
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=hidden_units,
                                      model_dir=model_dir)   #=tf.train.AdagradOptimizer(learning_rate=0.003))

LABEL = 'logerror'


def get_input_fn(data_set, label, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in train_columns}),
      y=pd.Series(label),
      num_epochs=num_epochs,
      shuffle=shuffle)

regressor.train(input_fn=get_input_fn(x_train, y_train), steps=2500)

for i in range(2000):
    print(str(i))

    ev = regressor.evaluate(
        input_fn=get_input_fn(x_train, y_train, num_epochs=1, shuffle=False))

    loss_score = ev["loss"]
    print("Train Loss: {0:f}".format(loss_score))

    ev = regressor.evaluate(
        input_fn=get_input_fn(x_valid, y_valid, num_epochs=1, shuffle=False))

    loss_score = ev["loss"]
    print("Valid Loss: {0:f}".format(loss_score))


    y = regressor.predict(
        input_fn=get_input_fn(x_train, [0] * y_train.shape[0], num_epochs=1, shuffle=False))
    # .predict() returns an iterator of dicts; convert to a list and print
    # predictions
    predictions = list(p["predictions"][0] for p in itertools.islice(y, x_train.shape[0]))

    mae = MAE(y_train, predictions)
    print("Train MAE: {}".format(mae))

    y = regressor.predict(
        input_fn=get_input_fn(x_valid, [0] * y_valid.shape[0], num_epochs=1, shuffle=False))
    # .predict() returns an iterator of dicts; convert to a list and print
    # predictions
    predictions = list(p["predictions"][0] for p in itertools.islice(y, x_valid.shape[0]))

    mae = MAE(y_valid, predictions)
    print("Valid MAE: {}".format(mae))
    if mae < 0.0511:
        print('hit')
        # break

    regressor.train(input_fn=get_input_fn(x_train, y_train), steps=50)

raw_input("Enter something to continue ...")
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

        # x_test_fold = prepare_data(x_test_fold, one_hot_encode_cols)
        # transactiondate = '2017-12-01'

        sub_cols = set(train_columns).intersection(set(x_test_fold.columns))
        x_test_fold = x_test_fold[list(sub_cols)]
        sLength = x_test_fold.shape[0]
        for train_col in train_columns:
            if train_col not in x_test_fold.columns:
                x_test_fold[train_col] = pd.Series(np.zeros((sLength)), index=x_test_fold.index)

        x_test_fold = x_test_fold[train_columns.tolist()]


        where_are_nan = np.isnan(x_test_fold)
        where_are_inf = np.isinf(x_test_fold)
        x_test_fold[where_are_nan] = 0
        x_test_fold[where_are_inf] = 0

        for n_col in numeric_cols:
            x_test_fold[n_col] = scaler_dict[n_col].transform(x_test_fold[n_col].values.reshape(-1,1))

        # predict p_test_cks with x_test_fold
        p_test_iter = regressor.predict(input_fn=get_input_fn(x_test_fold, [0]*x_test_fold.shape[0], num_epochs=1, shuffle=False))

        p_test_cks = list(p["predictions"][0] for p in itertools.islice(p_test_iter, x_test_fold.shape[0]))
        p_test = np.append(p_test, p_test_cks)

        del df_test_fold, x_test_fold; gc.collect()

    print(c)

    sub[c] = p_test

del prop, sample; gc.collect()

print('Writing csv ...')
file_path = '../../data/nn' + datetime.now().strftime("%m_%d_%H_%M_%S")+'.csv'
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

