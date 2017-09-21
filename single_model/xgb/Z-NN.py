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

import selu

drop_cols = ['logerror','parcelid']
one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid','heatingorsystemtypeid','storytypeid', 'regionidcity', 'regionidcounty','regionidneighborhood', 'regionidzip','hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag', 'propertylandusetypeid', 'propertycountylandusecode', 'propertyzoningdesc', 'typeconstructiontypeid', 'fips']


def MAE(yhat, y):
    return np.sum([abs(yhat[i] - y[i]) for i in range(len(yhat))]) / len(yhat)


def prepare_data(df, columns):
    df = pd.get_dummies(df, columns=columns, prefix=columns, sparse=True)
    return df


def mae(y, y_pred):
    return np.sum([abs(y[i] - y_pred[i]) for i in range(len(y))]) / len(y)


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

    # -----------------------------------------------------------------------------------------------

    # life of property
    df_train['N-life'] = 2018 - df_train['yearbuilt']

    # error in calculation of the finished living area of home
    df_train['N-LivingAreaError'] = df_train['calculatedfinishedsquarefeet'] / df_train['finishedsquarefeet12']

    # proportion of living area
    df_train['N-LivingAreaProp'] = df_train['calculatedfinishedsquarefeet'] / df_train['lotsizesquarefeet']
    df_train['N-LivingAreaProp2'] = df_train['finishedsquarefeet12'] / df_train['finishedsquarefeet15']

    # Amout of extra space
    df_train['N-ExtraSpace'] = df_train['lotsizesquarefeet'] - df_train['calculatedfinishedsquarefeet']
    df_train['N-ExtraSpace-2'] = df_train['finishedsquarefeet15'] - df_train['finishedsquarefeet12']

    # Total number of rooms
    df_train['N-TotalRooms'] = df_train['bathroomcnt'] * df_train['bedroomcnt']

    # Average room size
    df_train['N-AvRoomSize'] = df_train['calculatedfinishedsquarefeet'] / df_train['roomcnt']

    # Number of Extra rooms
    df_train['N-ExtraRooms'] = df_train['roomcnt'] - df_train['N-TotalRooms']

    # Ratio of the built structure value to land area
    df_train['N-ValueProp'] = df_train['structuretaxvaluedollarcnt'] / df_train['landtaxvaluedollarcnt']

    # Does property have a garage, pool or hot tub and AC?
    df_train['N-GarPoolAC'] = ((df_train['garagecarcnt'] > 0) & (df_train['pooltypeid10'] > 0) & (
    df_train['airconditioningtypeid'] != 5)) * 1

    df_train["N-location"] = df_train["latitude"] + df_train["longitude"]
    df_train["N-location-2"] = df_train["latitude"] * df_train["longitude"]
    df_train["N-location-2round"] = df_train["N-location-2"].round(-4)

    df_train["N-latitude-round"] = df_train["latitude"].round(-4)
    df_train["N-longitude-round"] = df_train["longitude"].round(-4)


    # ---------------------------------
    # Ratio of tax of property over parcel
    df_train['N-ValueRatio'] = df_train['taxvaluedollarcnt'] / df_train['taxamount']

    # TotalTaxScore
    df_train['N-TaxScore'] = df_train['taxvaluedollarcnt'] * df_train['taxamount']

    # polnomials of tax delinquency year
    df_train["N-taxdelinquencyyear-2"] = df_train["taxdelinquencyyear"] ** 2
    df_train["N-taxdelinquencyyear-3"] = df_train["taxdelinquencyyear"] ** 3

    # Length of time since unpaid taxes
    df_train['N-life-tax'] = 2018 - df_train['taxdelinquencyyear']

    #-------------------------------------------

    # Indicator whether it has AC or not
    df_train['N-ACInd'] = (df_train['airconditioningtypeid'] != 5) * 1

    # Indicator whether it has Heating or not
    df_train['N-HeatInd'] = (df_train['heatingorsystemtypeid'] != 13) * 1

    # There's 25 different property uses - let's compress them down to 4 categories
    df_train['N-PropType'] = df_train.propertylandusetypeid.replace(
        {31: "Mixed", 46: "Other", 47: "Mixed", 246: "Mixed", 247: "Mixed", 248: "Mixed", 260: "Home", 261: "Home",
         262: "Home", 263: "Home", 264: "Home", 265: "Home", 266: "Home", 267: "Home", 268: "Home", 269: "Not Built",
         270: "Home", 271: "Home", 273: "Home", 274: "Other", 275: "Home", 276: "Home", 279: "Home", 290: "Not Built",
         291: "Not Built"})

    #----------------------------------------------

    # polnomials of the variable
    df_train["N-structuretaxvaluedollarcnt-2"] = df_train["structuretaxvaluedollarcnt"] ** 2
    df_train["N-structuretaxvaluedollarcnt-3"] = df_train["structuretaxvaluedollarcnt"] ** 3

    # Average structuretaxvaluedollarcnt by city
    group = df_train.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
    df_train['N-Avg-structuretaxvaluedollarcnt'] = df_train['regionidcity'].map(group)

    # Deviation away from average
    df_train['N-Dev-structuretaxvaluedollarcnt'] = abs(
        (df_train['structuretaxvaluedollarcnt'] - df_train['N-Avg-structuretaxvaluedollarcnt'])) / df_train[
                                                       'N-Avg-structuretaxvaluedollarcnt']


    # ----------------------------------------------------


    return df


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

print('Loading data ...')

train = pd.read_csv('../../data/train_2016_v2.csv')
prop = pd.read_csv('../../data/properties_2016.csv').fillna(-1)  # , nrows=500)
sample = pd.read_csv('../../data/sample_submission.csv')

string_cols = []
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == object:
        # prop[c] = prop[c].astype(np.int32)  # categorical_column_with_hash_bucket only support string and int
        string_cols.append(c)

string_cols.extend(one_hot_encode_cols)
string_cols = set(string_cols)

print('Creating training set ...')
train = train.sort_values('transactiondate')
train = train[train.transactiondate < '2017-01-01']
split = train[train.transactiondate < '2016-10-01'].shape[0]
print(split)

train = train[train.logerror > -0.4]
train = train[train.logerror < 0.419]

prop['latitude'] = prop['latitude']*1e-6
prop['longitude'] = prop['longitude']*1e-6

prop['censustractandblock'] /= 1e12


brc = Birch(branching_factor=50, n_clusters=None, threshold=0.03, compute_labels=True)
prop['loc_label'] = brc.fit_predict(prop[['latitude', 'longitude']])
print('Number of loc label: {}'.format(len(set(prop['loc_label']))))

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train
x_train = get_features(x_train)
# x_train = prepare_data(x_train, one_hot_encode_cols)
x_train = x_train.drop(drop_cols, axis=1)

le_dict = {}
for str_col in string_cols:
    le = preprocessing.LabelEncoder()
    le.fit(prop[str_col])
    x_train[str_col] = le.transform(x_train[str_col])
    le_dict[str_col] = le

# x_train = x_train.drop('censustractandblock', axis=1)

train_columns = x_train.columns
numeric_cols = set(train_columns)-set(string_cols)
for n_col in numeric_cols:
    x_train[n_col] = (x_train[n_col] - np.mean(x_train[n_col])) / (np.std(x_train[n_col]) + 1)

y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)
print x_train.columns
pd.Series(list(x_train.columns)).to_csv('../../data/columns.csv')


del df_train; gc.collect()


x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Training ...')

model_dir = "../../data/model2/"


feature_cols = [tf.feature_column.numeric_column(k) for k in numeric_cols]
feature_category_cols = [tf.feature_column.categorical_column_with_hash_bucket(k, hash_bucket_size=1000, dtype=tf.int64) for k in string_cols]
feature_category_cols_emb = [tf.feature_column.embedding_column(k, dimension=8) for k in feature_category_cols]
feature_cols.extend(feature_category_cols_emb)
print(len(feature_cols))
hidden_units = [256]*32
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=hidden_units,
                                      model_dir=model_dir)

LABEL = 'logerror'


def get_input_fn(data_set, label, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in train_columns}),
      y=pd.Series(label),
      num_epochs=num_epochs,
      shuffle=shuffle)

regressor.train(input_fn=get_input_fn(x_train, y_train), steps=5000)

ev = regressor.evaluate(
    input_fn=get_input_fn(x_valid, y_valid, num_epochs=1, shuffle=False))

loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

y = regressor.predict(
    input_fn=get_input_fn(x_valid, [0]*x_valid.shape[0], num_epochs=1, shuffle=False))
# .predict() returns an iterator of dicts; convert to a list and print
# predictions
predictions = list(p["predictions"][0] for p in itertools.islice(y, x_valid.shape[0]))
mae = MAE(y_valid, predictions)
print("Valid MAE: {}".format(mae))

# raw_input("Enter something to continue ...")
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

        for str_col in string_cols:
            x_test_fold[str_col] = le_dict[str_col].transform(x_test_fold[str_col])

        for n_col in numeric_cols:
            x_test_fold[n_col] = (x_test_fold[n_col] - np.mean(x_test_fold[n_col])) / (np.std(x_test_fold[n_col]) + 1)

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

