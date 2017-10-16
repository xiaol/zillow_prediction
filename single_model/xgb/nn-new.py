import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import gc
import datetime as dt
import tensorflow as tf
import itertools

print('Loading Properties ...')
properties2016 = pd.read_csv('../../data/properties_2016.csv', low_memory=False)
properties2017 = pd.read_csv('../../data/properties_2017.csv', low_memory=False)

print('Loading Train ...')
train2016 = pd.read_csv('../../data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2017 = pd.read_csv('../../data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)


def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016) * 12 + df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016) * 4 + df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df


train2016 = add_date_features(train2016)
train2017 = add_date_features(train2017)

print('Loading Sample ...')
sample_submission = pd.read_csv('../../data/sample_submission.csv', low_memory=False)

print('Merge Train with Properties ...')
train2016 = pd.merge(train2016, properties2016, how='left', on='parcelid')
train2017 = pd.merge(train2017, properties2017, how='left', on='parcelid')

print('Tax Features 2017  ...')
train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan

print('Concat Train 2016 & 2017 ...')
train_df = pd.concat([train2016, train2017], axis=0)
test_df = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns={'parcelid': 'ParcelId'}), how='left',
                   on='ParcelId')

del properties2016, properties2017, train2016, train2017
gc.collect();

print('Remove missing data fields ...')

missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % len(exclude_missing))

del num_rows, missing_perc_thresh
gc.collect();

print ("Remove features with one unique value !!")
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % len(exclude_unique))

print ("Define training features !!")
exclude_other = ['parcelid', 'logerror', 'propertyzoningdesc']
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
            and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % len(train_features))

print ("Define categorial features !!")
cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
            and not 'sqft' in c \
            and not 'cnt' in c \
            and not 'nbr' in c \
            and not 'number' in c:
        cat_feature_inds.append(i)

print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

print ("Replacing NaN values by -999 !!")
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

print ("Training time !!")
X_train = train_df[train_features]
y_train = train_df.logerror
print(X_train.shape, y_train.shape)

test_df['transactiondate'] = pd.Timestamp('2017-12-01')
test_df = add_date_features(test_df)
X_test = test_df[train_features]
print(X_test.shape)

model_dir = "../../data/model7/"

numeric_cols = set(train_features)-set(cat_feature_inds)
feature_cols = [tf.feature_column.numeric_column(k) for k in numeric_cols]

for c, dtype in zip(test_df.columns, test_df.dtypes):
    if c in numeric_cols:
        test_df[c] = test_df[c].astype(np.string)  # categorical_column_with_hash_bucket only support string and int

for string_col in  [train_features[ind] for ind in cat_feature_inds]:
    voca_list = map(str,list(test_df[string_col].unique()))
    feature_category_col = tf.feature_column.categorical_column_with_vocabulary_list(key=string_col, vocabulary_list=voca_list, dtype=tf.string)
    if len(voca_list) < 10000:
        emb_dim = len(voca_list)
    else:
        emb_dim = max(int(np.log(len(voca_list))),1)
    feature_category_col_emb = tf.feature_column.embedding_column(feature_category_col, dimension=emb_dim)
    feature_cols.append(feature_category_col_emb)


print(len(feature_cols))
hidden_units = []
hidden_units.extend([1024, 512, 256])  # [2048, 1024, 1024, 512, 512, 200]
hidden_units.extend([])
print(hidden_units)
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=hidden_units,
                                      model_dir=model_dir)   #=tf.train.AdagradOptimizer(learning_rate=0.003))

LABEL = 'logerror'


def get_input_fn(data_set, label, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in train_features}),
      y=pd.Series(label),
      num_epochs=num_epochs,
      shuffle=shuffle)



num_ensembles = 5
y_pred = 0.0
for i in tqdm(range(num_ensembles)):
    y = regressor.train(input_fn=get_input_fn(X_train, y_train), steps=1000)
    X_test = list(p["predictions"][0] for p in itertools.islice(y, X_train.shape[0]))
    y_pred += regressor.predict(X_test)
y_pred /= num_ensembles

submission = pd.DataFrame({
    'ParcelId': test_df['ParcelId'],
})
test_dates = {
    '201610': pd.Timestamp('2016-09-30'),
    '201611': pd.Timestamp('2016-10-31'),
    '201612': pd.Timestamp('2016-11-30'),
    '201710': pd.Timestamp('2017-09-30'),
    '201711': pd.Timestamp('2017-10-31'),
    '201712': pd.Timestamp('2017-11-30')
}
for label, test_date in test_dates.items():
    print("Predicting for: %s ... " % (label))
    submission[label] = y_pred

submission.to_csv('../../data/Only_NN.csv', float_format='%.6f', index=False)