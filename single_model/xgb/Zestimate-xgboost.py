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

from bayes_opt import BayesianOptimization

# TODO propertyzoningdesc ,remove from one hot , regionidcity, regionidneighborhood, regionidzip,
drop_cols = ['parcelid', 'logerror', 'propertyzoningdesc'] # 'regionidcity', 'regionidneighborhood', 'regionidzip',]
one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid','heatingorsystemtypeid','storytypeid', 'regionidcounty','hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag', 'propertylandusetypeid', 'propertycountylandusecode', 'typeconstructiontypeid', 'fips']


def prepare_data(df, columns):
    df = pd.get_dummies(df, columns=columns, prefix=columns, sparse=True)
    return df


def get_features(df):
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    hard_date = datetime(2016,1,1)

    df['transaction_month'] = df['transactiondate'].dt.month.astype(np.int8)
    df['transaction_day'] = df['transactiondate'].dt.weekday.astype(np.int8)
    df['transaction_month_day'] = df['transactiondate'].dt.day.astype(np.int8)
    df['transaction_quarter'] = df['transactiondate'].dt.quarter.astype(np.int8)
    df['transaction_date'] = df['transactiondate'] - hard_date
    df['transaction_date'] = df['transaction_date'].dt.days

    df = df.drop('transactiondate', axis=1)
    df['tax_rt'] = df['taxamount'] / df['taxvaluedollarcnt']
    df['extra_bathroom_cnt'] = df['bathroomcnt'] - df['bedroomcnt']
    df['room_sqt'] = df['calculatedfinishedsquarefeet']/(df['roomcnt'] + 1)
    df['structure_tax_rt'] = df['structuretaxvaluedollarcnt'] / df['taxvaluedollarcnt']
    df['land_tax_rt'] = df['landtaxvaluedollarcnt'] / df['taxvaluedollarcnt']

    # 商圈内待售房屋数量
    df = merge_nunique(df, ['loc_label'], 'parcelid', 'loc_building_num')
    df = merge_nunique(df, ['regionidzip'], 'parcelid', 'region_property_num')
    df = merge_nunique(df, ['regionidcity'], 'parcelid', 'city_property_num')
    df = merge_nunique(df, ['regionidcounty'], 'parcelid', 'county_property_num')
    df = merge_nunique(df, ['lati', 'long'], 'parcelid', 'county_property_num')

    for col_time in [('transaction_month','month'), ('transaction_month_day','month_day'), ('transaction_day', 'day'), ('yearbuilt','year_built'),
                     ('assessmentyear', 'assessmentyear'), ('buildingqualitytypeid','buildingqualitytypeid'), ('heatingorsystemtypeid', 'heatingorsystemtypeid'),('storytypeid', 'storytypeid'),
                     ('propertylandusetypeid','propertylandusetypeid'), ('pooltypeid10','pooltypeid10'), ('pooltypeid2','pooltypeid2'), ('pooltypeid7','pooltypeid7'),
                     ('architecturalstyletypeid', 'architecturalstyletypeid'), ('buildingclasstypeid','buildingclasstypeid'),
                     ('propertylandusetypeid','propertylandusetypeid'),('propertycountylandusecode', 'propertycountylandusecode') ,('propertyzoningdesc', 'propertyzoningdesc'),
                     ('typeconstructiontypeid', 'typeconstructiontypeid')]:
        df = merge_count(df, [col_time[0],'regionidcity'], 'parcelid', col_time[1]+'_city_transaction_count')
        df = merge_count(df, [col_time[0],'regionidzip'], 'parcelid', col_time[1]+'_region_transaction_count')
        df = merge_count(df, [col_time[0],'regionidcounty'], 'parcelid', col_time[1]+'_county_transaction_count')
        df = merge_count(df, [col_time[0],'loc_label'], 'parcelid', col_time[1]+'_loc_transaction_count')
        df = merge_count(df, [col_time[0],'lati', 'long'], 'parcelid', col_time[1]+'_lati_long_transaction_count')
    # 商圈房屋状况均值
    for col in ['finishedsquarefeet12', 'garagetotalsqft', 'yearbuilt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet',
                'unitcnt', 'poolcnt', 'taxamount', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'buildingqualitytypeid','bathroomcnt','roomcnt',
                'fullbathcnt','calculatedbathnbr']:
        #TODO select features
        df = merge_mean(df, ['loc_label'], col, 'loc_'+col+'_mean')
        df = merge_mean(df, ['regionidzip'], col, 'region_'+col+'_mean')
        df = merge_mean(df, ['regionidcity'], col, 'city_'+col+'_mean')
        df = merge_mean(df, ['regionidcounty'], col, 'county_'+col+'_mean')
        df = merge_mean(df, ['lati', 'long'], col, 'lati_long_'+col+'_mean')


        df = merge_median(df, ['loc_label'], col, 'loc_'+col+'_median')
        df = merge_median(df, ['regionidzip'], col, 'region_'+col+'_median')
        df = merge_median(df, ['regionidcity'], col, 'city_'+col+'_median')
        df = merge_median(df, ['regionidcounty'], col, 'county_'+col+'_median')
        df = merge_median(df, ['lati', 'long'], col, 'lati_long_'+col+'_median')

        df = merge_std(df, ['loc_label'], col, 'loc_'+col+'_std')
        df = merge_std(df, ['regionidzip'], col, 'region_'+col+'_std')
        df = merge_std(df, ['regionidcity'], col, 'city_'+col+'_std')
        df = merge_std(df, ['regionidcounty'], col, 'county_'+col+'_std')
        df = merge_std(df, ['lati', 'long'], col, 'lati_long_'+col+'_std')

    for col in ['finishedsquarefeet12', 'garagetotalsqft', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet',
                'unitcnt', 'poolcnt', 'taxamount', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt']:

        df = merge_sum(df, ['loc_label'], col, 'loc_'+col+'_sum')
        df = merge_sum(df, ['regionidzip'], col, 'region_'+col+'_sum')
        df = merge_sum(df, ['regionidcity'], col, 'city_'+col+'_sum')
        df = merge_sum(df, ['regionidcounty'], col, 'county_'+col+'_sum')
        df = merge_sum(df, ['lati', 'long'], col, 'lati_long_'+col+'_sum')
    # -----------------------------------------------------------------------------------------------

    # life of property
    df['N-life'] = 2018 - df['yearbuilt']

    # error in calculation of the finished living area of home
    df['N-LivingAreaError'] = df['calculatedfinishedsquarefeet'] / df['finishedsquarefeet12']

    # proportion of living area
    df['N-LivingAreaProp'] = df['calculatedfinishedsquarefeet'] / df['lotsizesquarefeet']
    df['N-LivingAreaProp2'] = df['finishedsquarefeet12'] / df['finishedsquarefeet15']

    # Amout of extra space
    df['N-ExtraSpace'] = df['lotsizesquarefeet'] - df['calculatedfinishedsquarefeet']
    df['N-ExtraSpace-2'] = df['finishedsquarefeet15'] - df['finishedsquarefeet12']

    # Total number of rooms
    df['N-TotalRooms'] = df['bathroomcnt'] * df['bedroomcnt']

    # Average room size
    df['N-AvRoomSize'] = df['calculatedfinishedsquarefeet'] / df['roomcnt']

    # Number of Extra rooms
    df['N-ExtraRooms'] = df['roomcnt'] - df['N-TotalRooms']

    # Ratio of the built structure value to land area
    df['N-ValueProp'] = df['structuretaxvaluedollarcnt'] / df['landtaxvaluedollarcnt']

    # Does property have a garage, pool or hot tub and AC?
    df['N-GarPoolAC'] = ((df['garagecarcnt'] > 0) & (df['pooltypeid10'] > 0) & (
    df['airconditioningtypeid'] != 5)) * 1

    df["N-location"] = df["latitude"] + df["longitude"]
    df["N-location-2"] = df["latitude"] * df["longitude"]
    df["N-location-2round"] = df["N-location-2"].round(-4)

    df["N-latitude-round"] = df["latitude"].round(-4)
    df["N-longitude-round"] = df["longitude"].round(-4)


    # ---------------------------------
    # Ratio of tax of property over parcel
    df['N-ValueRatio'] = df['taxvaluedollarcnt'] / df['taxamount']

    # TotalTaxScore
    df['N-TaxScore'] = df['taxvaluedollarcnt'] * df['taxamount']

    # polnomials of tax delinquency year
    df["N-taxdelinquencyyear-2"] = df["taxdelinquencyyear"] ** 2
    df["N-taxdelinquencyyear-3"] = df["taxdelinquencyyear"] ** 3

    # Length of time since unpaid taxes
    df['N-life-tax'] = 2018 - df['taxdelinquencyyear']

    #-------------------------------------------

    # Indicator whether it has AC or not
    df['N-ACInd'] = (df['airconditioningtypeid'] != 5) * 1

    # Indicator whether it has Heating or not
    df['N-HeatInd'] = (df['heatingorsystemtypeid'] != 13) * 1



    #----------------------------------------------

    # polnomials of the variable
    df["N-structuretaxvaluedollarcnt-2"] = df["structuretaxvaluedollarcnt"] ** 2
    df["N-structuretaxvaluedollarcnt-3"] = df["structuretaxvaluedollarcnt"] ** 3

    # Average structuretaxvaluedollarcnt by city
    group = df.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
    df['N-Avg-structuretaxvaluedollarcnt'] = df['regionidcity'].map(group)

    # Deviation away from average
    df['N-Dev-structuretaxvaluedollarcnt'] = abs(
        (df['structuretaxvaluedollarcnt'] - df['N-Avg-structuretaxvaluedollarcnt'])) / df['N-Avg-structuretaxvaluedollarcnt']


    # ----------------------------------------------------


    return df


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

print('Loading data ...')

train = pd.read_csv('../../data/train_2017.csv')
prop = pd.read_csv('../../data/properties_2017.csv').fillna(-0.001)  # , nrows=500)
sample = pd.read_csv('../../data/sample_submission.csv')
'''
print('Binding to float32')
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)
'''

print('Creating training set ...')
train = train.sort_values('transactiondate')
train = train[train.transactiondate < '2017-10-01']
split = train[train.transactiondate < '2017-10-01'].shape[0]
print(split)

train = train[train.logerror > -0.4]
train = train[train.logerror < 0.419]

db = DBSCAN(eps=0.2, min_samples=25).fit(prop[['latitude', 'longitude']])
prop.loc[:, 'loc_label'] = db.labels_
num_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print('Number of clusters: {}'.format(num_clusters))

prop['lati'] = prop['latitude']/10000
prop['long'] = prop['longitude']/10000
prop['lati'] = prop['lati'].apply(np.round)
prop['long'] = prop['long'].apply(np.round)

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
'''
 Step |Time | Value | alpha | colsample_bytree |gamma |max_depth |min_child_weight |subsample |
25 9m50s | - 0.05225 | 9.7207 |0.3694        | 0.0154  | 11.3092  | 19.7752  |  0.9527  |  
7 | 11m46s | -0.05226| 5.8705 |0.2563        | 0.0096 |  14.5790 |  19.7840 |   0.8182 |

18 | 35m01s |-0.05236 |9.1410 |0.5087        | 0.0786 |  14.5949 |  1.7226 |      0.9350 |
26 | 96m17s |   -0.05251 | 0.3411 |0.3419 |    0.2926 |     13.8395 |             1.0616 |      0.8094 |
27 | 18m21s |   -0.05244 |    6.9886 |             0.2620 |    0.0979 |     11.0493 |            19.9869 |      0.5133 |
 29 | 34m36s |   -0.05229 |    9.5245 |             0.7570 |    0.0529 |     14.6558 |            19.5906 |      0.8808 |
'''
params = {'eta': 0.015, 'objective': 'reg:linear', 'eval_metric': 'mae', 'aplpha': 5.8705, 'colsample_bytree': 0.2563, 'gamma':0.0096,'max_depth': 14, 'min_child_weight': 19,'subsample':0.8182, 'silent': 1}

print(params)

watchlist = [(d_train, 'train')]
# cross-validation
# TODO bad news 0.0644361 higher than previous CV set, interesting. 858
# remove cv. back to last point. and continue to test features.
# fold 2 , 0.0643877, overfitting is working. 620+-
print("Running XGBoost CV....")
res = xgb.cv(params, d_train, num_boost_round=2000, nfold=2,
                 early_stopping_rounds=100, verbose_eval=10, show_stdv=True)
num_best_rounds = len(res)
print("Number of best rounds: {}".format(num_best_rounds))
'''

def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma,
                 alpha):

    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)

    cv_result  = xgb.cv(params, d_train, num_boost_round=2000, nfold=2,
                 early_stopping_rounds=100, verbose_eval=10, show_stdv=True)

    return -cv_result['test-mae-mean'].values[-1]

xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (5, 15),
                                                'subsample': (0.5, 1),
                                                'gamma': (0, 10),
                                                'alpha': (0, 10),
                                                })

num_iter = 25
init_points = 5
xgbBO.maximize(init_points=init_points, n_iter=num_iter)

raw_input("Enter something to continue ...")

num_best_rounds = 520

'''
clf = xgb.train(params, d_train, num_best_rounds, watchlist, verbose_eval=10)  # watchlist,  early_stopping_rounds=100, verbose_eval=10)

fig, ax = plt.subplots(figsize=(40,120))
xgb.plot_importance(clf, max_num_features=500, height=0.8, ax=ax)
plt.savefig('../../data/importance.pdf')
# ft_weights = pd.DataFrame(clf.feature_importances_, columns=['weights'], index=train_columns)

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

