
#encoding=utf8
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
from scipy import sparse
import xgboost
import lightgbm

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss,mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
import gc

from sklearn.cluster import MiniBatchKMeans
from util import *

def stacking_reg(clf,train_x,train_y,test_x,clf_name):
    train=np.zeros((train_x.shape[0],1))
    test=np.zeros((test_x.shape[0],1))
    test_pre=np.empty((folds,test_x.shape[0],1))
    cv_scores=[]

    for i,(train_index,test_index) in enumerate(kf):
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        te_x=train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf","ada","gb","et","lr","lsvc","knn"]:
            clf.fit(tr_x,tr_y)
            pre=clf.predict(te_x).reshape(-1,1)
            train[test_index]=pre
            test_pre[i,:]=clf.predict(test_x).reshape(-1,1)
            cv_scores.append(mean_absolute_error(te_y, pre))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            params = {'eta': 0.015, 'objective': 'reg:linear', 'eval_metric': 'mae', 'min_child_weight': 1.5, 'colsample_bytree': 0.2, 'max_depth': 7, 'lambda': 0.3, 'alpha': 0.6, 'silent': 1}
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round,evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(test_matrix,ntree_limit=model.best_ntree_limit).reshape(-1,1)
                train[test_index]=pre
                test_pre[i, :]= model.predict(z, ntree_limit=model.best_ntree_limit).reshape(-1,1)
                cv_scores.append(mean_absolute_error(te_y, pre))

        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            #z = clf.Dataset(test_x, label=te_y)
            #z=test_x
            params = {'max_bin': 10, 'learning_rate': 0.0021, 'boosting_type': 'gbdt', 'objective': 'regression','metric': 'l1', 'sub_feature': 0.345, 'bagging_fraction': 0.85, 'bagging_freq': 40,'num_leaves': 512, 'min_data': 500, 'min_hessian': 0.05, 'verbose': 0, 'feature_fraction_seed': 2,'bagging_seed': 3}

            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(te_x,num_iteration=model.best_iteration).reshape(-1,1)
                train[test_index]=pre
                test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration).reshape(-1,1)
                cv_scores.append(mean_absolute_error(te_y, pre))

        elif clf_name in ["nn"]:
            from keras.layers import Dense, Dropout, BatchNormalization
            from keras.optimizers import SGD,RMSprop
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from keras.utils import np_utils
            from keras.regularizers import l2
            from keras.models import Sequential
            clf = Sequential()
            clf.add(Dense(64, input_dim=tr_x.shape[1], activation="relu", W_regularizer=l2()))
            # model.add(Dropout(0.2))
            clf.add(Dense(64, activation="relu", W_regularizer=l2()))
            # model.add(Dropout(0.2))
            clf.add(Dense(1))
            clf.summary()
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            reduce = ReduceLROnPlateau(min_lr=0.0002,factor=0.05)
            clf.compile(optimizer="rmsprop", loss="mae")
            clf.fit(tr_x, tr_y,
                      batch_size=2560,
                      nb_epoch=5000,
                      validation_data=[te_x, te_y],
                      callbacks=[early_stopping, reduce])
            pre=clf.predict(te_x).reshape(-1,1)
            train[test_index]=pre
            test_pre[i,:]=clf.predict(test_x).reshape(-1,1)
            cv_scores.append(mean_absolute_error(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print "%s now score is:"%clf_name,cv_scores
        with open("score.txt","a") as f:
            f.write("%s now score is:"%clf_name+str(cv_scores)+"\n")
    test[:]=test_pre.mean(axis=0)
    print "%s_score_list:"%clf_name,cv_scores
    print "%s_score_mean:"%clf_name,np.mean(cv_scores)
    with open("score.txt", "a") as f:
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1,1),test.reshape(-1,1)

def rf_reg(x_train, y_train, x_valid):
    x_train = np.log(x_train + 1)
    x_valid = np.log(x_valid + 1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = 0
    x_valid[where_are_inf] = 0

    randomforest = RandomForestRegressor(n_estimators=600, max_depth=20, n_jobs=-1, random_state=2017, max_features="auto",verbose=1)
    rf_train, rf_test = stacking_reg(randomforest, x_train, y_train, x_valid,"rf")
    return rf_train, rf_test,"rf_reg"

def ada_reg(x_train, y_train, x_valid):
    adaboost = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking_reg(adaboost, x_train, y_train, x_valid,"ada")
    return ada_train, ada_test,"ada_reg"

def gb_reg(x_train, y_train, x_valid):
    x_train = np.log(x_train + 1)
    x_valid = np.log(x_valid + 1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = 0
    x_valid[where_are_inf] = 0

    gbdt = GradientBoostingRegressor(learning_rate=0.06, n_estimators=100, subsample=0.8, random_state=2017,max_depth=5,verbose=1)
    gbdt_train, gbdt_test = stacking_reg(gbdt, x_train, y_train, x_valid,"gb")
    return gbdt_train, gbdt_test,"gb_reg"

def et_reg(x_train, y_train, x_valid):
    x_train = np.log(x_train + 1)
    x_valid = np.log(x_valid + 1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = 0
    x_valid[where_are_inf] = 0

    extratree = ExtraTreesRegressor(n_estimators=600, max_depth=22, max_features="auto", n_jobs=-1, random_state=2017,verbose=1)
    et_train, et_test = stacking_reg(extratree, x_train, y_train, x_valid,"et")
    return et_train, et_test,"et_reg"

def lr_reg(x_train, y_train, x_valid):
    x_train=np.log(x_train+1)
    x_valid=np.log(x_valid+1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = 0
    x_valid[where_are_inf] = 0

    lr_reg=LinearRegression(n_jobs=-1)
    lr_train, lr_test = stacking_reg(lr_reg, x_train, y_train, x_valid, "lr")
    return lr_train, lr_test, "lr_reg"

def xgb_reg(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking_reg(xgboost, x_train, y_train, x_valid,"xgb")
    return xgb_train, xgb_test,"xgb_reg"

def lgb_reg(x_train, y_train, x_valid):
    lgb_train, lgb_test = stacking_reg(lightgbm, x_train, y_train, x_valid,"lgb")
    return lgb_train, lgb_test,"lgb_reg"

def nn_reg(x_train, y_train, x_valid):
    x_train=np.log(x_train+1)
    x_valid=np.log(x_valid+1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = 0
    x_valid[where_are_inf] = 0

    scale=StandardScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)

    nn_train, nn_test = stacking_reg("", x_train, y_train, x_valid, "nn")
    return nn_train, nn_test, "nn_reg"


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
    df['room_sqt'] = df['calculatedfinishedsquarefeet']/df['roomcnt']
    # df['structure_tax_rt'] = df['structuretaxvaluedollarcnt'] / df['taxvaluedollarcnt']
    '''
    df['land_tax_rt'] = df['landtaxvaluedollarcnt'] / df['taxvaluedollarcnt']
    '''

    # 商圈内待售房屋数量
    # df = merge_nunique(df, ['loc_label'], 'parcelid', 'loc_building_num')
    df = merge_nunique(df, ['regionidzip'], 'parcelid', 'region_property_num')
    df = merge_nunique(df, ['regionidcity'], 'parcelid', 'city_property_num')
    # df = merge_nunique(df, ['regionidcounty'], 'parcelid', 'county_property_num')

    # df = merge_count(df, ['transaction_month','regionidcity'], 'parcelid', 'city_month_transaction_count')
    # 商圈房屋状况均值
    # df = merge_median(df, ['regionidcity'], 'buildingqualitytypeid', 'city_quality_median')
    '''
    for col in ['finishedsquarefeet12', 'garagetotalsqft', 'yearbuilt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet']:
        df = merge_median(df, ['loc_label'], col, 'loc_'+col+'_median')
    '''

    return df


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def preprocess():

    print('Loading data ...')

    train = pd.read_csv('data/train_2016_v2.csv')
    prop = pd.read_csv('data/properties_2016.csv').fillna(-0.001)  # , nrows=500)
    sample = pd.read_csv('data/sample_submission.csv')
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


    kmeans = MiniBatchKMeans(n_clusters=320, batch_size=1000).fit(prop[['latitude', 'longitude']])
    prop.loc[:, 'loc_label'] = kmeans.labels_

    df_train = train.merge(prop, how='left', on='parcelid').fillna(-0.001)

    x_train = df_train
    x_train = get_features(x_train)
    x_train = prepare_data(x_train, one_hot_encode_cols)
    x_train = x_train.drop(drop_cols, axis=1)

    train_columns = x_train.columns

    y_train = df_train['logerror'].values
    print(x_train.shape, y_train.shape)
    print x_train.columns
    pd.Series(list(x_train.columns)).to_csv('data/columns.csv')

    del df_train; gc.collect()

    x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
    return x_train,y_train,x_valid, y_valid

from sklearn.feature_selection import SelectFromModel
def select_feature(clf,x_train,x_valid):
    clf.fit(x_train, y_train)
    model = SelectFromModel(clf, prefit=True, threshold="mean")

    print x_train.shape
    x_train = model.transform(x_train)
    x_valid = model.transform(x_valid)
    print x_train.shape

    return x_train,x_valid

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    # return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))
    return 'mae', np.sum([abs(yhat[i]-y[i]) for i in range(len(yhat))])/ len(yhat)

def MAE(yhat, y):
    return np.sum([abs(yhat[i]-y[i]) for i in range(len(yhat))])/ len(yhat)


if __name__=="__main__":
    np.random.seed(1)
    x_train, y_train, x_valid, y_valid= preprocess()

    # clf=GradientBoostingRegressor()
    # x_train,x_valid=select_feature(clf,x_train,x_valid)
    x_train = np.array(x_train)
    x_valid = np.array(x_valid)

    folds = 5
    seed = 1
    kf = KFold(x_train.shape[0], n_folds=folds, shuffle=True, random_state=seed)

    clf_list = [lgb_reg,rf_reg,gb_reg,et_reg,xgb_reg,nn_reg]

    column_list = []
    train_data_list=[]
    test_data_list=[]
    for clf in clf_list:
        train_data,test_data,clf_name=clf(x_train,y_train,x_valid)
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        column_list.append("select_%s" % clf_name)

    train = np.concatenate(train_data_list, axis=1)
    test = np.concatenate(test_data_list, axis=1)

    dtrain = xgboost.DMatrix(train, label=y_train)
    dtest = xgboost.DMatrix(test)

    xgb_params = {'seed':0, 'colsample_bytree':0.8, 'silent':1, 'subsample':0.6, 'learning_rate': 0.01, 'objective': 'reg:linear', 'max_depth': 4, 'num_parallel_tree':1, 'min_child_weight':1, 'eval_metric':'mae'}
    res = xgboost.cv(xgb_params, dtrain, num_boost_round=1000, nfold=4, seed=0, stratified=False, feval=xg_eval_mae,
                     early_stopping_rounds=25, verbose_eval=10, show_stdv=True,  maximize=False)

    best_nrounds = res.shape[0] -1
    cv_mean = res.iloc[-1, 0]
    cv_std = res.iloc[-1, 1]

    print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

    gdbt = xgboost.train(xgb_params, dtrain, best_nrounds)

    result = gdbt.predict(dtest)

    f_test = MAE(y_valid, result)
    print 'Test mae:', f_test