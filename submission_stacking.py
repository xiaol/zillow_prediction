# encoding=utf8
import sys
from sklearn.externals import joblib
import xgboost as xgb
import lightgbm as lgb
from preprocessing import *
from keras.models import load_model
import pickle
from datetime import datetime


def predict_on_test(train_columns):
    sub = pd.read_csv('data/sample_submission.csv')
    sample = pd.read_csv('data/sample_submission.csv')
    sample['parcelid'] = sample['ParcelId']
    prop = pd.read_csv('data/properties_2016.csv').fillna(-0.001)  # , nrows=500)

    kmeans = MiniBatchKMeans(n_clusters=320, batch_size=1000).fit(prop[['latitude', 'longitude']])
    prop.loc[:, 'loc_label'] = kmeans.labels_

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

            transactiondate = c[:4] + '-' + c[4:] + '-01'
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

            p_test_cks = stacking(x_test_fold)

            p_test = np.append(p_test, p_test_cks)

            gc.collect()
            del df_test_fold, x_test_fold;
            gc.collect()

        print(c)

        sub[c] = p_test

    print('Writing csv ...')
    file_path = 'data/stacking_' + datetime.now().strftime("%m_%d_%H_%M_%S") + '.csv'
    sub.to_csv(file_path, index=False, float_format='%.4f')


def stacking(test_x):

    clf_list = ['lgb', 'rf', 'gb', 'et', 'xgb', 'nn']

    test_data_list = []
    for clf_name in clf_list:
        test_data = stacking_predict(test_x, clf_name)
        test_data_list.append(test_data)

    test = np.concatenate(test_data_list, axis=1)

    dtest = xgb.DMatrix(test)

    bst = xgb.Booster({'nthread': 4})
    bst.load_model('data/model/xgb_layer_2.model')
    # best_rounds = pd.Series.from_csv('data/model/xgb_layer_2_best_rounds.conf')[0]

    result = bst.predict(dtest)
    return result


def stacking_predict(test_x, clf_name, folds=folds):
    test_pre = np.empty((folds, test_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))

    for i in range(0, folds):
        if clf_name in ["nn"]:
            clf = load_model('data/model/'+clf_name+'_layer_1_fold_'+str(i)+'.h5')
        elif clf_name in ["xgb"]:
            clf = xgb.Booster({'nthread':4})
            clf.load_model('data/model/'+clf_name+'_layer_1_fold_'+str(i)+'.model')
            clf.best_ntree_limit = pd.Series.from_csv('data/model/'+clf_name+'_layer_1_fold_'+str(i)+'_best_ntree_limit.conf')[0]
        elif clf_name in ["lgb"]:
            clf = lgb.Booster(model_file='data/model/'+clf_name+'_layer_1_fold_'+str(i)+'.model')
        else:
            clf = pickle.load(open('data/model/'+clf_name+'_layer_1_fold_'+str(i)+'.pkl', 'rb'))

        if clf_name in ["ada", "lsvc", "knn"]:
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
        elif clf_name in ["lr", "gb", "et", "rf"]:
            test_x = np.log(test_x + 1)
            where_are_nan = np.isnan(test_x)
            where_are_inf = np.isinf(test_x)
            test_x[where_are_nan] = 0
            test_x[where_are_inf] = 0

            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
        elif clf_name in ["nn"]:
            test_x = np.log(test_x + 1)
            where_are_nan = np.isnan(test_x)
            where_are_inf = np.isinf(test_x)
            test_x[where_are_nan] = 0
            test_x[where_are_inf] = 0

            scaler = pickle.load(open('data/model/' + clf_name + '_layer_1_fold_' + str(i) + '_scale.pkl', 'rb'))
            test_x = scaler.transform(test_x)

            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
        elif clf_name in ["xgb"]:
            z = xgb.DMatrix(test_x)
            test_pre[i, :] = clf.predict(z, ntree_limit=clf.best_ntree_limit).reshape(-1, 1)

        elif clf_name in ["lgb"]:
            test_pre[i, :] = clf.predict(test_x, num_iteration=clf.best_iteration).reshape(-1, 1)

        else:
            raise IOError("Please add new clf.")

    test[:] = test_pre.mean(axis=0)
    return test.reshape(-1, 1)

if __name__ == "__main__":
    ps_train_columns = pd.Series.from_csv('data/columns.csv')
    predict_on_test(ps_train_columns)

