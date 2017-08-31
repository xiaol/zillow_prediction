# encoding=utf8
from sklearn.cross_validation import KFold
from scipy import sparse
import xgboost as xgb
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import pickle
from preprocessing import *
from sklearn.feature_selection import SelectFromModel


def stacking(clf, train_x, train_y, test_x, clf_name, scale=None):
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []

    for i, (train_index, test_index) in enumerate(kf):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]

        if clf_name in ["rf", "ada", "gb", "et", "lr", "lsvc", "knn"]:
            clf.fit(tr_x, tr_y)
            pre = clf.predict(te_x).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
            cv_scores.append(mean_absolute_error(te_y, pre))

        elif clf_name in ["xgb"]:
            train_matrix = xgb.DMatrix(tr_x, label=tr_y)
            test_matrix = xgb.DMatrix(te_x, label=te_y)
            z = xgb.DMatrix(test_x)
            params = {'eta': 0.015, 'objective': 'reg:linear', 'eval_metric': 'mae', 'min_child_weight': 1.5,
                      'colsample_bytree': 0.2, 'max_depth': 7, 'lambda': 0.3, 'alpha': 0.6, 'silent': 1}
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                clf = xgb.train(params, train_matrix, num_boost_round=num_round, evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds
                                )
                pre = clf.predict(test_matrix, ntree_limit=clf.best_ntree_limit).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = clf.predict(z, ntree_limit=clf.best_ntree_limit).reshape(-1, 1)
                cv_scores.append(mean_absolute_error(te_y, pre))

        elif clf_name in ["lgb"]:
            train_matrix = lgb.Dataset(tr_x, label=tr_y)
            test_matrix = lgb.Dataset(te_x, label=te_y)
            # z = clf.Dataset(test_x, label=te_y)
            # z=test_x
            params = {'max_bin': 10, 'learning_rate': 0.0021, 'boosting_type': 'gbdt', 'objective': 'regression',
                      'metric': 'l1', 'sub_feature': 0.345, 'bagging_fraction': 0.85, 'bagging_freq': 40,
                      'num_leaves': 512, 'min_data': 500, 'min_hessian': 0.05, 'verbose': 0, 'feature_fraction_seed': 2,
                      'bagging_seed': 3}

            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                clf = lgb.train(params, train_matrix, num_round, valid_sets=test_matrix,
                                early_stopping_rounds=early_stopping_rounds
                                )
                pre = clf.predict(te_x, num_iteration=clf.best_iteration).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = clf.predict(test_x, num_iteration=clf.best_iteration).reshape(-1, 1)
                cv_scores.append(mean_absolute_error(te_y, pre))

        elif clf_name in ["nn"]:
            from keras.layers import Dense, Dropout, BatchNormalization
            from keras.optimizers import SGD, RMSprop
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
            reduce = ReduceLROnPlateau(min_lr=0.0002, factor=0.05)
            clf.compile(optimizer="rmsprop", loss="mae")
            clf.fit(tr_x, tr_y,
                    batch_size=2560,
                    nb_epoch=5000,
                    validation_data=[te_x, te_y],
                    callbacks=[early_stopping, reduce])
            pre = clf.predict(te_x).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
            cv_scores.append(mean_absolute_error(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print "%s now score is:" % clf_name, cv_scores
        with open("data/score.txt", "a") as f:
            f.write("%s now score is:" % clf_name + str(cv_scores) + "\n")

        if clf_name in ["nn"]:
            clf.save('data/model/' + clf_name + '_layer_1_fold_' + str(i) + '.h5')
            pickle.dump(scale, open('data/model/' + clf_name + '_layer_1_fold_' + str(i) + '_scale.pkl', 'wb'))
        elif clf_name in ["xgb"]:
            clf.save_model('data/model/' + clf_name + '_layer_1_fold_' + str(i) + '.model')
            pd.Series.to_csv(pd.Series(clf.best_ntree_limit), 'data/model/' + clf_name + '_layer_1_fold_' + str(i) + '_best_ntree_limit.conf')
        elif clf_name in ["lgb"]:
            clf.save_model('data/model/' + clf_name + '_layer_1_fold_' + str(i) + '.model', num_iteration=clf.best_iteration )
        else:
            pickle.dump(clf, open('data/model/' + clf_name + '_layer_1_fold_' + str(i) + '.pkl', 'wb'))
    test[:] = test_pre.mean(axis=0)
    print "%s_score_list:" % clf_name, cv_scores
    print "%s_score_mean:" % clf_name, np.mean(cv_scores)
    with open("data/score.txt", "a") as f:
        f.write("%s_score_mean:" % clf_name + str(np.mean(cv_scores)) + "\n")
    return train.reshape(-1, 1), test.reshape(-1, 1)


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

    randomforest = RandomForestRegressor(n_estimators=600, max_depth=20, n_jobs=-1, random_state=2017,
                                         max_features="auto", verbose=1)
    rf_train, rf_test = stacking(randomforest, x_train, y_train, x_valid, "rf")
    return rf_train, rf_test, "rf_reg"


def ada_reg(x_train, y_train, x_valid):
    adaboost = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking(adaboost, x_train, y_train, x_valid, "ada")
    return ada_train, ada_test, "ada_reg"


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

    gbdt = GradientBoostingRegressor(learning_rate=0.06, n_estimators=100, subsample=0.8, random_state=2017,
                                     max_depth=5, verbose=1)
    gbdt_train, gbdt_test = stacking(gbdt, x_train, y_train, x_valid, "gb")
    return gbdt_train, gbdt_test, "gb_reg"


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

    extratree = ExtraTreesRegressor(n_estimators=600, max_depth=22, max_features="auto", n_jobs=-1, random_state=2017,
                                    verbose=1)
    et_train, et_test = stacking(extratree, x_train, y_train, x_valid, "et")
    return et_train, et_test, "et_reg"


def lr_reg(x_train, y_train, x_valid):
    x_train = np.log(x_train + 1)  # TODO remove log constraint for all regressors.
    x_valid = np.log(x_valid + 1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = 0
    x_valid[where_are_inf] = 0

    lr_reg = LinearRegression(n_jobs=-1)
    lr_train, lr_test = stacking(lr_reg, x_train, y_train, x_valid, "lr")
    return lr_train, lr_test, "lr_reg"


def xgb_reg(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking(xgb, x_train, y_train, x_valid, "xgb")
    return xgb_train, xgb_test, "xgb_reg"


def lgb_reg(x_train, y_train, x_valid):
    lgb_train, lgb_test = stacking(lgb, x_train, y_train, x_valid, "lgb")
    return lgb_train, lgb_test, "lgb_reg"


def nn_reg(x_train, y_train, x_valid):
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

    scale = StandardScaler()
    scale.fit(x_train)
    x_train = scale.transform(x_train)
    x_valid = scale.transform(x_valid)

    nn_train, nn_test = stacking("", x_train, y_train, x_valid, "nn", scale)
    return nn_train, nn_test, "nn_reg"


def select_feature(clf, x_train, x_valid):
    clf.fit(x_train, y_train)
    model = SelectFromModel(clf, prefit=True, threshold="mean")

    print x_train.shape
    x_train = model.transform(x_train)
    x_valid = model.transform(x_valid)
    print x_train.shape

    return x_train, x_valid


if __name__ == "__main__":
    np.random.seed(1)
    x_train, y_train, x_valid, y_valid = preprocess_training_data()

    # clf=GradientBoostingRegressor()
    # x_train,x_valid=select_feature(clf,x_train,x_valid)
    x_train = np.array(x_train)
    x_valid = np.array(x_valid)

    seed = 1
    kf = KFold(x_train.shape[0], n_folds=folds, shuffle=True, random_state=seed)

    clf_list = [lgb_reg, rf_reg, gb_reg, et_reg, xgb_reg, nn_reg]

    column_list = []
    train_data_list = []
    test_data_list = []
    for clf_one in clf_list:
        train_data, test_data, clf_name = clf_one(x_train, y_train, x_valid)
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        column_list.append("select_%s" % clf_name)

    train = np.concatenate(train_data_list, axis=1)
    test = np.concatenate(test_data_list, axis=1)

    dtrain = xgb.DMatrix(train, label=y_train)
    dtest = xgb.DMatrix(test)

    xgb_params = {'seed': 0, 'colsample_bytree': 0.8, 'silent': 1, 'subsample': 0.6, 'learning_rate': 0.01,
                  'objective': 'reg:linear', 'max_depth': 4, 'num_parallel_tree': 1, 'min_child_weight': 1,
                  'eval_metric': 'mae'}
    res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=4, seed=0, stratified=False, feval=xg_eval_mae,
                 early_stopping_rounds=25, verbose_eval=10, show_stdv=True, maximize=False)

    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 0]
    cv_std = res.iloc[-1, 1]

    print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
    with open("data/score.txt", "a") as f:
        f.write('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std) + "\n")

    gdbt = xgb.train(xgb_params, dtrain, best_nrounds)
    gdbt.save_model('data/model/xgb_layer_2.model')
    # pd.Series.to_csv(pd.Series(best_nrounds), 'data/model/xgb_layer_2_best_rounds.conf')
    result = gdbt.predict(dtest)

    f_test = MAE(y_valid, result)
    print 'Test mae:', f_test
    with open("data/score.txt", "a") as f:
        f.write('Test mae:' + str(f_test) + "\n")
