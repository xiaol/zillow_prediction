# encoding=utf8
from util import *
from sklearn.cluster import MiniBatchKMeans
import gc

drop_cols = ['parcelid', 'logerror']
one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',
                       'heatingorsystemtypeid', 'storytypeid', 'regionidcity', 'regionidcounty', 'regionidneighborhood',
                       'regionidzip', 'hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag', 'propertylandusetypeid',
                       'propertycountylandusecode', 'propertyzoningdesc', 'typeconstructiontypeid', 'fips']

folds =2

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
    df['room_sqt'] = df['calculatedfinishedsquarefeet'] / df['roomcnt']
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


def preprocess_training_data():
    print('Loading data ...')

    train = pd.read_csv('data/train_2016_v2.csv')
    prop = pd.read_csv('data/properties_2016.csv').fillna(-0.001)  # , nrows=500)
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

    del df_train;
    gc.collect()

    x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
    return x_train, y_train, x_valid, y_valid