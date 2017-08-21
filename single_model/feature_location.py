import pandas as pd
import math
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import MiniBatchKMeans


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # mile
    return c * r * 1000  # meter


def building_num(la, lo, n, buildings):
    num = 0
    for s in buildings:
        slo = float(s[1])
        sla = float(s[0])
        dis = haversine(slo, sla, lo, la)
        if dis <= n:
            num += 1
    return num


def prepare_data():
    prop = pd.read_csv('../data/properties_2016.csv')
    train = pd.read_csv('../data/train_2016_v2.csv').fillna(-1)
    df_train = train.merge(prop, how='left', on='parcelid')

    df_geo = prop[['latitude', 'longitude', 'parcelid']]
    df_geo.dropna(subset=['latitude', 'longitude'], axis=0, inplace=True)

    df_geo['latitude'] /= 1e6
    df_geo['longitude'] /= 1e6

    kmeans = MiniBatchKMeans(n_clusters=300, batch_size=1000).fit(df_geo[['latitude', 'longitude']])
    df_geo.loc[:, 'loc_label'] = kmeans.labels_

    building_la = list(df_geo.latitude)
    building_lo = list(df_geo.longitude)
    buildings = zip(building_la, building_lo)
    sale_buildings = zip(list(df_train.latitude), list(df_train.longitude))

    ''' 
    df_train['loc_building_num'] = map(lambda la, lo: building_num(la/1e6, lo/1e6, 1000, buildings), df_train['latitude'], df_train['longitude'])
    df_train['loc_sale_num'] = map(lambda la, lo: building_num(la/1e6, lo/1e6, 1000, sale_buildings), df_train['latitude'], df_train['longitude'])
    df_train['loc_sale_rt'] = df_train['loc_sale_num'] / df_train['loc_building_num']
    '''

    df_geo[['parcelid', 'loc_label']].to_csv('../data/location_2016.csv', index=None)


prepare_data()