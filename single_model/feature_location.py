import pandas as pd
import math
from math import radians, cos, sin, asin, sqrt


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
    train = pd.read_csv('../data/train_2016_v2.csv')
    df_train = train.merge(prop, how='left', on='parcelid')

    df_geo = prop[['latitude', 'longitude', 'parcelid']]
    df_geo.dropna(subset=['latitude', 'longitude'], axis=0, inplace=True)

    df_geo['latitude'] /= 1e6
    df_geo['longitude'] /= 1e6

    building_la = list(df_geo.latitude)
    building_lo = list(df_geo.longitude)
    buildings = zip(building_la, building_lo)

    df_train['loc_building_num'] = map(lambda la, lo: building_num(la, lo, 1000, buildings), df_train['latitdue'], df_train['logitude'])

    df_train[['parcelid', 'loc_building_num']].to_csv('../data/location_2016.csv', index=None)


prepare_data()