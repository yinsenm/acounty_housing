import pandas as pd
import numpy as np
import os
from util import hmaSymmetricWeights, Henderson
from statsmodels.tsa.seasonal import seasonal_decompose

# read HVI prediction from data folder.
dat = pd.read_pickle("../data/prediction.pkl")

# get top NEIGHCODE time series
dat_neighcode = dat[['PARID', 'NEIGHCODE']].drop_duplicates()['NEIGHCODE'].value_counts()

topn = 200
top_neighcode = dat_neighcode.head(topn)
neighcodes = [code for code in top_neighcode.index]

indexes = []
for neighcode in neighcodes:
    prc_pivot = dat.query("NEIGHCODE == '%s'" % neighcode)[["PARID", "SALEDATE", "PREDPRICE"]].\
        pivot_table(index="SALEDATE", values="PREDPRICE", columns="PARID")
    ret_pivot = prc_pivot.pct_change()
    wgt_pivot = prc_pivot.div(prc_pivot.sum(axis=1), axis=0).shift(1)
    index_ret = (ret_pivot * wgt_pivot).sum(axis=1)
    index = 100 * (1.0 + index_ret).cumprod()
    # smooth the signals using Henderson filter
    smoothed_index = Henderson(index, 5)
    result = seasonal_decompose(smoothed_index, model='additive',
                                freq=12, extrapolate_trend='freq')
    trend, seasonality, noise = result.trend, result.seasonal, result.resid
    indexes.append(trend)

neigh_index = pd.concat(indexes, axis=1)
neigh_index.columns = neighcodes
neigh_index.to_csv("../data/top%d_neigh_index.csv" % topn, float_format='%8.3f')


# get top MUNICODE time series
map_parid_municode = pd.read_pickle("../data/cleandata2.pkl")[["PARID", "MUNICODE"]].drop_duplicates()
dat = dat.merge(map_parid_municode, on="PARID")
dat_municode = dat[['PARID', 'MUNICODE']].drop_duplicates()['MUNICODE'].value_counts()
municodes = [code for code in dat_municode.index]

indexes = []
for municode in municodes:
    prc_pivot = dat.query("MUNICODE == '%s'" % municode)[["PARID", "SALEDATE", "PREDPRICE"]].\
        pivot_table(index="SALEDATE", values="PREDPRICE", columns="PARID")
    ret_pivot = prc_pivot.pct_change()
    wgt_pivot = prc_pivot.div(prc_pivot.sum(axis=1), axis=0).shift(1)
    index_ret = (ret_pivot * wgt_pivot).sum(axis=1)
    index = 100 * (1.0 + index_ret).cumprod()
    # smooth the signals using Henderson filter
    smoothed_index = Henderson(index, 5)
    result = seasonal_decompose(smoothed_index, model='additive',
                                freq=12, extrapolate_trend='freq')
    trend, seasonality, noise = result.trend, result.seasonal, result.resid
    indexes.append(trend)

muni_index = pd.concat(indexes, axis=1)
muni_index.columns = municodes
muni_index.to_csv("../data/muni_index.csv", float_format='%8.3f')