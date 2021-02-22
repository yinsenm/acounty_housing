# Author: Yinsen Miao
import pandas as pd
import numpy as np
import os
import random 
import matplotlib.pyplot as plt
import lightgbm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import train_test_split
from itertools import product
from tqdm import tqdm
# import Henderson Moving Average
from util import hmaSymmetricWeights, Henderson
from statsmodels.tsa.seasonal import seasonal_decompose
sns.set_context("poster")

# specify model and image paths
model_path = "../models"
image_path = "../images"

dat = pd.read_pickle("../data/cleandata2.pkl")

# data engineering
origin_date = "1990-01-01"  # let us use this date as the origin
dat["TIME"] = (dat["SALEDATE"] - pd.to_datetime(origin_date)).dt.days / 30
dat["MONTH"] = dat["SALEDATE"].dt.month
dat["LOGPRICE"] = np.log(dat["PRICE"])
dat["LOGFAIRMARKETTOTAL"] = np.log(dat["FAIRMARKETTOTAL"])
dat["LOGLOTAREA"] = np.log(dat["LOTAREA"])
dat["LOGFINISHEDLIVINGAREA"] = np.log(dat["FINISHEDLIVINGAREA"])
dat["PRICEPERSQFT"] = dat["FAIRMARKETTOTAL"] / dat["LOTAREA"]

tiers_cut = dat["FAIRMARKETTOTAL"].quantile([0.05, 0.35, 0.65, 0.95]).tolist()
dat["TIERS"] = pd.cut(dat["FAIRMARKETTOTAL"], tiers_cut, labels=["Bottom", "Middle", "Top"])
# let us drop all data that are considered as outliers
dat.dropna(axis=0, inplace=True)

# compute the median price
median_prc = dat.groupby(["SALEDATE"])["PRICE"].median()
mean_prc = dat.groupby(["SALEDATE"])["PRICE"].mean()

# convert object column to category type
for col in ["NEIGHCODE", "EXTFINISH_DESC", "STYLEDESC", "MUNICODE", "TIERS"]:
    dat[col] = dat[col].astype("category")

# split the data into training and testing dataset
train_dat, valid_dat = train_test_split(dat, train_size=0.8, random_state=2021)
ntrain, nvalid = len(train_dat), len(valid_dat)
print("ntrain = %d, ntest = %d" % (ntrain, nvalid))

# separate features into multiple categories
# continuous features
x_feats = [
    'TIME', 'GRADERANK', 'CDURANK', 'SCHOOLRANK',
    'STORIES', 'BEDROOMS', 'ADJUSTBATHS', 'BSMTGARAGE', 'FIREPLACES', 'YEARBLT', 'BASEMENT',
    'LOGLOTAREA', 'LOGFINISHEDLIVINGAREA', 'PRICEPERSQFT',
    'LATITUDE', 'LONGITUDE', 'ANXIETY', 'OLD', 'POOR', 'VACANT'
]

# nominal features
x_categorical_feats = [
    'NEIGHCODE', 'EXTFINISH_DESC', 'STYLEDESC', 'MONTH', 'TIERS'
]

# target variable log price
y_feats = [
    "LOGPRICE"
]


# create training and testing dataloaders
train_dataloader = lightgbm.Dataset(data=train_dat[x_feats + x_categorical_feats],
                                    label=train_dat[y_feats],
                                    categorical_feature=x_categorical_feats)

valid_dataloader = lightgbm.Dataset(data=valid_dat[x_feats + x_categorical_feats],
                                     label=valid_dat[y_feats],
                                     categorical_feature=x_categorical_feats)

# use the LGBM ML parameter
parameters = {
    'objective': 'mae',
    'metric': ['rmse'],
    'boosting': 'gbdt',
    'num_leaves': 32,
    'min_child_samples': 69,
    'feature_fraction': 0.6597079764069385,
    'bagging_fraction': 0.6171731471091209,
    'bagging_freq': 4,
    'lambda_l1': 3.5479988320298905,
    'lambda_l2': 3.503857243823598,
    'learning_rate': 0.021626597418816215,
    'verbose': 0,
    'seed': 2021
}


# train LGBM model
model = lightgbm.train(params=parameters,
                       train_set=train_dataloader,
                       valid_sets=valid_dataloader,
                       num_boost_round=10000,
                       early_stopping_rounds=1000,
                       verbose_eval=False)

# compute prediction
train_prc_predict = np.exp(model.predict(data=train_dat[x_feats + x_categorical_feats]))  # predict train price
valid_prc_predict = np.exp(model.predict(data=valid_dat[x_feats + x_categorical_feats]))  # predict valid price

# assess the prediction performance
scale = 10000  # show RMSE in 10K
train_rmse = sqrt(mean_squared_error(train_dat["PRICE"] / scale, train_prc_predict / scale))
valid_rmse = sqrt(mean_squared_error(valid_dat["PRICE"] / scale, valid_prc_predict / scale))
train_corr = pearsonr(train_dat["PRICE"], train_prc_predict)[0]
valid_corr = pearsonr(valid_dat["PRICE"], valid_prc_predict)[0]

print("Training RMSE %.3f, Testing RMSE %.3f" % (train_rmse, valid_rmse))
print("Training Corr %.3f, Testing Corr %.3f" % (train_corr, valid_corr))

date_min, date_max = pd.to_datetime("1990-01-31"), dat["SALEDATE"].max()
parids = dat["PARID"].unique().tolist()
print("Now predict housing price for %d homes" % (len(parids)))
dates = pd.date_range(start=date_min, end=date_max, freq='M').tolist()
universe_df = pd.DataFrame(product(parids, dates), columns=["PARID", "SALEDATE"])

# select feats from housing properties
house_df = dat[[
    'PARID', 'GRADERANK', 'CDURANK', 'SCHOOLRANK',
    'STORIES', 'BEDROOMS', 'ADJUSTBATHS', 'BSMTGARAGE', 'FIREPLACES', 'YEARBLT', 'BASEMENT',
    'LOGLOTAREA', 'LOGFINISHEDLIVINGAREA', 'PRICEPERSQFT',
    'LATITUDE', 'LONGITUDE', 'ANXIETY', 'OLD', 'POOR', 'VACANT',
    'NEIGHCODE', 'EXTFINISH_DESC', 'STYLEDESC', 'TIERS'
]].drop_duplicates()

# left merge with selected df
universe_df = pd.merge(universe_df, house_df, on=['PARID'], how='left')

# remove the row that SALEDATE < YEARBLT, let us only consider the house that were already built
universe_df["VALID"] = universe_df["SALEDATE"].dt.year - universe_df["YEARBLT"] > 0
universe_df = universe_df[universe_df["VALID"]]

# create additional features
# time in month with reference to 2012
universe_df["TIME"] = (universe_df["SALEDATE"] - pd.to_datetime(origin_date)).dt.days / 30
universe_df["MONTH"] = universe_df["SALEDATE"].dt.month


# https://www.kaggle.com/jens0306/easy-prediction-using-lightgbm-model
# https://www.zillow.com/research/zhvi-methodology/
# https://markthegraph.blogspot.com/2014/06/henderson-moving-average.html
# predict housing price

# the code below take a long time to run
# test_log_prc = model.predict(data=universe_df[x_feats + x_categorical_feats])
# test_prc = np.exp(test_log_prc)
# universe_df["PREDPRICE"] = test_prc


universe_df = pd.read_pickle("../data/prediction.pkl")

# pivot table of universe_df
universe_pivot = universe_df[["PARID", "SALEDATE", "PREDPRICE"]].pivot_table(index="SALEDATE", values="PREDPRICE", columns="PARID")
universe_mth_ret = universe_pivot.pct_change()
universe_mth_wgt = universe_pivot.div(universe_pivot.sum(axis=1), axis=0).shift(1)
index_mth_ret = (universe_mth_ret * universe_mth_wgt).sum(axis=1)
index_mth = 100 * (index_mth_ret + 1).cumprod()

# del universe_df
# smooth the signals using Henderson filter
smoothed_index_mth = Henderson(index_mth, 5)

# decompose signals into level, trend, seasonality, noise
# https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
result = seasonal_decompose(smoothed_index_mth, model='additive',
                            freq=12, extrapolate_trend='freq')
trend, seasonality, noise = result.trend, result.seasonal, result.resid

# save index
trend.to_csv("../clean_data/acounty_index.csv")

# save decomposed time series
fig = result.plot()
fig.set_size_inches(15, 9)
fig.savefig("%s/timeseries_decompose.png" % image_path)
plt.close()

# read the ZHVI index
ZHVI = pd.read_csv("../clean_data/County_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_mon.csv").\
    query("RegionName == 'Allegheny County'").iloc[0].iloc[9:]
ZHVI = pd.Series(ZHVI.values.astype(float), index=ZHVI.index)
ZHVI.index = pd.to_datetime(ZHVI.index)

scaler = trend["1996-01-31"] / ZHVI["1996-01-31"]
ZHVI = ZHVI * scaler
scaler_median = trend["1996-01-31"] / median_prc["1996-01-31"]
scaler_mean = trend["1996-01-31"] / mean_prc["1996-01-31"]
scaled_median = scaler_median * median_prc
scaled_mean = scaler_mean * mean_prc

# plot my home value index overlaying the index from mean
plt.figure(figsize=(12, 6))
plt.plot(scaled_mean["1990-01-31":], alpha=0.5, label="MeanHVI")
plt.plot(trend, label="YMHVI")
plt.axvline(x=pd.to_datetime("2008-12-31"), color="red", alpha=0.5, linestyle="--", linewidth=3)
plt.axvline(x=pd.to_datetime("2020-02-28"), color="blue", alpha=0.5, linestyle="--", linewidth=3)
plt.ylabel("Home Value Index")
plt.legend(loc="best")
plt.ylim((95, 300))
plt.savefig("%s/index.png" % image_path)
plt.close()

# plot my home value index overlaying the index from mean
plt.figure(figsize=(12, 6))
plt.plot(scaled_mean["1990-01-31":], alpha=0.5, label="MeanHVI")
plt.plot(trend, label="YMHVI")
plt.plot(ZHVI, label="ZHVI")
plt.axvline(x=pd.to_datetime("2008-12-31"), color="red", alpha=0.5, linestyle="--", linewidth=3)
plt.axvline(x=pd.to_datetime("2020-02-28"), color="blue", alpha=0.5, linestyle="--", linewidth=3)
plt.ylabel("Home Value Index")
plt.ylim((95, 300))
plt.legend(loc="best")
plt.savefig("%s/index_com_zillow.png" % image_path)
plt.close()

# universe_df.to_pickle("../data/prediction.pkl")
