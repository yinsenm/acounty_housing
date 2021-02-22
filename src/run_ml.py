# Author Yinsen Miao
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import lightgbm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
sns.set_context("poster")


model_path = "../models"
image_path = "../images"
model_id = 1

dat = pd.read_pickle("../data/cleandata.pkl")
# print available features
print(dat.columns)

# data engineering
origin_date = "2010-01-01"  # let us use this date as the origin
dat["TIME"] = (dat["SALEDATE"] - pd.to_datetime(origin_date)).dt.days / 30
dat["LOGPRICE"] = np.log(dat["PRICE"])
dat["LOGFAIRMARKETTOTAL"] = np.log(dat["FAIRMARKETTOTAL"])
dat["LOGLOTAREA"] = np.log(dat["LOTAREA"])
dat["LOGFINISHEDLIVINGAREA"] = np.log(dat["FINISHEDLIVINGAREA"])
dat["PRICEPERSQFT"] = dat["FAIRMARKETTOTAL"] / dat["LOTAREA"]

# convert object column to category type
for col in ["NEIGHCODE", "EXTFINISH_DESC", "STYLEDESC", "MUNICODE"]:
    dat[col] = dat[col].astype("category")
print(dat.dtypes)

# split data into training and testing base on year
# training before 2016-01-01
split_date = "2016-01-01"
train_dat = dat.query("SALEDATE < '%s'" % split_date)
# testing after 2016-01-01
test_dat = dat.query("SALEDATE > '%s'" % split_date)

ntrain, ntest = len(train_dat), len(test_dat)
print("ntrain = %d, ntest = %d" % (ntrain, ntest))

# continuous features
x_feats = [
    'TIME', 'GRADERANK', 'CDURANK', 'SCHOOLRANK',
    'STORIES', 'BEDROOMS', 'ADJUSTBATHS', 'BSMTGARAGE', 'FIREPLACES', 'YEARBLT', 'BASEMENT',
    'LOGFAIRMARKETTOTAL', 'LOGLOTAREA', 'LOGFINISHEDLIVINGAREA'
]

# nominal features
x_categorical_feats = [
    'NEIGHCODE', 'EXTFINISH_DESC', 'STYLEDESC',
]

# target variable log price
y_feats = [
    "LOGPRICE"
]

# create LGBM dataset
# ref https://www.kaggle.com/c/home-credit-default-risk/discussion/58950
train_dataloader = lightgbm.Dataset(data=train_dat[x_feats + x_categorical_feats],
                                    label=train_dat[y_feats],
                                    categorical_feature=x_categorical_feats)

test_dataloader = lightgbm.Dataset(data=test_dat[x_feats + x_categorical_feats],
                                    label=test_dat[y_feats],
                                    categorical_feature=x_categorical_feats)

# use the parameter
parameters = {
    'objective': 'mae',
    'metric': ['rmse'],
    'boosting': 'gbdt',
    'num_leaves': 10,
    'min_child_samples': 56,
    'feature_fraction': 0.795,
    'bagging_fraction': 0.767,
    'bagging_freq': 1,
    'lambda_l1': 0.00243,
    'lambda_l2': 4.753,
    'learning_rate': 0.03,
    'verbose': 0,
    'seed': 2021
}


model = lightgbm.train(params=parameters,
                       train_set=train_dataloader,
                       valid_sets=test_dataloader,
                       num_boost_round=10000,
                       early_stopping_rounds=1000)

# compute prediction
train_prc_predict = np.exp(model.predict(data=train_dat[x_feats + x_categorical_feats]))  #  predict price
test_prc_predict = np.exp(model.predict(data=test_dat[x_feats + x_categorical_feats]))

# assess the prediction performance
train_rmse = sqrt(mean_squared_error(train_dat["PRICE"], train_prc_predict))
test_rmse = sqrt(mean_squared_error(test_dat["PRICE"], test_prc_predict))
train_corr = pearsonr(train_dat["PRICE"], train_prc_predict)[0]
test_corr = pearsonr(test_dat["PRICE"], test_prc_predict)[0]

print("Training RMSE %.3f, Testing RMSE %.3f" % (train_rmse, test_rmse))
print("Training Corr %.3f, Testing Corr %.3f" % (train_corr, test_corr))

# plot training
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(train_dat["PRICE"], train_prc_predict, s=10, zorder=10)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Truth ($)")
ax.set_ylabel("Prediction ($)")
f = lambda x, pos: f'{x/10**6:,.1f}M'
ax.yaxis.set_major_formatter(FuncFormatter(f))
ax.xaxis.set_major_formatter(FuncFormatter(f))
ax.figure.savefig("%s/train_model%d.png" % (image_path, model_id))
plt.close()


# plot testing
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(test_dat["PRICE"], test_prc_predict, s=10, zorder=10)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Truth ($)")
ax.set_ylabel("Prediction ($)")
f = lambda x, pos: f'{x/10**6:,.1f}M'
ax.yaxis.set_major_formatter(FuncFormatter(f))
ax.xaxis.set_major_formatter(FuncFormatter(f))
ax.figure.savefig("%s/test_model%d.png" % (image_path, model_id))
plt.close()