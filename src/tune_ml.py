import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import lightgbm
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from math import sqrt

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

# let us keep all data lower than 1M to avoid outlier
dat = dat.query("PRICE < %d" % 1e6)

# convert object column to category type
for col in ["NEIGHCODE", "EXTFINISH_DESC", "STYLEDESC", "MUNICODE"]:
    dat[col] = dat[col].astype("category")
print(dat.dtypes)

# split data into training and testing base on year
# specify the range of date for validation
valid_bgn, valid_end = "2016-01-01", "2021-02-28"
train_dat = dat.query("SALEDATE < '%s'" % valid_bgn)
# testing after 2016-01-01
valid_data = dat.query("SALEDATE > '%s' and SALEDATE < '%s'" % (valid_bgn, valid_end))

ntrain, nvalid = len(train_dat), len(valid_data)
print("ntrain = %d, ntest = %d" % (ntrain, nvalid))

# continuous features
x_feats = [
    'TIME', 'GRADERANK', 'CDURANK', 'SCHOOLRANK',
    'STORIES', 'BEDROOMS', 'ADJUSTBATHS', 'BSMTGARAGE', 'FIREPLACES', 'YEARBLT', 'BASEMENT',
    'LOGFAIRMARKETTOTAL', 'LOGLOTAREA', 'LOGFINISHEDLIVINGAREA', 'PRICEPERSQFT'
]

# nominal features
x_categorical_feats = [
    'NEIGHCODE', 'EXTFINISH_DESC', 'STYLEDESC',
]

# target variable log price
y_feats = [
    "LOGPRICE"
]


def objective(trial):
    # create LGBM dataset
    # ref https://www.kaggle.com/c/home-credit-default-risk/discussion/58950
    train_dataloader = lightgbm.Dataset(data=train_dat[x_feats + x_categorical_feats],
                                        label=train_dat[y_feats],
                                        categorical_feature=x_categorical_feats)

    valid_dataaloader = lightgbm.Dataset(data=valid_data[x_feats + x_categorical_feats],
                                         label=valid_data[y_feats],
                                         categorical_feature=x_categorical_feats)

    # use the parameter
    parameters = {
        'objective': trial.suggest_categorical('objective', ['mae', 'rmse', 'huber', 'quantile', 'mape', 'poisson']),
        'metric': ['rmse'],
        'boosting': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
        'verbose': -1,
        'seed': 2021
    }

    model = lightgbm.train(params=parameters,
                           train_set=train_dataloader,
                           valid_sets=valid_dataaloader,
                           num_boost_round=10000,
                           early_stopping_rounds=1000,
                           verbose_eval=False)

    y_valid = model.predict(data=valid_data[x_feats + x_categorical_feats])
    return sqrt(mean_squared_error(valid_data["PRICE"], np.exp(y_valid))) # compare predicte price vs the truth


if __name__ == "__main__":
    sampler = TPESampler(seed=2020)
    study = optuna.create_study(direction="minimize",
                                study_name="fit housing",
                                load_if_exists=True,
                                storage="sqlite:///lgbm_model2.db",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=20))
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))