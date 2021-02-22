# Author Yinsen Miao
import pandas as pd
import numpy as np
import lightgbm
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
import optuna
sns.set_context("poster")

dat = pd.read_pickle("../data/cleandata2.pkl")

# filter the data via the timestamp
# end_date = "20141231"
end_date = "20201231"
dat.query("SALEDATE <= '%s'" % end_date, inplace=True)

# data engineering
origin_date = "1990-01-01"  # let us use this date as the origin
dat["TIME"] = (dat["SALEDATE"] - pd.to_datetime(origin_date)).dt.days / 30
dat["MONTH"] = dat["SALEDATE"].dt.month
dat["LOGPRICE"] = np.log(dat["PRICE"])
dat["LOGFAIRMARKETTOTAL"] = np.log(dat["FAIRMARKETTOTAL"])
dat["LOGLOTAREA"] = np.log(dat["LOTAREA"])
dat["LOGFINISHEDLIVINGAREA"] = np.log(dat["FINISHEDLIVINGAREA"])
dat["PRICEPERSQFT"] = dat["FAIRMARKETTOTAL"] / dat["LOTAREA"]

# segment the house based on FAIRMARKETTOTAL of 2012
tiers_cut = dat["FAIRMARKETTOTAL"].quantile([0.05, 0.35, 0.65, 0.95]).tolist()
dat["TIERS"] = pd.cut(dat["FAIRMARKETTOTAL"], tiers_cut, labels=["Bottom", "Middle", "Top"])

# let us drop all data that are considered as outliers
dat.dropna(axis=0, inplace=True)

# convert object column to category type
for col in ["NEIGHCODE", "EXTFINISH_DESC", "STYLEDESC", "MUNICODE", "TIERS"]:
    dat[col] = dat[col].astype("category")

# split the data into training and testing based on 80-20 rule
train_dat, valid_dat = train_test_split(dat, train_size=0.8, random_state=2021)
ntrain, nvalid = len(train_dat), len(valid_dat)
print("ntrain = %d, ntest = %d" % (ntrain, nvalid))

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

train_dataloader = lightgbm.Dataset(data=train_dat[x_feats + x_categorical_feats],
                                    label=train_dat[y_feats],
                                    categorical_feature=x_categorical_feats)

valid_dataloader = lightgbm.Dataset(data=valid_dat[x_feats + x_categorical_feats],
                                     label=valid_dat[y_feats],
                                     categorical_feature=x_categorical_feats)


def objective(trial):
    # create LGBM dataset
    # ref https://www.kaggle.com/c/home-credit-default-risk/discussion/58950
    train_dataloader = lightgbm.Dataset(data=train_dat[x_feats + x_categorical_feats],
                                        label=train_dat[y_feats],
                                        categorical_feature=x_categorical_feats)

    valid_dataaloader = lightgbm.Dataset(data=valid_dat[x_feats + x_categorical_feats],
                                         label=valid_dat[y_feats],
                                         categorical_feature=x_categorical_feats)

    # use the parameter
    parameters = {
        'objective': trial.suggest_categorical('objective', ['mae', 'rmse', 'huber', 'quantile', 'mape', 'poisson']),
        'metric': ['rmse'],
        'boosting': 'gbdt',
        'lambda_l1': round(trial.suggest_loguniform('lambda_l1', 1e-8, 10.0), 2),
        'lambda_l2': round(trial.suggest_loguniform('lambda_l2', 1e-8, 10.0), 2),
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

    y_valid = model.predict(data=valid_dat[x_feats + x_categorical_feats])
    # return sqrt(mean_squared_error(valid_data["PRICE"], np.exp(y_valid)))  # compare predicted price vs the truth
    return sqrt(mean_squared_error(valid_dat["LOGPRICE"], y_valid))  # compare predicted price vs the log truth


if __name__ == "__main__":
    sampler = TPESampler(seed=2020)
    study = optuna.create_study(direction="minimize",
                                study_name="fit housing",
                                load_if_exists=True,
                                storage="sqlite:///lgbm_model_index_%s.db" % end_date,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=20))
    study.optimize(objective, n_trials=1000, show_progress_bar=True)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))