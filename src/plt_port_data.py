# Author Yinsen Miao
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

dat = pd.read_pickle("../data/cleandata2.pkl")
dat_municode = dat[['PARID', 'MUNICODE']].drop_duplicates()['MUNICODE'].value_counts()
municodes = [str(code) for code in dat_municode.index]
dat_municode.index = municodes

# get housing HVI index
hvi_dat = pd.read_csv("../data/muni_index.csv",
                      parse_dates=["SALEDATE"],
                      index_col=["SALEDATE"])

# compute the past 10 year
hvi_yret = hvi_dat["2005-12-31": "2015-12-31"].resample("Y").last().pct_change().dropna()
n, _ = hvi_yret.shape
avg_rets = round((((hvi_yret + 1).prod()) ** (1.0 / n) - 1.0) * 100., 2)
avg_stds = round((hvi_yret).std() * 100, 2)

dat_summary = pd.concat([dat_municode[municodes], avg_rets[municodes], avg_stds[municodes]], axis=1).reset_index()
dat_summary.columns = ["MUNICODE", "Count", "Return", "Risk"]

dat_summary = dat_summary.sort_values("Return", ascending=False)
# write municode, return risks


topn = 20
top_municodes = dat_summary.head(topn)["MUNICODE"].to_list()
dat_summary.to_csv("../clean_data/counts_rets_rsks.csv")

# hvi_dat.to_csv("../data/top_hvi.csv", index_label="ZIP")
nrow = 5
ncol = int(topn / nrow)
fig, axs = plt.subplots(ncol, nrow, figsize=(25, 12))
for idx, municode in enumerate(top_municodes):
    i, j = idx // nrow, idx % nrow
    axs[i, j].plot(hvi_dat.index, hvi_dat[municode])
    if i != ncol - 1:
        axs[i, j].get_xaxis().set_ticks([])
    else:
        axs[i, j].tick_params(axis='x', labelrotation=45)
    if j != 0:
        axs[i, j].get_yaxis().set_ticks([])
    axs[i, j].set_title("%s" % municode)
    axs[i, j].set_ylim(80, 450)
    axs[i, j].axvline(x=pd.to_datetime("2015-12-31"), color="red", alpha=0.5, linestyle="--", linewidth=3)
    # axs[i, j].axvline(x=pd.to_datetime("2020-01-31"), color="blue", alpha=0.3, linestyle="--", linewidth=3)
plt.savefig("../images/municode.png")
plt.close()

# compute statistics
hvi_dat = pd.read_csv("../data/muni_index.csv",
                      parse_dates=["SALEDATE"],
                      index_col=["SALEDATE"])
port_dat = hvi_dat[top_municodes]
port_dat.to_csv("../clean_data/top20assets.csv")  # save selected municode for portfolio optimization