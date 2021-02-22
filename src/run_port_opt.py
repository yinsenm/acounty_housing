# Author Yinsen Miao
import pandas as pd
import numpy as np
from mvo.util import get_mean_variance_space, plot_efficient_frontiers, get_frontier_limits
from mvo.portfolio_analysze import evaluate_port_performance
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

prc = pd.read_csv("../data/top20assets.csv",
                  parse_dates=["SALEDATE"],
                  index_col=["SALEDATE"])
# split data into training and testing
train_prc = prc["2005-12-31": "2015-12-31"]
test_prc = prc["2015-12-31": ]

train_ret = train_prc.pct_change().dropna()
test_ret = test_prc.pct_change().dropna()


target_volatilities_array = np.linspace(0.005, 0.025, num=10)
obj_function_list = ['equalWeighting', 'minVariance', 'maxReturn',  # optimization target
                     'maxSharpe', 'maxSortino', 'riskParity']

result = get_mean_variance_space(train_ret, target_volatilities_array,
                                 obj_function_list, cov_function="GS1")

dat = pd.DataFrame(result["asset"]).transpose()
dat.columns = ["ret", "std"]
plt.figure(figsize=(10, 8))
plt.plot(dat["std"], dat["ret"], 'o')
plt.plot(result["mvo"]["stds"], result["mvo"]["rets"])
for obj_function, ret_std in result['asset'].items():
    plt.scatter(ret_std[1], ret_std[0], marker='o', c='green', s=20, alpha=0.5)
    plt.text(ret_std[1], ret_std[0], obj_function, verticalalignment='top',
             horizontalalignment='left', fontsize=12, c='green', alpha=0.5)
for port_name, port_dict in result["port_opt"].items():
    if port_name == "maxSharpe":
        ret, rsk = port_dict["ret_std"]
        plt.plot(rsk, ret, marker="*", markersize=20)
        plt.text(rsk, ret, port_name, verticalalignment='top',
                 horizontalalignment='left', fontsize=14, c='black', alpha=0.5)
plt.xlabel("Annualized Volatility (Standard Deviation)")
plt.ylabel("Annualized Return")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.savefig("../images/mvo.png")
plt.close()


# compute cumulative return
for port_name, port_dict in result["port_opt"].items():
    wgt = port_dict["weights"]
    cumu_ret = (test_ret.dot(wgt) + 1).prod()
    print("%s $%3.2f, ret %3.2f%%" % (port_name, (cumu_ret) * 5, (cumu_ret - 1) * 100))

cumu_ret = prc.loc["2020-12-31"] / prc.loc["2015-12-31"] - 1.0
cumu_ret.quantile([0.05, 0.5, 0.95])


equity_curves = []
port_names = []
# compute equity curve
for port_name, port_dict in result["port_opt"].items():
    wgt = port_dict["weights"]
    equity_curve = np.cumprod(test_ret.dot(wgt) + 1) * 5e6
    equity_curve.index.name = port_name
    equity_curves.append(equity_curve)
    port_names.append(port_name)
equity_curve_df = pd.concat(equity_curves, axis=1).reset_index()
equity_curve_df.columns = ["date"] + port_names
equity_curve_df.to_csv("../clean_data/equity_curves.csv", index=False)

# compute performance
perf_df = evaluate_port_performance("../clean_data/equity_curves.csv")
perf_df.to_csv("../clean_data/performance.csv")
print(perf_df)


# read HVI market performance
market = pd.read_csv("../clean_data/acounty_index.csv")
market.columns = ["date", "index"]
market.set_index("date", inplace=True)
cumu_ret = market.loc["2020-12-31"] / market.loc["2015-12-31"] - 1.0
market["2015-12-31":].to_csv("../clean_data/temp.csv")

market_perf_df = evaluate_port_performance("../clean_data/temp.csv")
print(market_perf_df)