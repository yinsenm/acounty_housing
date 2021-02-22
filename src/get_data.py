# Author: Yinsen Miao
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
sns.set_context("poster")

"""
Reference:

[1] ALLEGHENY county municipality map https://apps.alleghenycounty.us/website/munimap.asp
[2] Ways to compute property tax: https://www.investopedia.com/articles/tax/09/calculate-property-tax.asp
[3] Allegheny County Property Sale Transactions https://data.wprdc.org/dataset/real-estate-sales

"""

"""
important variables considered:
PARID, # Land ID
PROPERTYADDRESS,
PROPERTYZIP,

# Categorical features
SCHOOLCODE, # School district number associated with specified parcel.
SCHOOLDESC, # School district name associated with specified parcel.
NEIGHCODE,  # The ID number for the valuation neighborhood containing this parcel. [*]
NEIGHDESC, 
MUNICODE, # Municipality number associated with specified parcel.
TAXCODE, # Tax Status indicates whether or not real estate taxes apply to a parcel's assessment. E, T, P 
OWNERDESC, # Descriptions for numeric owner codes, e.g. individuals ('REGULAR') vs corporations ('CORPORATION').
CLASSDESC, # Broad self-explanatory categories for describing the general use of a parcel.
USEDESC, # More detailed than class, these categories further describe the primary use of the parcel
SALEDESC, # A subjective categorization (description) as to whether or not the sale price was representative of current market value.
STYLEDESC, # Description for building style
EXTFINISH_DESC, # Description of building material used for exterior walls.
ROOFDESC, # Description of roofing material.
BASEMENTDESC, # Description of basement type, if one exists.

# Ordinal features
GRADEDESC, # Grade refers to quality of construction
CONDITIONDESC, # Description for the overall physical condition
CDUDESC, # CDU is a composite rating Condition, Desirability and Utility
HEATINGCOOLINGDESC, # Description for the type Heating / Cooling system.

# numerical features
LOTAREA, # sum of area of land
FINISHEDLIVINGAREA, # Finished living area, as measured from the outside of the structure.

# Count type of variables
STORIES,  # The story height of the main dwelling
TOTALROOMS, # Total number of rooms in the main dwelling
BEDROOMS, # The total number of separate rooms designed to be used as bedrooms
FULLBATHS, # A full bath has a toilet, sink and bathing facility
HALFBATHS, # A half bath has a toilet and sink only
FIREPLACES, # The number of wood-burning fireplace, chimneys/vents
BSMTGARAGE, # The number of vehicle spaces available in a garage that is basement level


# numerical output features
SALEPRICE, # Amount paid for the sale

---------- tax and fair market value ---------------
COUNTYTOTAL, # The assessed property value 
LOCALTOTAL, # The assessed property value 
FAIRMARKETTOTAL, # The base year appraised fair market value of land and building together
----------------------------------------------------

# datetime
SALEDATE,
YEARBLT,

# other useful information
(PREVSALEDATE, PREVSALEPRICE)
(PREVSALEDATE2, PREVSALEPRICE2)
"""

"""
Data Error:
Line 526782, SALEDATE, 10-11-2018
Line 178775, SALEDATE, 09-02-2019
Line 21993, RECORDDATE, 03-11-2020
Line 391198, RECORDDATE,  08-21-2020
Line 458027, RECORDDATE,  12-22-2020
"""

image_path = "../images"
os.makedirs(image_path, exist_ok=True)
dat_path = "../data/2021-01-assessments.csv" # assessment data
sdat_path = "../data/5bbe6c55-bce6-4edb-9d04-68edeb6bf7b1.csv"  # sale data after 2012-01

# load assessment data of 2021-02
dat = pd.read_csv(dat_path)

# load sale data
sdat = pd.read_csv(sdat_path, parse_dates=["SALEDATE"])

# convert datestring to date
for _date in ["RECORDDATE", "SALEDATE", "PREVSALEDATE", "PREVSALEDATE2"]:
    dat[_date] = pd.to_datetime(dat[_date], format="%m-%d-%Y")

# filter out the properties that are marked with "VALID SALE" or "OTHER VALID"
dat = dat[dat.SALEDESC.isin(["VALID SALE", "OTHER VALID"])].reset_index(drop=True)
sdat = sdat[sdat.SALEDESC.isin(["VALID SALE", "OTHER VALID"])].reset_index(drop=True)

# collect PARID, SALEDATE, SALEPRICE
prc_data1 = dat[["PARID", "SALEDATE", "SALEPRICE"]]
prc_data2 = dat[["PARID", "PREVSALEDATE", "PREVSALEPRICE"]].dropna()
prc_data3 = dat[["PARID", "PREVSALEDATE2", "PREVSALEPRICE2"]].dropna()
prc_data4 = sdat[["PARID", "SALEDATE", "PRICE"]]
prc_data1.columns = prc_data2.columns = prc_data3.columns = prc_data4.columns = ["PARID", "SALEDATE", "PRICE"]

# rbind the above 4 sale data
prc_data = pd.concat([prc_data1, prc_data2, prc_data3, prc_data4], axis=0)
del prc_data1, prc_data2, prc_data3, prc_data4, dat, sdat  # free memory
prc_data = prc_data[prc_data["PRICE"] > 1000].reset_index(drop=True)  # filter out sale price less than 1000
prc_data["SALEDATE"] = prc_data["SALEDATE"] + pd.offsets.MonthEnd(0)  # convert date to month end

# drop duplicated
prc_data = prc_data.drop_duplicates()
prc_data = prc_data.sort_values(["SALEDATE"]).reset_index(drop=True)  # sort by transaction date
prc_data = prc_data[prc_data.SALEDATE > "1980-01-01"]  # filter transaction after 1980-01
prc_data = prc_data[prc_data.SALEDATE < "2021-01-01"].reset_index(drop=True)  # filter transaction before 2021-01-01
# check all included housing ids with valid price
univ_parid = prc_data["PARID"].unique().tolist()

# processing features for each house
dat = pd.read_csv(dat_path)  # reread assessment data
dat = dat[dat["PARID"].isin(univ_parid)]  # filter out data that has valid price

# from CLASSDESC variable remove all other than RESIDENTIAL
dat = dat[dat["CLASSDESC"] == "RESIDENTIAL"]

# let us only consider the following use cases
# I remove other houses type for liquidity reasons, especially Multifamily Home
# since single-family properties are also easier to sell
# https://www.rocketmortgage.com/learn/what-is-a-multifamily-home
dat = dat[dat["USEDESC"].isin(["SINGLE FAMILY"])]

# let us filter the BEDROOMS to a reasonable ranges (0, 7)
dat = dat[dat["BEDROOMS"] < 7]
dat = dat[dat["BEDROOMS"] > 0]

dat["FULLBATHS"] = dat["FULLBATHS"].fillna(value=0)  # if na, then fill with 0
dat["HALFBATHS"] = dat["HALFBATHS"].fillna(value=0)  # if na, then fill with 0

# ADJUSTBATHS = FULLBATHS + HALFBATHS / 2.0
dat["ADJUSTBATHS"] = dat["FULLBATHS"] + dat["HALFBATHS"] / 2.0

# fill na with default values
dat["BASEMENT"] = dat["BASEMENT"].fillna(value=1)
dat["EXTFINISH_DESC"] = dat["EXTFINISH_DESC"].fillna(value="OTHER")
dat["STYLEDESC"] = dat["STYLEDESC"].fillna(value="OTHER")

# let us filter out extreme small or extreme large house
q_min, q_max = dat["FINISHEDLIVINGAREA"].quantile([0.01, 0.99])
dat = dat[dat["FINISHEDLIVINGAREA"] > q_min]
dat = dat[dat["FINISHEDLIVINGAREA"] < q_max]


# let us filter the house via NEIGHCODE, we only keep NEIGHCODE with house number larger than 100
neighcode_house_count = dat["NEIGHCODE"].value_counts()
selected_neigh = neighcode_house_count[neighcode_house_count > 100].index.tolist()
dat = dat[dat["NEIGHCODE"].isin(selected_neigh)]

# let us filter the house via MUNICODE, we only keep MUNICODE with house number larger than 100
municode_house_count = dat["MUNICODE"].value_counts()
selected_municode = municode_house_count[municode_house_count > 100].index.tolist()
dat = dat[dat["MUNICODE"].isin(selected_municode)]


# now we update univ_parid
univ_parid = dat["PARID"].unique().tolist()


# filter the valid transaction given univ_parid
prc_data = prc_data[prc_data["PARID"].isin(univ_parid)].reset_index(drop=True)

# visualization, volume per year
len(prc_data)

# visualization, volume vs. date
ax = prc_data.groupby(["SALEDATE"]).size().plot(figsize=(12, 6))
ax.set_xlim(pd.Timestamp('1980-01-01'), pd.Timestamp('2021-05-01'))
ax.set_xlabel(""); ax.set_ylabel("Volume");
ax.figure.savefig("%s/volumne.png" % image_path)
plt.close()

# visualization, median sell price vs. date
ax = prc_data.groupby(["SALEDATE"])["PRICE"].median().plot(figsize=(12, 6))
ax.set_xlim(pd.Timestamp('1980-01-01'), pd.Timestamp('2021-05-01'))
ax.set_xlabel(""); ax.set_ylabel("Median House Price ($)");
f = lambda x, pos: f'{x/10**3:,.0f}K'
ax.yaxis.set_major_formatter(FuncFormatter(f))
ax.figure.savefig("%s/median_price.png" % image_path)
plt.close()


# ordinal variables
# GRADEDESC
grade_rank = [
    'POOR',
    'BELOW AVERAGE -', 'BELOW AVERAGE', 'BELOW AVERAGE +',
    'AVERAGE -', 'AVERAGE', 'AVERAGE +',
    'GOOD -', 'GOOD', 'GOOD +',
    'VERY GOOD -', 'VERY GOOD', 'VERY GOOD +',
    'EXCELLENT -', 'EXCELLENT', 'EXCELLENT +'
][::-1]
dat["GRADEDESC"] = dat["GRADEDESC"].fillna(value="AVERAGE")  # fill na with AVERAGE
grade_encoder = OrdinalEncoder(categories=[grade_rank])
dat["GRADERANK"] = grade_encoder.fit_transform(dat["GRADEDESC"].values.reshape(-1, 1))


# CDUDESC ranking CONDITIONDESC and CDUDESC overlapped 90%, here let us use CDUDESC
cdu_rank = [
    'EXCELLENT', 'VERY GOOD', 'GOOD',
    'AVERAGE', 'FAIR', 'POOR', 'VERY POOR', 'UNSOUND'
]
dat["CDUDESC"] = dat["CDUDESC"].fillna(value="AVERAGE")  # fill na with AVERAGE
cdu_encoder = OrdinalEncoder(categories=[cdu_rank])
dat["CDURANK"] = cdu_encoder.fit_transform(dat["CDUDESC"].values.reshape(-1, 1))

# read school grade
school_dict = pd.read_csv("../data/school_district_score.csv").set_index("SCHOOLDESC")['GRADE'].to_dict()
dat["SCHOOLGRADE"] = dat["SCHOOLDESC"].map(school_dict)
dat["SCHOOLGRADE"] = dat["SCHOOLGRADE"].fillna(value="C")
school_rank = [
    "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+"
]
school_encoder = OrdinalEncoder(categories=[school_rank])
dat["SCHOOLRANK"] = school_encoder.fit_transform(dat["SCHOOLGRADE"].values.reshape(-1, 1))



# map STYLEDESC to the top 4 major STYLEs
stypedesc_map = {
    "COLONIAL": "COLONIAL", "OLD STYLE": "OLD STYLE",
    "RANCH": "RANCH", "CAPE COD": "CAPE COD",
    "BI-LEVEL": "BI-LEVEL",
    "SPLIT LEVEL": "OTHER",
    "SEMI DETACHED": "OTHER",
    "BUNGALOW": "OTHER",
    "CONTEMPORARY": "OTHER",
    "TUDOR": "OTHER",
    "VICTORIAN": "OTHER",
    "MULTI-FAMILY": "OTHER",
    "MODULAR HOME": "OTHER",
    "LOG CABIN": "OTHER",
    "MANUFACTURED": "OTHER",
    "TOWNHOUSE": "OTHER",
    "CONVENTIONAL": "OTHER",
    "ROW END": "OTHER",
    "CONDO": "OTHER",
    "ROW INTERIOR": "OTHER",
    "OTHER": "OTHER"
}
dat["STYLEDESC"] = dat["STYLEDESC"].replace(stypedesc_map)


# map EXTFINISH_DESC to the top 4 major EXTFINISHs
extfinish_desc_map = {
    "Brick": "Brick",
    "Frame": "Frame",
    "Masonry FRAME": "Masonry FRAME",
    "Stone": "OTHER",
    "Stucco": "OTHER",
    "Concrete Block": "OTHER",
    "Log": "OTHER",
    "Concrete": "OTHER",
    "OTHER": "OTHER",
}
dat["EXTFINISH_DESC"] = dat["EXTFINISH_DESC"].replace(extfinish_desc_map)

# after converting ordinal values to ranking values
# here we list all possible numerical variables
numeric_feat_names = [
    "GRADERANK",
    "CDURANK",
    "SCHOOLRANK",
    "LOTAREA",
    "FINISHEDLIVINGAREA",
    "STORIES",
    "BEDROOMS",
    "ADJUSTBATHS",   # FULLBATHS + HALFBATHS / 2.0
    "BSMTGARAGE",
    "FIREPLACES",
    "YEARBLT",
    "FAIRMARKETTOTAL",
    "BASEMENT"
]

dat["EXTFINISH_DESC"].value_counts()
# clean data and remove na
dat["FIREPLACES"] = dat["FIREPLACES"].fillna(value=0)  # if na, then fill with 0
dat["BSMTGARAGE"] = dat["BSMTGARAGE"].fillna(value=0)  # if na, then fill with 0


# nominal categorical features
nominal_feat_names = [
    "PARID", "MUNICODE", "EXTFINISH_DESC", "STYLEDESC", "NEIGHCODE", "NEIGHDESC", "PROPERTYZIP"
]

# visualization, count versus # of sales
ax = prc_data["PARID"].value_counts().value_counts().plot(figsize=(10, 8), style="-X")
ax.set_xlabel("# of Sales"); ax.set_ylabel("Count");
f = lambda x, pos: f'{x/10**3:,.0f}K'
ax.yaxis.set_major_formatter(FuncFormatter(f))
ax.figure.savefig("%s/sales.png" % image_path)
plt.close()

# # check for any house with larger than 2 sale transactions, let us see the holding in month
# holding_months = []
# univ_parid = prc_data["PARID"].value_counts()
# for parid in univ_parid[univ_parid > 1].index:
#     sale_dat = prc_data[prc_data["PARID"] == parid].sort_values(["SALEDATE"])
#     holding_months.append((sale_dat["SALEDATE"].diff()[1:].dt.days / 30).tolist())
# con_holding_months = sum(holding_months, [])
#
# # check holding time for parid that is sold only one time
# for parid in univ_parid[univ_parid == 1].index:
#     sale_dat = prc_data[prc_data["PARID"] == parid]["SALEDATE"]
#     con_holding_months.append(float((pd.to_datetime("2021-02-01") - sale_dat).dt.days / 30))
#
# con_holding_years = np.array(con_holding_months) / 12.
#
# # now let us check the histogram of holding in years
# _min, _max, _binwidth = min(con_holding_years), max(con_holding_years), 1
# plt.figure(figsize=(8, 6))
# plt.hist(con_holding_years, bins=np.arange(_min, _max + _binwidth, _binwidth))
# plt.savefig("hist.png")
# plt.close()
full_data = pd.merge(prc_data, dat[numeric_feat_names + nominal_feat_names], how="left", on=["PARID"])
full_data = full_data.query("LOTAREA > 0")  # remove row with LOTAREA equal to 0

print("Get %d single family house with valid pricing!" % len(full_data["PARID"].unique()))
print("Get %d single family with valid sale records!" % len(full_data))

# check frequency of trading
print("Count of trading ...")
print(full_data["PARID"].value_counts().value_counts())

full_data.to_pickle("../data/cleandata.pkl")
print("Data cleaning done!")