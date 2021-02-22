import pandas as pd
"""
rocketmortgage.com/learn/what-is-a-multifamily-home
https://www2.alleghenycounty.us/

Alternative data:
https://data.wprdc.org/dataset/allegheny-county-poor-condition-residential-parcel-rates
https://data.wprdc.org/dataset/pre-1950-housing
https://data.wprdc.org/dataset/vacant-properties
https://data.wprdc.org/dataset/anxiety

https://data.wprdc.org/dataset/allegheny-county-911-dispatches-ems-and-fire
"""

# https://data.wprdc.org/dataset/allegheny-county-poor-condition-residential-parcel-rates
dat_poor_house = pd.read_csv("../data/poorhousingconditions.csv")
# This number was calculated by an independent contractor for Allegheny County Economic Development (REINVESTMENT Fund).
# There are instances which are NULL or 0 due to an insufficient amount of sampling.
dat_poor_house["PPoorCon"] = dat_poor_house["PPoorCon"] * 100
dat_poor_house.columns = ["census_tract", "poor"]

# https://data.wprdc.org/dataset/pre-1950-housing
dat_old_house = pd.read_csv("../data/pre1950housing.xls-pre1950housing.csv")
# This data is taken from US Census Data, American Factfinder.
dat_old_house["Pre1950"] = dat_old_house["Pre1950"] * 100
dat_old_house.columns = ["census_tract", "old"]

# https://data.wprdc.org/dataset/vacant-properties
dat_vacant_house = pd.read_csv("../data/vacantpropusps2016q2.csv")[["Tract1", "percent of vacant properties"]]
dat_vacant_house["percent of vacant properties"] = dat_vacant_house["percent of vacant properties"] * 100
dat_vacant_house.columns = ["census_tract", "vacant"]

# https://data.wprdc.org/dataset/anxiety
dat_anxiety = pd.read_csv("../data/anxiety_all_2016.csv")
dat_anxiety["anxiety"] = (dat_anxiety["XPAN"] / dat_anxiety["XPAD"]) * 100.
dat_anxiety = dat_anxiety[["CT", "anxiety"]]
dat_anxiety.columns = ["census_tract", "anxiety"]

# # https://data.wprdc.org/dataset/allegheny-county-911-dispatches-ems-and-fire
# dat_911 = pd.read_csv("../data/ff33ca18-2e0c-4cb5-bdcd-60a5dc3c0418.csv")
# dat_911["census_block_group_center__y"] = dat_911["census_block_group_center__y"].round(2)
# dat_911["census_block_group_center__x"] = dat_911["census_block_group_center__x"].round(2)
# dat_911 = dat_911.groupby(["census_block_group_center__x", "census_block_group_center__y"]).size().reset_index(name="counts")
# dat_911.columns = ["latitude", "longitude", "counts"]

dat_geo = pd.read_csv("../data/parid_geo.csv")
dat_alternative = dat_geo.\
    merge(dat_anxiety, on=["census_tract"], how="left").\
    merge(dat_old_house, on=["census_tract"], how="left").\
    merge(dat_poor_house, on=["census_tract"], how="left").\
    merge(dat_vacant_house, on=["census_tract"], how="left")
dat_alternative.columns = [col.upper() for col in dat_alternative.columns]
dat_alternative = dat_alternative.fillna(0)

# load clean_data
dat = pd.read_pickle("../data/cleandata.pkl")
dat2 = dat.merge(dat_alternative, on=["PARID"])
dat2.to_pickle("../data/cleandata2.pkl")