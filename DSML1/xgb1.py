# Required imports
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt


# Load datasets
nber = pd.read_csv('Datasets/nberces5818v1_n2012.csv') # Base dataset
robotics = pd.read_csv('Datasets/robotics_data.csv')
imf = pd.read_csv('Datasets/IMF.csv', encoding = "latin-1")
wb = pd.read_csv('Datasets/API_TX.VAL.MANF.ZS.UN_DS2_en_csv_v2_7563.csv', skiprows =4)

# derived features
nber = nber[nber["naics"].astype(str).str.startswith("3")].copy()
nber["cap_labor_ratio"] = nber["cap"] / nber["emp"]
nber["non_prod_workers"] = nber["emp"] - nber["prode"]

# target
nber["labor_productivity"] = nber["vadd"] / nber["emp"]
nber["revenue_per_emp"] = nber["vship"] / nber["emp"]

nber = nber.replace([np.inf, -np.inf], np.nan)

# Robotics dataset
robotics_mfg = robotics[robotics["Industry"] == "Manufacturing"].copy()
robotics_mfg = robotics_mfg.rename(columns={
    "Year":"year",
    "Robots_Adopted":"robots_adopted",
    "Productivity_Gain":"robotics_productivity_gain",
    "Cost_Savings":"robotics_cost_savings",
    "Jobs_Displaced": "jobs_displaced",
    "Training_Hours": "training_hours"
})

# WB Manufacturing Export Share
wb_usa = wb[wb["Country Name"] == "United States"].copy()
year_cols_wb = [c for c in wb_usa.columns if str(c).isdigit()]
wb_long = wb_usa.melt(
    id_vars = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
    value_vars = year_cols_wb,
    var_name = "year",
    value_name = "manuf_export_share"
)
wb_long["year"] = wb_long["year"].astype(int)
wb_long["manuf_export_share"] = pd.to_numeric(wb_long["manuf_export_share"])

# IMF dataset
imf_usa = imf[imf["Country"] == "United States"].copy()
imf_usa = imf_usa[imf_usa["WEO Subject Code"].isin(["NGDPDPC","PCPIEPCH", "NGDP_R"])].copy()
year_cols_imf = [c for c in imf_usa.columns if str(c).isdigit() and len(str(c)) == 4]
shared_cols = ["WEO Country Code", "ISO", "Country"]

def melt_imf(df, code, new_name):
    out = df[df["WEO Subject Code"] == code].melt(
        id_vars=shared_cols,
        value_vars=year_cols_imf,
        var_name="year",
        value_name=new_name
    )
    out["year"] = out["year"].astype(int)
    out[new_name] = pd.to_numeric(out[new_name].astype(str).str.replace(",", ""), errors = "coerce")
    return out[shared_cols + ["year", new_name]]
gdp_pc = melt_imf(imf_usa, "NGDPDPC", "gdp_per_capita")
inflation = melt_imf(imf_usa, "PCPIEPCH", "inflation")
real_gdp = melt_imf(imf_usa, "NGDP_R", "real_gdp")
macro = gdp_pc.merge(inflation, on = shared_cols + ["year"], how="inner")
macro = macro.merge(real_gdp, on=shared_cols + ["year"], how="inner")
macro = macro[["year", "gdp_per_capita", "inflation", "real_gdp"]]

df = nber.merge(robotics_mfg, on="year", how = "inner")
df = df.merge(wb_long, on="year", how="left")
df = df.merge(macro, on="year", how="left")
df = df.sort_values(["naics", "year"]).reset_index(drop=True)

print("Merged shape: " + str(df.shape))
print("Years in final dataset: " + str(len(df)))
print(df.head())

#