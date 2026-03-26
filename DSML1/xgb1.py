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

# productivity

target_col = "labor_productivity"

# productivity feature
feature_cols = [
    # Group 1
    "cap", "plant", "equip", "invest", "cap_labor_ratio",

    # Group 2
    "emp", "prode", "non_prod_workers", "pay", "prodh", "prodw",

    # Group 3
    "matcost",

    # Automation
    "robots_adopted", "robotics_productivity_gain", "robotics_cost_savings", "jobs_displaced", "training_hours",

    # Group 4
    "manuf_export_share",

    # Group 5
    "gdp_per_capita", "inflation", "real_gdp",

    # time-scale
    "year"
]

model_df = df[["naics", target_col] + feature_cols].copy()
model_df = model_df.dropna()
print("Modeling rows after NA drop: ", model_df.shape)

# Model Creation
# Training/ Testing
X = model_df[feature_cols]
y = model_df[target_col]

# Split to ensure test & train have different values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
train_set, test_set = next(gss.split(X, y, groups=model_df["naics"]))

X_train, X_test = X.iloc[train_set], X.iloc[test_set]
y_train, y_test = y.iloc[train_set], y.iloc[test_set]

xgb_model = XGBRegressor(
     n_estimators=300,
     max_depth=8,
     learning_rate=0.5,
     subsample=0.9,
     colsample_bytree=0.9,
     reg_alpha=0.1,
     reg_lamba=1.0,
     random_state=2026,
 )
xgb_model.fit(X_train, y_train)
preds = xgb_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending = False)

print(importance.head(15))

plt.figure(figsize=(10,8))
plt.barh(importance["feature"].head(15)[::-1], importance["importance"].head(15)[::-1])
plt.xlabel("Importance")
plt.title("Top 15 XGBoost Feature Importances")
plt.tight_layout()
plt.show()
