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

# naics sector name retrieval
naics_file = pd.read_excel('/Users/brandon/Documents/DSMLProject/DS_ML_Project_colab-integration/DSML1/Datasets/2022-NAICS-Codes-listed-numerically-2-Digit-through-6-Digit.xlsx',
sheet_name = "Two-Six Digit NAICS")

# Search for  Manufactuuring codes
def best_naics_match(code, naics_lookup):
    code = str(code).strip()
    if code in naics_lookup:
        return naics_lookup[code], "exact"

    for length in [5, 4, 3, 2]:
        prefix = code[:length]
        if prefix in naics_lookup:
            return naics_lookup[prefix], f"prefix_{length}_digit"
    return "Title not found in NAICS file", "unmatched"

# shorten the names of sectors for readability
def shorten_title(title, max_len = 40):
    if pd.isna(title):
        return "Unknown"
    title = str(title).strip()
    replacements = {
        "Manufacturing": "Mfg",
        "manufacturing": "Mfg",
        "and":"&",
        "And":"&",
        "Products":"Prod",
        "Product":"Prod",
        "Equipment":"Equip",
        "Services":"Svc",
        "Service":"Svc",
        "Preparation":"Prep",
        "Processing":"Proc",
        "Fabricated":"Fab",
        "Industries":"Ind",
        "Industry":"Ind",
        "Wholesale":"Whsl",
        "Except": "Ex.",
        "Miscellaneous":"Misc",
        "Machiner":"Mach",
        "Electrical":"Elec",
        "Electronics":"Elec",
        "Beverages":"Bev",
        "Textiles":"Text",
        "Transportation":"Transp",
        "Printing":"Print",
        "Chemical":"Chem",
        "Pharmaceutical":"Pharm",
        "Medical":"Med",
        "Supplies":"Sup",
        "Apparel": "App",
        "Furniture":"Furn",
        "Plastics":"Plasts",
        "Rubber":"Rubber",
        "Paperboard":"Paperbd",
        "Paper":"Paper",
        "Packaging": "Pkg"
    }

    for old, new in replacements.items():
        title = title.replace(old, new)
    title = " ".join(title.split())

    if len(title) > max_len:
        title = title[:max_len - 3] + "..."
    return title
def top_feature_importance(model, feature_names, top_n=15):
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(top_n)
    return imp
def melt_imf(df, subject_code, new_name, year_cols):
    out = df[df["WEO Subject Code"] == subject_code].melt(
        id_vars = ["Country"],
        value_vars = year_cols,
        var_name="year",
        value_name = new_name
    )
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out[new_name] = pd.to_numeric(
        out[new_name].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype(int)
    out = out[["Country", "year", new_name]].drop_duplicates(subset=["Country","year"])
    return out

# Gather top features of the model
def top_features_importance(model, feature_names, top_n=15):
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending = "False").head(top_n)
    return imp

# Create lookup
naics_file = naics_file.rename(columns={
    "2022 NAICS US Code": "naics_code",
    "2022 NAICS US Title": "naics_title"
})

naics_file["naics_code"] = naics_file["naics_code"].astype(str).str.strip()
naics_file["naics_title"] = naics_file["naics_title"].astype(str).str.strip()

file = naics_file[["naics_code", "naics_title"]].drop_dupplicates()
lookup = dict(zip(naics_file["naics_code"], naics_file["naics_title"]))

# data preparation
nber["naics"] = nber["naics"].astype(str).str.strip()
nber = nber[nber["naics"].str.startswith(("31","32","33"))].copy()

# derived features
nber["cap_labor_ratio"] = nber["cap"] / nber["emp"]
nber["non_prod_workers"] = nber["emp"] - nber["prode"]

# target
nber["labor_productivity"] = nber["vadd"] / nber["emp"]
nber["revenue_per_emp"] = nber["vship"] / nber["emp"]

nber = nber.replace([np.inf, -np.inf], np.nan)

# add cleaned labels
naics_matches = nber["naics"].drop_duplicates().to_frame()
naics_matches[["naics_title", "match_level"]] = naics_matches["naics"].apply(
    lambda x: pd.Series(best_naics_match(x, lookup))
)
naics_matches["short_title"] = naics_matches["naics_title"] .apply(shorten_title)
naics_matches["short_label"] = naics_matches["naics"] + " | " + naics_matches["short_title"]
naics_matches["full_label"] = naics_matches["naics"] + " - " + naics_matches["naics_title"]
nber = nber.merge(naics_matches, on = "naics", how="left")

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
wb_long["year"] = pd.to_numeric(wb_long["year"], errors= "coerce")
wb_long["manuf_export_share"] = pd.to_numeric(wb_long["manuf_export_share"], errors="coerce")
wb_long = wb_long.dropna(subset=["year"])
wb_long["year"] = wb_long["year"].astype(int)
wb_long = wb_long[["year", "manuf_export_share"]]

# IMF dataset
imf_usa = imf[imf["Country"] == "United States"].copy()
imf_usa = imf_usa[imf_usa["WEO Subject Code"].isin(["NGDPDPC","PCPIEPCH", "NGDP_R"])].copy()
year_cols_imf = [c for c in imf_usa.columns if str(c).isdigit() and len(str(c)) == 4]


gdp_pc = melt_imf(imf_usa, "NGDPDPC", "gdp_per_capita")
inflation = melt_imf(imf_usa, "PCPIEPCH", "inflation")
real_gdp = melt_imf(imf_usa, "NGDP_R", "real_gdp")
macro = gdp_pc.merge(inflation, on = ["Country", "year"], how="inner")
macro = macro.merge(real_gdp, on=["Country", "year"], how="inner")
macro = macro[["year", "gdp_per_capita", "inflation", "real_gdp"]]

df = nber.merge(robotics_mfg, on="year", how = "inner")
df = df.merge(wb_long, on="year", how="left")
df = df.merge(macro, on="year", how="left")
df = df.sort_values(["naics", "year"]).reset_index(drop=True)

print("Merged shape: " + str(df.shape))
print("Years in final dataset: " + str(len(df)))

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
required_cols = (["naics", "naics_title", "short_title", "short_label", "full_label", "match_level", "year"] +
                 feature_cols + ["labor_productivity", "revenue_per_worker"])
for col in feature_cols + ["labor_productivity", "revenue_per_worker"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
model_df = df[required_cols].dropna().copy()
print("Modeling rows after NA drop: ", model_df.shape)

# Model Creation
# Training/ Testing
X = model_df[feature_cols]
y_prod = model_df["labor_productivity"]
y_rev = model_df["revenue_per_worker"]

# Split to ensure test & train have different values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
train_set, test_set = next(gss.split(X, y_prod, groups=model_df["naics"]))
X_train = X.iloc[train_set]
X_test = X.iloc[test_set]

y_prod_train = y_prod.iloc[train_set]
y_prod_test = y_prod.iloc[test_set]
y_rev_train = y_rev.iloc[train_set]
y_rev_test = y_rev.iloc[test_set]

xgb_model = XGBRegressor(
     n_estimators=300,
     max_depth=8,
     learning_rate=0.05,
     subsample=0.9,
     colsample_bytree=0.9,
     reg_alpha=0.1,
     reg_lamba=1.0,
     random_state=2026,
 )
prod_model = xgb_model.fit(X_train, y_prod_train)
rev_model = xgb_model.fit(X_train, y_rev_train)

prod_preds_test = prod_model.predict(X_test)
rev_pred_test = rev_model.predict(X_test)

prod_rmse = np.sqrt(mean_squared_error(y_prod_test, prod_preds_test))
prod_mae = mean_absolute_error(y_prod_test, prod_preds_test)
prod_r2 = r2_score(y_prod_test, prod_preds_test)
print(f"RMSE: {prod_rmse:.4f}")
print(f"MAE: {prod_mae:.4f}")
print(f"R2: {prod_r2:.4f}")

rev_rmse = np.sqrt(mean_squared_error(y_rev_test, rev_pred_test))
rev_mae = mean_absolute_error(y_rev_test, rev_pred_test)
rev_r2 = r2_score(y_rev_test, rev_pred_test)
print(f"RMSE: {rev_rmse:.4f}")
print(f"MAE: {rev_mae:.4f}")
print(f"R2: {rev_r2:.4f}")

# Robot/automation scenario adoption
# Scenario 1 10% adoption 5% training
X_all = model_df[feature_cols].copy()
scenario_all = X_all.copy()

scenario_all["robots_adopted"] = scenario_all["robots_adopted"] * 1.10
scenario_all["training_hours"] = scenario_all["training_hours"] * 1.05

baseline_prod = prod_model.predict(X_all)
scenario_prod = prod_model.predict(scenario_all)

baseline_rev = rev_model.predict(X_all)
scenario_rev = rev_model.predict(scenario_all)

results = model_df[[
    "naics", "naics_title", "short_title", "short_label", "full_label", "match_level",
    "year"
]].copy()
results["baseline_productivity"] = baseline_prod
results["scenario_productivity"] = scenario_prod
results["productivity_change"] = results["scenario_productivity"] - results["baseline_productivity"]
results["prod_pct_change"] = np.where(
    results["baseline_productivity"] != 0,
    (results["productivity_change"] / results["baseline_productivity"]) * 100,
    np.nan
)
results["baseline_revenue_per_worker"] = baseline_rev
results["scenario_revenue_per_worker"] = scenario_rev
results["revenue_change"] = results["scenario_revenue_per_worker"] - results["baseline_revenue_per_worker"]
results["revenue_pct_change"] = np.where(
    results["baseline_revenue_per_worker"] != 0,
    (results["revenue_change"] / results["baseline_revenue_per_worker"]) * 100,
    np.nan
)

prod_importance = top_feature_importance(prod_model, feature_cols, top_n=15)
rev_importance = top_feature_importance(rev_model, feature_cols, top_n=15)

print("TOP PRODUCTIVITY MODEL FEATURES")
print(prod_importance.to_string(index=False))
print()

print("TOP REVENUE MODEL FEATURES")
print(rev_importance.to_string(index=False))
print()
# Sector summaries
sector_summary = (results.groupby(["naics", "naics_title", "short_title", "short_label", "full_label",
                                   "match_level"], as_index=False).agg(
    avg_prod_pct_change = ("prod_pct_change", "mean"),
    avg_revenue_pct_change = ("revenue_pct_change", "mean"),
    avg_baseline_revenue = ("baseline_revenue_per_worker", "mean"),
    avg_scenario_revenue = ("scenario_revenue_per_worker", "mean")
).sort_values("avg_prod_pct_change", ascending = False).reset_index(drop=True))
top15 = sector_summary.head(15).copy()
top_prod_features = prod_importance["feature"].head(5).tolist()
top_rev_features = rev_importance["feature"].head(5).tolist()

sector_explanations = top15[[
    "naics", "naics_title", "short_label",
    "avg_prod_pct_change", "avg_revenue_pct_change"
]].copy()

sector_explanations["productivity_explanation"] = (
    "High predicted productivity gain; model most influenced by: " +
    ", ".join(top_prod_features)
)

sector_explanations["revenue_explanation"] = (
    "Projected revenue response mainly associated with: " +
    ", ".join(top_rev_features)
)

print("TOP 15 SECTOR EXPLANATIONS")
print(sector_explanations[[
    "short_label", "avg_prod_pct_change", "avg_revenue_pct_change",
    "productivity_explanation", "revenue_explanation"
]].to_string(index=False))
print()

# =========================================================
# 17. EXPORT FILES
# =========================================================
naics_matches.to_csv("nber_naics_readable_mapping.csv", index=False)
results.to_csv("xgboost_scenario_predictions_by_sector_year.csv", index=False)
sector_summary.to_csv("sector_summary_with_readable_naics.csv", index=False)
top15.to_csv("top15_sectors_productivity_and_revenue.csv", index=False)
sector_explanations.to_csv("top15_sector_explanations.csv", index=False)
prod_importance.to_csv("productivity_feature_importance.csv", index=False)
rev_importance.to_csv("revenue_feature_importance.csv", index=False)

metrics = pd.DataFrame({
    "model": ["productivity", "revenue_per_worker"],
    "rmse": [prod_rmse, rev_rmse],
    "mae": [prod_mae, rev_mae],
    "r2": [prod_r2, rev_r2],
    "rows_used": [len(model_df), len(model_df)],
    "years_used": [f"{model_df['year'].min()}-{model_df['year'].max()}"] * 2
})
metrics.to_csv("xgboost_model_metrics.csv", index=False)

# Line chart & productivity
plt.figure(figsize=(16, 7))
plt.plot(top15["short_label"], top15["avg_prod_pct_change"], marker="o", linewidth=2)
plt.xticks(rotation=60, ha="right")
plt.xlabel("Top 15 Manufacturing Sectors")
plt.ylabel("Avg % Productivity Increase")
plt.title("Top 15 Sectors: Productivity Impact (+10% Automation)")
plt.tight_layout()
plt.savefig("chart_1_productivity_percent_increase_line.png", dpi=300, bbox_inches="tight")
plt.show()

# bar chart for productivity

plt.figure(figsize=(16, 7))
plt.bar(top15["short_label"], top15["avg_prod_pct_change"])
plt.xticks(rotation=60, ha="right")
plt.xlabel("Top 15 Manufacturing Sectors")
plt.ylabel("Avg % Productivity Increase")
plt.title("Productivity % Increase by Sector")
plt.tight_layout()
plt.savefig("chart_2_productivity_percent_increase_bar.png", dpi=300, bbox_inches="tight")
plt.show()

# line chart for predicted revenue
rev_plot = top15[["short_label", "avg_baseline_revenue", "avg_scenario_revenue"]].copy()

plt.figure(figsize=(16, 7))
plt.plot(rev_plot["short_label"], rev_plot["avg_baseline_revenue"], marker="o", linewidth=2, label="Baseline")
plt.plot(rev_plot["short_label"], rev_plot["avg_scenario_revenue"], marker="o", linewidth=2, label="Scenario")
plt.xticks(rotation=60, ha="right")
plt.xlabel("Top 15 Manufacturing Sectors")
plt.ylabel("Revenue per Worker")
plt.title("Projected Revenue Trend (+10% Automation)")
plt.legend()
plt.tight_layout()
plt.savefig("chart_3_projected_revenue_trend_line.png", dpi=300, bbox_inches="tight")
plt.show()

# bar chart for projected revenue
x = np.arange(len(rev_plot))
width = 0.35

plt.figure(figsize=(16, 7))
plt.bar(x - width / 2, rev_plot["avg_baseline_revenue"], width, label="Baseline")
plt.bar(x + width / 2, rev_plot["avg_scenario_revenue"], width, label="Scenario")
plt.xticks(x, rev_plot["short_label"], rotation=60, ha="right")
plt.xlabel("Top 15 Manufacturing Sectors")
plt.ylabel("Revenue per Worker")
plt.title("Revenue Comparison (Baseline vs Automation)")
plt.legend()
plt.tight_layout()
plt.savefig("chart_4_projected_revenue_trend_bar.png", dpi=300, bbox_inches="tight")
plt.show()

# bar chart for feature importance
plt.figure(figsize=(12, 6))
plt.barh(prod_importance["feature"][::-1], prod_importance["importance"][::-1])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Productivity Model Features")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.barh(rev_importance["feature"][::-1], rev_importance["importance"][::-1])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Revenue Model Features")
plt.tight_layout()
plt.show()

print("Pipeline complete.")
print("Saved files:")
print("- nber_naics_readable_mapping.csv")
print("- xgboost_scenario_predictions_by_sector_year.csv")
print("- sector_summary_with_readable_naics.csv")
print("- top15_sectors_productivity_and_revenue.csv")
print("- top15_sector_explanations.csv")
print("- productivity_feature_importance.csv")
print("- revenue_feature_importance.csv")
print("- xgboost_model_metrics.csv")
print("- chart_1_productivity_percent_increase_line.png")
print("- chart_2_productivity_percent_increase_bar.png")
print("- chart_3_projected_revenue_trend_line.png")
print("- chart_4_projected_revenue_trend_bar.png")
