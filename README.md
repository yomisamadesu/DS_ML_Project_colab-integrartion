**This project was coded in Python and requires the following dependency installation**

`pip install pandas numpy matplotlib xgboost lightgbm scikit-learn scipy openpyxl`

Predictive Analysis of Automation in U.S. Manufacturing utilizing tree-based modeling
XGBoost vs LightGBM | Productivity and Revenue

**Overview**

This project analyzes the impact of industrial robotics and automation on U.S. manufacturing sectors using machine learning. It builds predictive models to estimate:
* Labor Productivity
* Revenue per Worker

The project compares XGBoost and LightGBM to evaluate how well each model captures the relationship between automation, economic structure, and macroeconomic conditions.

**Objectives**
* Evaluate how automation affects productivity and revenue across sectors
* Compare performance of XGBoost vs LightGBM
* Forecast future outcomes (2024–2027) under automation scenarios

**Models Used**
* XGBoost (Extreme Gradient Boosting)
* LightGBM (Light Gradient Boosting Machine)

Both models:
Use identical hyperparameters for fair comparison
Are trained on sector-level manufacturing data
Predict multiple economic targets

**Data Sources**

The dataset is built from multiple sources:
* National Bureau of Economic Research [NBER Manufacturing Dataset](https://www.naics.com/wp-content/uploads/2022/05/2022-NAICS-Codes-listed-numerically-2-Digit-through-6-Digit.xlsx&ved=2ahUKEwjn55rO3syTAxU8lmoFHZ59MjAQFnoECBoQAQ&usg=AOvVaw36SrAtuwxw3XQbG-lOpnST)
* Kaggle [Industrial Robotics Adoption Dataset](https://www.kaggle.com/datasets/kennedywanakacha/industrial-robotics-and-automation-dataset/data) 
* Organisation for Economic Co-operation and Development [Trade In Value Added Dataset](https://data-explorer.oecd.org/vis?pg=0&bp=true&tm=%2522%2528TIVA%2529%25202025%2520edition%2522&snb=5&vw=tb&df%5Bds%5D=dsDisseminateFinalDMZ&df%5Bid%5D=DSD_TIVA_MAINLV%2540DF_MAINLV&df%5Bag%5D=OECD.STI.PIE&df%5Bvs%5D=1.1&dq=FFD_DVA...USA..A&pd=1%2C9%2C9%2C5%2C%25%2C2%2CC%2C2%2C0%2C2%2C2&to%5BTIME_PERIOD%5D=false)
* World Bank [Manufacture's exports Dataset](https://data.worldbank.org/indicator/TX.VAL.MANF.ZS.UN)
* North American Industry Classification System [Manufacturing Industry Code Dataset] (https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.naics.com/wp-content/uploads/2022/05/2022-NAICS-Codes-listed-numerically-2-Digit-through-6-Digit.xlsx&ved=2ahUKEwj7w9KMkIyUAxXkxckDHU9eHVsQFnoECBcQAQ&usg=AOvVaw36SrAtuwxw3XQbG-lOpnST)

**Features**
Key features include:
* Structural Variables
* Capital stock, plant, equipment, investment
* Employment, wages, production hours
* Material costs

Automation Variables
* Robots adopted
* Productivity gains from robotics
* Cost savings
* Training hours
* Jobs displaced

Macroeconomic Variables
* GDP per capita
* Inflation
* Real GDP
* Manufacturing export share

**Pipeline**
1. Data Cleaning
2. Remove missing/infinite values
3. Convert all features to numeric
4. Feature Engineering
5. Capital-to-labor ratio
6. Labor productivity
7. Revenue per worker
8. Train/Test Split
9. Group-based split by NAICS sector
10. Prevents data leakage across industries
11. Model Training
12. XGBoost and LightGBM trained separately
13. Evaluated using RMSE, MAE, R²
14. Scenario Analysis
15. Simulate increased automation
16. Measure impact on productivity, revenue, and jobs
17. Forecasting
18. Predict sector outcomes for 2024–2027
19. Uses hybrid nonlinear + sector-aware forecasting 

Key Results
Model Performance
Model	Target	R² Score
XGBoost	Productivity	Low / Negative
LightGBM	Productivity	High
XGBoost	Revenue per Worker	High
LightGBM	Revenue per Worker	Moderate

**Insights**
* Capital-labor ratio is the strongest predictor
* Robotics variables show low short-term importance
* XGBoost struggles with productivity due to:
* Small dataset (2015–2018)
* Sector-based validation (hard generalization)


**Forecast Highlights (2024–2027)**

Capital-intensive sectors dominate:
* Petroleum refining
* Chemical manufacturing

Model differences:
* XGBoost favors capital-heavy sectors
* LightGBM highlights investment-driven sectors

**Visualizations**
* Model comparison charts (RMSE, MAE, R²)
* Feature importance plots
* Automation scenario impact charts
* Future projections by sector

**Known Limitations**
* Limited time window (2015–2018)
* Robotics data is aggregated, not sector-specific
* Small dataset (~728 observations)
* Tree models may not react to small scenario changes
