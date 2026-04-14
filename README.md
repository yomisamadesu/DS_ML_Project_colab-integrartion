This study aims to investigate the evolving role of robotics and automation in economic growth, 
productivity and the accuracy as to which ensemble methods such as XGBoost and LightGBM can forecast those trends.
It examines and takes into account factors that were divided into various groups depending upon their origin. Group 1: 
[capital stock, investment in machinery, capital expenses, 
capital labor ratio], Group 2: [total employees, production workers, non-production workers, total wages, hours worked, production worker wages],
Group 3: [value added, shipments, output, intermediate inputs], Group 4: [country name, country code, indicator name,
indicator code], Group 5 [Country].
Through empirical analysis and predictive modeling this research aims to facilitate  informed policymaking in industry 
and provide general economic forecasting for manufacturing industries specifically.


The datasets and their sources utilized in this research are listed below: (2022-NAICS-Codes-listed-numerically-2-Digit-through-6-Digit.csv, 
2022-NAICS-Codes-listed-numerically-2-Digit-through-6-Digit.xlsx sourced from NAICS, North American Industry 
Classification System 
https://www.naics.com/wp-content/uploads/2022/05/2022-NAICS-Codes-listed-numerically-2-Digit-through-6-Digit.xlsx&ved=2ahUKEwjn55rO3syTAxU8lmoFHZ59MjAQFnoECBoQAQ&usg=AOvVaw36SrAtuwxw3XQbG-lOpnST),
(API_TX.VAL.MANF.ZS.UN_DS2_en_csv_v2_7563.csv sourced from World Bank Group, https://data.worldbank.org/indicator/TX.VAL.MANF.ZS.UN),
(IMF.csv, IMF.xlsx sourced from International Monetary Fund, 
https://www.imf.org/en/publications/weo/weo-database/2025/april/download-entire-database), (nberces5818v1_n2012.csv, 
nberces5818v1_n2012.xlsx, nberces5818v1_summary_stats.pdf sourced from National Bureau of Economic Research, 
https://www.nber.org/research/data/nber-ces-manufacturing-industry-database), 
(OECD_CSV_Version_OECD.STI.PIE,DSD_TIVA_MAINLV@DF_MAINLV,1.1+FFD_DVA...W..A.csv, 
OECD_CSV_Version_OECD.STI.PIE,DSD_TIVA_MAINLV@DF_MAINLV,1.1+FFD_DVA...W..A.xlsx sourced from Organization for Economic 
Co-operation and Development, 
https://data-explorer.oecd.org/vis?pg=0&bp=true&tm=%2522%2528TIVA%2529%25202025%2520edition%2522&snb=5&vw=tb&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_TIVA_MAINLV%2540DF_MAINLV&df[ag]=OECD.STI.PIE&df[vs]=1.1&dq=FFD_DVA...USA..A&pd=1%2C9%2C9%2C5%2C%25%2C2%2CC%2C2%2C0%2C2%2C2&to[TIME_PERIOD]=false), 
and (robotics_data.csv sourced from Kaggle Industrial Robotics and Automation Dataset, 
https://www.kaggle.com/datasets/kennedywanakacha/industrial-robotics-and-automation-dataset/data)

For data normalization and scaling implementation of  RobustScaler would be the best choice as the manufacturing
industry has a wide range of sectors which also contributes to a large number of outliers. As the mean and standard 
deviation of the values would be greatly influenced by the outliers MinMaxScaler and StandardScaler would be greatly 
influenced by the skewed data. Due to the use of Gradient Boosting Models in this research the use of such techniques
are not necessary for tree-boosting models.

For this research we have observed that LightGBM models productivity better than XGBoost, producing an RMSE of ~74 and
MAE around  ~49 compared to XGBoost's ~124 and ~57 respectively. XGboost attains an r^2 score of ~0.01 whereas LightGBM 
attained a score of ~0.64, showing that XGBoost is performing considerably worse when modeling productivity. 
In contrast XGBoost performs better at modeling revenue than LightGBM. XGBoost produces an r^2 score ~0.85 compared to 
LightGBM's 0.65. The most consistent and important features include capital labor ratio, capital stock, equipment investment, 
material costs, and investment. We tested for a +10%, +15% & +25% increase in robot adoption with an accompanied +5% and +15% in employee 
training hours, these results  depicted an uneven positive effect across manufacturing sectors. 
It would appear that the more an organization contributes to capital labor ratio, capital stock, equipment investment, 
and material costs, the more likely they are to benefit from an increase in robot adoption and automation.

Moving forward into Milestone 3 we plan to also implement predictions for the jobs displaced by automation at each of the adoption intervals.