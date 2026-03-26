import pandas as pd
import numpy as np
file = '/Users/brandon/Documents/DSMLProject/DS_ML_Project_colab-integration/DSML1/Datasets/OECD_CSV_Version_OECD.STI.PIE,DSD_TIVA_MAINLV@DF_MAINLV,1.1+FFD_DVA...W..A.csv'
data = pd.read_csv(file)

data = data[['REF_AREA', 'Reference area', 'ACTIVITY', 'Economic activity',
             'MEASURE', 'Measure', 'TIME_PERIOD', 'OBS_VALUE']].copy()

data = data[data['REF_AREA'] == 'USA'].copy()
data_subsectors = data[data['ACTIVITY'].astype(str).str.startswith('C')].copy()

group4_features = [
    'EXGR_DVA',
    'EXGR_FVA',
    'EXGR',
    'DEXFVApSH',
    'FEXDVApSH'
]

data_subsectors = data_subsectors[data_subsectors['MEASURE'].isin(group4_features)].copy()
data_subsectors['TIME_PERIOD'] = pd.to_numeric(data_subsectors['TIME_PERIOD'], errors='coerce')
data_subsectors['OBS_VALUE'] = pd.to_numeric(data_subsectors['OBS_VALUE'], errors='coerce')
data_subsectors = data_subsectors.dropna(subset=['TIME_PERIOD', 'OBS_VALUE'])

data_cleaned = data_subsectors.pivot_table(
    index=['REF_AREA', 'Reference area', 'ACTIVITY', 'Economic activity', 'TIME_PERIOD'],
    columns='MEASURE',
    values='OBS_VALUE',
    aggfunc='mean'
).reset_index()

data_cleaned.columns.name = None

data_cleaned = data_cleaned.rename(columns={
    'TIME_PERIOD': 'year',
    'EXGR_DVA': 'domestic_value_in_exports',
    'EXGR_FVA': 'foreign_value_added_in_exports',
    'DEXFVApSH': 'gvc_backward_participation',
    'FEXDVApSH': 'gvc_forward_participation',
})

if {'gvc_backward_participation', 'gvc_forward_participation'}.issubset(data_cleaned.columns):
    data_cleaned['gvc_total_participation'] = (
        data_cleaned['gvc_backward_participation'].fillna(0) +
        data_cleaned['gvc_forward_participation'].fillna(0)
    )

total_exports = data[
    (data['REF_AREA'] == 'USA') & (data['MEASURE'] == 'EXGR')
].copy()

data_cleaned.to_csv('cleaned_group4.csv', index=False)