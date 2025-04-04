import os

import pandas as pd

datasets_names = [
                  'relative_humidity_5.625deg',
                  'temperature_5.625deg',
                  'potential_vorticity_5.625deg',
                  'specific_humidity_5.625deg'
                  ]
combs = [(26, 52), (104, 52), (52, 2), (26, 2)]

mean_df = pd.DataFrame()
for name in datasets_names:
    df = pd.read_csv(f'level_metrics.csv')
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    names = df.index.values
    for comb in combs:
        filtered_names = [n for n in names if name in n]
        filtered_names = [n for n in filtered_names if str(comb) in n]
        sub_df = df.loc[filtered_names]
        mean_df[f'{name}_{comb}'] = sub_df.mean().tolist()
mean_df = mean_df.T
mean_df.columns = ['mae', 'mse', 'mae_norm', 'mse_norm', 'ssim', 'psnr']
mean_df.to_csv('mean_level_metrics.csv')
