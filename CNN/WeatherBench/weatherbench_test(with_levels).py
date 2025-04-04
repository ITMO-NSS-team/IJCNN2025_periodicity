import os

import numpy as np
import pandas as pd
import torch
from pytorch_msssim import ssim
from torch import nn
from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

from BenchLoader import NpyLoader
from Metrics import WRMSE, wrmse, wmae

device = 'cuda'


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def reverse_normalization(matrix: np.ndarray, max_val, min_val):
    print(f'Reverse data normalization to max_value = {max_val}, min_value = {min_val}')
    matrix = matrix * (max_val - min_val) + min_val
    return matrix

# specify if no need to calc all
levels = [50, 100, 150,200,250,300,400,500,600,700,850,925]
datasets_names = [
    'relative_humidity_5.625deg',
    'temperature_5.625deg',
    'potential_vorticity_5.625deg',
    'specific_humidity_5.625deg'
]

combs = [(26, 52), (104, 52), (52, 2), (26, 2)]
lats_grid = np.load('../grid_files/weights_5.625.npy')

df = pd.DataFrame()
df['metric'] = ['mae', 'mse', 'mae_norm', 'mse_norm', 'ssim', 'psnr']

for name in datasets_names:
    source_path = f'D:/WeatherBench/matrices/{name}'
    for level in os.listdir(source_path):  #for level in os.listdir(levels):
        level_source_path = f'{source_path}/{level}'
        lev_folder = f'{name}_{level}'
        for comb in combs:
            folder = f'{lev_folder}/{comb}_{name}'
            print(folder)

            prehistory = comb[0]
            forecast_size = comb[1]

            dates, data = NpyLoader(level_source_path).load_data(('20150101', '20181231'), step='7D',
                                                                 norm=True)

            dataset = multi_output_tensor(data=data, forecast_len=forecast_size, pre_history_len=prehistory)

            norm_file = pd.read_csv(f'{folder}/norm_{name}.csv')
            min_val = norm_file['min'].values[0]
            max_val = norm_file['max'].values[0]

            dims = (data.shape[1], data.shape[2])
            print(f'Loaded dataset of shape {dataset}')

            model = ForecasterBase(input_size=[data.shape[1], data.shape[2]],
                                   in_time_points=prehistory,
                                   out_time_points=forecast_size,
                                   n_layers=5,
                                   finish_activation_function=nn.ReLU(),
                                   n_transpose_layers=3
                                   )
            print(model)
            model_path = f'{folder}/{name}_ep4999.pt'
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model = model.to(device)

            criterion = WRMSE(torch.tensor(lats_grid).to(device))

            mae_errors = []
            rmse_errors = []
            mae_norm_errors = []
            rmse_norm_errors = []
            psnr_errors = []
            ssim_errors = []
            for features, targets in dataset:
                features = features.to(device)
                outputs = model(features)
                targets = targets.to(device)

                wrmse_val = criterion(outputs, targets).item()
                ssim_val = ssim(outputs[None, :, :, :], targets[None, :, :, :], data_range=1).item()
                ssim_errors.append(ssim_val)

                prediction = outputs.cpu().detach().numpy()
                target = targets.cpu().detach().numpy()

                psnr_c = calculate_psnr(prediction, target)
                psnr_errors.append(psnr_c)
                print(f'PSNR={psnr_c}')

                wrmse_n = wrmse(prediction, target, lats_grid)
                rmse_norm_errors.append(wrmse_n)
                print(f'WRMSE={wrmse_n}')
                wmae_n = wmae(prediction, target, lats_grid)
                mae_norm_errors.append(wmae_n)
                print(f'WMAE={wmae_n}')

                # to calculate in absolute values
                prediction = reverse_normalization(prediction, max_val, min_val)
                target = reverse_normalization(target, max_val, min_val)

                wrmse_c = wrmse(prediction, target, lats_grid)
                rmse_errors.append(wrmse_c)
                print(f'WRMSE={wrmse_c}')
                wmae_c = wmae(prediction, target, lats_grid)
                mae_errors.append(wmae_c)
                print(f'WMAE={wmae_c}')

            df[f'{comb}_{name}_{level}'] = [np.mean(mae_errors),
                                            np.mean(rmse_errors),
                                            np.mean(mae_norm_errors),
                                            np.mean(rmse_norm_errors),
                                            np.mean(ssim_errors),
                                            np.mean(psnr_errors)]

        df.to_csv(f'level_metrics.csv', index=False)
