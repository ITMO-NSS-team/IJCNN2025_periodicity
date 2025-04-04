import numpy as np
import pandas as pd
import torch
from skimage.transform import resize
from torchcnnbuilder.models import ForecasterBase
from pytorch_msssim import ssim

from BenchLoader import NpyLoader
from Metrics import WRMSE, wrmse, wmae
from torchcnnbuilder.preprocess.time_series import multi_output_tensor
from Diffusion.Diffusor import Diffusor

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


datasets_names = ['2m_temperature_5.625deg',
                  'geopotential_500_5.625deg',
                  'toa_incident_solar_radiation_5.625deg',
                  'total_cloud_cover_1.40625deg',
                  'total_precipitation_1.40625deg',
                  '10m_u_component_of_wind_5.625deg',
                  '10m_v_component_of_wind_5.625deg',
                  ]

combs = [(26, 52), (104, 52), (52, 2), (26, 2)]
lats_grid = np.load('grid_files/weights_5.625.npy')

df = pd.DataFrame()
df['metric'] = ['mae', 'mse', 'mae_norm', 'mse_norm', 'ssim', 'psnr']

for name in datasets_names:
    source_path = f'D:/WeatherBench/matrices/{name}'
    for comb in combs:
        folder = f'{name}/{comb}_{name}'
        denoise_path = f'{name}/denoise/{name}_mse(uniq500)_16000.pt'
        print(folder)

        prehistory = comb[0]
        forecast_size = comb[1]
        emb_num = prehistory

        dates, data = NpyLoader(source_path).load_data(('20150101', '20181231'), step='7D',
                                                       norm=True)
        if '1.40625deg' in name:
            data = resize(data, (data.shape[0], data.shape[1] // 4, data.shape[2] // 4))
        dataset = multi_output_tensor(data=data, forecast_len=forecast_size, pre_history_len=prehistory)

        norm_file = pd.read_csv(f'{folder}/norm_{name}.csv')
        min_val = norm_file['min'].values[0]
        max_val = norm_file['max'].values[0]

        dims = (data.shape[1], data.shape[2])
        print(f'Loaded dataset of shape {dataset}')

        model = ForecasterBase(input_size=dims,
                               n_layers=5,
                               in_time_points=forecast_size + 1 + emb_num,  # x_t+t+latent add features
                               out_time_points=forecast_size)
        ep = 3000
        if prehistory == 104:
            ep = 2000

        model_path = f'{folder}/{name}_{prehistory}_{forecast_size}_1000_emb{prehistory}_{ep}.pt'
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print('Model loaded')
        model.to(device)

        T = 100
        diffusion_steps = np.linspace(1, 0, T)
        diffusor = Diffusor(T)

        clear_model = ForecasterBase(input_size=dims,
                                     in_time_points=1,
                                     out_time_points=1,
                                     n_layers=5)
        clear_model.load_state_dict(
            torch.load(denoise_path, weights_only=True))
        clear_model.eval()

        criterion = WRMSE(torch.tensor(lats_grid).to(device))

        mae_errors = []
        rmse_errors = []
        mae_norm_errors = []
        rmse_norm_errors = []
        psnr_errors = []
        ssim_errors = []

        for features, target in dataset:
            x_t = np.random.normal(0, 0.1, forecast_size * dims[0] * dims[1]).reshape((forecast_size, dims[0], dims[1]))
            emb_features = features.numpy()

            n = 0
            for i in diffusion_steps:
                print(f'{n}/{T} step of back diffusion')
                t_sample = np.expand_dims(np.full(dims, i), axis=0)
                xt_and_t = np.concatenate((x_t, t_sample, emb_features), axis=0)
                t_noise = model(torch.Tensor(xt_and_t).to(device)).cpu().detach().numpy()
                x_t = diffusor.denoise_image(x_t, t_noise)
                n = n + 1

            deniosed_pred = []
            for c in range(forecast_size):
                x_t_deniose = clear_model(torch.Tensor(np.expand_dims(x_t[c], axis=0))).detach().numpy()[0]
                deniosed_pred.append(x_t_deniose)

            deniosed_pred = np.array(deniosed_pred)
            deniosed_pred[deniosed_pred < 0] = 0
            deniosed_pred[deniosed_pred > 1] = 1

            outputs = torch.tensor(deniosed_pred).to(device)
            targets = target.to(device)

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

        df[f'{comb}_{name}'] = [np.mean(mae_errors),
                                np.mean(rmse_errors),
                                np.mean(mae_norm_errors),
                                np.mean(rmse_norm_errors),
                                np.mean(ssim_errors),
                                np.mean(psnr_errors)]

        df.to_csv('metrics.csv', index=False)
