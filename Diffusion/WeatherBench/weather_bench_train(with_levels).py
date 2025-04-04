import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchcnnbuilder.models import ForecasterBase

from BenchLoader import NpyLoader
from Metrics import WRMSE, wrmse, wmae
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

from Diffusion.Diffusor import Diffusor

device = 'cuda'


def get_train_dataset(train_set):
    noise_level = 1.2
    noise = np.random.normal(0, 0.1, list(train_set.shape))
    noised_dataset = train_set + noise_level * noise
    noised_dataset = np.expand_dims(noised_dataset, axis=1)
    train_set = np.expand_dims(train_set, axis=1)
    tensor_dataset = TensorDataset(torch.Tensor(noised_dataset.astype(float)), torch.Tensor(train_set))
    return tensor_dataset

datasets_names = [
                  'temperature_5.625deg',
                  'potential_vorticity_5.625deg',
                  'specific_humidity_5.625deg'
                  ]

combs = [(26, 52), (104, 52), (52, 2), (26, 2)]

for name in datasets_names:
    source_path = f'D:/WeatherBench/matrices/{name}'
    for level in os.listdir(source_path):
        level_source_path = f'{source_path}/{level}'
        lev_folder = f'{name}_{level}'
        if not os.path.exists(lev_folder):
            os.makedirs(lev_folder)

            for comb in combs:
                folder = f'{lev_folder}/{comb}_{name}'
                if not os.path.exists(folder):
                    os.makedirs(folder)

                prehistory = comb[0]
                forecast_size = comb[1]

                norm_file = f'{folder}/norm_{name}.csv'
                dates, data = NpyLoader(level_source_path, log_file=norm_file).load_data(('20100101', '20151231'), step='7D',
                                                                                   norm=True)
                if '1.40625deg' in name:
                    data = resize(data, (data.shape[0], data.shape[1] // 4, data.shape[2] // 4))
                dataset = multi_output_tensor(data=data, forecast_len=forecast_size, pre_history_len=prehistory)

                dims = (data.shape[1], data.shape[2])
                print(f'Loaded dataset of shape {dataset}')

                batch_size = 100
                dataloader = DataLoader(dataset, batch_size=batch_size)

                emb_num = prehistory
                model = ForecasterBase(input_size=dims,
                                       n_layers=5,
                                       in_time_points=forecast_size + 1 + emb_num,  # x_t+t+latent add features
                                       out_time_points=forecast_size)

                print(model)
                model.to(device)

                T = 100  #число шагов диффузии
                diffusor = Diffusor(T)

                optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
                criterion = WRMSE(torch.tensor(np.load('/grid_files/weights_5.625.npy')).to(device))

                losses = []
                i = 0
                epochs = 1000

                for ep in range(epochs):
                    for features, target in dataloader:
                        target = target.numpy()
                        emb_features = features.numpy()

                        x_t, t, noise = diffusor.noise_images(target)
                        t_sample = np.expand_dims(np.array([np.full(dims, ti) for ti in t]), axis=1)
                        t_sample = np.concatenate((x_t, t_sample, emb_features), axis=1)
                        pred_noise = model(torch.Tensor(t_sample).float().to(device))
                        loss = criterion(torch.Tensor(noise).to(device), pred_noise)
                        losses.append(loss.item())
                        print(f' {i}/{len(dataloader) * epochs} loss = {loss.item()}')
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        i = i + 1
                        if i % 1000 == 0:
                            torch.save(model.state_dict(),
                                       f'{folder}/{name}_{prehistory}_{forecast_size}_{epochs}_emb{emb_num}_{i}.pt')

                torch.save(model.state_dict(), f'{folder}/{name}_{prehistory}_{forecast_size}_{epochs}_emb{emb_num}_{i}.pt')

                plt.title(f'Convergence curve\nbest loss={round(losses[-1], 5)}')
                plt.plot(np.arange(len(losses)), losses)
                plt.ylabel('MSE loss')
                plt.yscale('log')
                plt.xlabel('Sample number (epoch equivalent)')
                plt.show()

            # DENOISE PART
            if not os.path.exists(f'{name}/denoise'):
                os.makedirs(f'{name}/denoise')
            model = ForecasterBase(input_size=dims,
                                   in_time_points=1,
                                   out_time_points=1,
                                   n_layers=5)
            print(model)
            model.to(device)
            optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
            metric = nn.MSELoss()
            batch_size = 10
            losses = []
            i = 0

            uniq_noise_samp = 500
            for u in range(uniq_noise_samp):
                dataset = get_train_dataset(data)
                dataloader = DataLoader(dataset, batch_size=batch_size)
                for features, target in dataloader:
                    pred_image = model(features.to(device))
                    loss = metric(target.to(device), pred_image)
                    losses.append(loss.item())
                    print(f' {i}/{len(dataloader) * uniq_noise_samp} loss = {loss.item()}')
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    i = i + 1
                    if i % 5000 == 0:
                        torch.save(model.state_dict(),
                                   f'{name}/denoise/{name}_mse(uniq{uniq_noise_samp})_{i}.pt')
            torch.save(model.state_dict(), f'{name}/denoise/{name}_mse(uniq{uniq_noise_samp})_{i}.pt')
            plt.title(f'Convergence curve\nbest loss={round(losses[-1], 5)}')
            plt.plot(np.arange(len(losses)), losses)
            plt.ylabel('MSE loss')
            plt.yscale('log')
            plt.xlabel('Sample number (epoch equivalent)')
            plt.tight_layout()
            plt.show()
