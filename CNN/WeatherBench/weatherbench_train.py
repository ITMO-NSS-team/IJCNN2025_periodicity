import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

from BenchLoader import NpyLoader
from Metrics import WRMSE

device = 'cuda'


datasets_names = ['2m_temperature_5.625deg',
                  'geopotential_500_5.625deg',
                  'toa_incident_solar_radiation_5.625deg',
                  'total_cloud_cover_1.40625deg',
                  'total_precipitation_1.40625deg',
                  '10m_u_component_of_wind_5.625deg',
                  '10m_v_component_of_wind_5.625deg',
                  ]

combs = [(26, 52), (104, 52), (52, 2), (26, 2)]

for name in datasets_names:
    source_path = f'D:/WeatherBench/matrices/{name}'
    if len(os.listdir(source_path)) < 50:
        print(source_path)
        break
    if not os.path.exists(f'{name}'):
        os.makedirs(name)
    for comb in combs:
        folder = f'{name}/{comb}_{name}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        source_path = f'D:/WeatherBench/matrices/{name}'


        prehistory = comb[0]
        forecast_size = comb[1]

        norm_file = f'{folder}/norm_{name}.csv'
        dates, data = NpyLoader(source_path, log_file=norm_file).load_data(('20100101', '20151231'), step='7D',
                                                                           norm=True)
        if '1.40625deg' in name:
            data = resize(data, (data.shape[0], data.shape[1] // 4, data.shape[2] // 4))
        dataset = multi_output_tensor(data=data, forecast_len=forecast_size, pre_history_len=prehistory)

        dims = (data.shape[1], data.shape[2])
        print(f'Loaded dataset of shape {dataset}')

        batch_size = 100
        dataloader = DataLoader(dataset, batch_size=batch_size)

        model = ForecasterBase(input_size=[data.shape[1], data.shape[2]],
                               in_time_points=prehistory,
                               out_time_points=forecast_size,
                               n_layers=5,
                               finish_activation_function=nn.ReLU(),
                               n_transpose_layers=3
                               )

        print(model)
        model = model.to(device)


        optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
        criterion = WRMSE(torch.tensor(np.load('C:/Users/Julia/Documents/NSS_lab/cvpr_2025/grid_files/weights_5.625.npy')).to(device))

        epochs = 5000
        losses = []
        epoches = []
        for epoch in range(0, epochs):
            loss = 0
            v_loss = 0
            for train_features, train_targets in dataloader:
                train_features = train_features.to(device)
                train_targets = train_targets.to(device)
                optim.zero_grad()
                outputs = model(train_features)
                train_loss = criterion(outputs, train_targets)  # for squared error as in formula
                train_loss.backward()
                optim.step()
                loss += train_loss.item()

            loss = loss / len(dataloader)
            print(f'{epoch}/{epochs},loss={loss}')
            losses.append(loss)
            epoches.append(epoch)

            if epoch % 500 == 0:
                torch.save(model.state_dict(), f'{folder}/{name}_ep{epoch}.pt')
                torch.save(optim.state_dict(), f'{folder}/optim_{name}_ep{epoch}.pt')
                df = pd.DataFrame()
                df['epoch'] = epoches
                df['train_loss'] = losses
                df.to_csv(f'{folder}/{name}_ep{epoch}.csv', index=False)

        torch.save(model.state_dict(), f'{folder}/{name}_ep{epoch}.pt')
        df = pd.DataFrame()
        df['epoch'] = epoches
        df['train_loss'] = losses
        df.to_csv(f'{folder}/{name}_ep{epoch}.csv', index=False)

