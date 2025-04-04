import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageSequence
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch import nn, optim
from torch.utils.data import DataLoader

from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess.time_series import multi_output_tensor


def get_anime_timeseries(rgb=True):
    with Image.open('../../media/anime.gif') as im:
        array = []
        for frame in ImageSequence.Iterator(im):
            if rgb:
                im_data = frame.copy().convert('RGB').getdata()
                im_array = np.array(im_data).reshape(frame.size[1], frame.size[0], 3)
            else:
                im_data = frame.copy().convert('L').getdata()
                im_array = np.array(im_data).reshape(frame.size[1], frame.size[0], 1)
            array.append(im_array[:, :, 0])
        array = np.array(array)
        array = array / 255
    array = resize(array, (array.shape[0], array.shape[1] // 5, array.shape[2] // 5))
    plt.imshow(array[0], cmap='Greys_r')
    plt.show()
    return array


def get_cycled_data(cycles_num, is_rgb):
    array = get_anime_timeseries(rgb=is_rgb)
    arr = []
    for i in range(cycles_num):
        arr.append(array)
    arr = np.array(arr)
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
    return arr

combs = [(2, 8), (4, 8), (8, 8), (16, 8)]
times = []
for comb in combs:
    prehistory = comb[0]
    forecast = comb[1]
    folder = f'models_anime_{prehistory}_{forecast}'
    if not os.path.exists(folder):
        os.makedirs(folder)

    data = get_cycled_data(15, False)
    train_dataset = multi_output_tensor(data=data,
                                        pre_history_len=prehistory,
                                        forecast_len=forecast)
    model = ForecasterBase(input_size=[data.shape[1], data.shape[2]],
                           in_time_points=prehistory,
                           out_time_points=forecast,
                           n_layers=5,
                           finish_activation_function=nn.ReLU(inplace=True))
    device = 'cuda'
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 90
    batch_size = 500

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    losses = []
    epochs_list = []

    start = time.time()

    for epoch in range(0, epochs):
        loss = 0

        for train_features, test_features in dataloader:
            train_features = train_features.to(device)
            test_features = test_features.to(device)

            optimizer.zero_grad()
            outputs = model(train_features)
            train_loss = criterion(outputs, test_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        loss = loss / len(dataloader)
        losses.append(loss)
        epochs_list.append(epoch)


        print(f'epoch {epoch}/{epochs}, loss={np.round(loss, 5)}')

        if epoch % 10000 == 0 or epoch == epochs-1:
            torch.save(model.state_dict(), f'{folder}/anime_{epoch}.pt')
            torch.save(optimizer.state_dict(), f'{folder}/optim_anime_{epoch}.pt')
            df = pd.DataFrame()
            df['epoch'] = epochs_list
            df['loss'] = losses
            df.to_csv(f'{folder}/anime_{epoch}.csv', index=False)

    end = time.time()
    times.append((end-start)/60)
    print(f'Finished {(end-start)/60} min')

print(combs)
print(times)