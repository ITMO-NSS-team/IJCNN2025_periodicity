import os
from skimage.metrics import structural_similarity as ssim
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


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def get_anime_timeseries(rgb=False):
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
    return array


def get_cycled_data(cycles_num, is_rgb):
    array = get_anime_timeseries(rgb=is_rgb)
    arr = []
    for i in range(cycles_num):
        arr.append(array)
    arr = np.array(arr)
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
    return arr


df = pd.read_csv('anime_metrics.csv', delimiter='\t', decimal=',')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(df['Features size'], df['MSE'], 'g-')
ax2.plot(df['Features size'], df['Time, min'], 'b-')
ax1.set_xlabel('Prehistory size')
ax1.set_ylabel('MSE', color='g')
ax2.set_ylabel('Time, min', color='b')
plt.show()


device = 'cuda'

prehistory = 2
forecast = 8
folder = f'models_anime_{prehistory}_{forecast}'
model_path = f'{folder}/anime_90000.pt'
df = pd.read_csv(f'{folder}/anime_90000.csv')
plt.plot(df['epoch'], df['loss'])
plt.yscale('log')
plt.show()

data = get_cycled_data(2, False)

model = ForecasterBase(input_size=[data.shape[1], data.shape[2]],
                       in_time_points=prehistory,
                       out_time_points=forecast,
                       n_layers=5,
                       finish_activation_function=nn.ReLU(inplace=True))

model.load_state_dict(torch.load(model_path))
print(model)
model.to(device)

features = data[-prehistory:]
target = data[:forecast]

features_tensor = torch.tensor(np.expand_dims(features, axis=0)).float().to(device)
prediction = model(features_tensor).cpu().detach().numpy()[0]


fig, axs = plt.subplots(2, forecast+prehistory, figsize=(11, 3.4))

for i in range(prehistory):
    axs[0][i].imshow(features[i], cmap='Greys_r')

    axs[0][i].set_yticks([])
    axs[0][i].set_xticks([])
    axs[1][i].set_yticks([])
    axs[1][i].set_xticks([])

    axs[1][i].spines['top'].set_visible(False)
    axs[1][i].spines['right'].set_visible(False)
    axs[1][i].spines['bottom'].set_visible(False)
    axs[1][i].spines['left'].set_visible(False)


mae_list = []
mse_list = []
psnr_list = []
ssim_list = []

for i in range(forecast):
    axs[1][i+prehistory].imshow(prediction[i], cmap='Greys_r')
    axs[0][i+prehistory].imshow(target[i], cmap='Greys_r')

    axs[0][i+prehistory].set_yticks([])
    axs[0][i+prehistory].set_xticks([])
    axs[1][i+prehistory].set_yticks([])
    axs[1][i+prehistory].set_xticks([])

    mae_list.append(np.mean(abs(prediction[i]-target[i])))
    mse_list.append(np.mean((prediction[i] - target[i])**2))
    psnr_list.append(calculate_psnr(prediction[i], target[i]))
    ssim_list.append(ssim(prediction[i], target[i], data_range=1))

plt.tight_layout()

print(f'MAE={np.mean(mae_list)}')
print(f'MSE={np.mean(mse_list)}')
print(f'PSNR={np.mean(psnr_list)}')
print(f'SSIM={np.mean(ssim_list)}')


plt.show()





