import os.path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch import tensor, nn
from torch.utils.data import TensorDataset

from examples.ice_example.Loader import IceLoader
from torchcnnbuilder.models import ForecasterBase
from pytorch_msssim import ssim


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))



device = 'cuda'

df = pd.DataFrame()

combs = [(26, 52), (104, 52), (156, 52)]


for e in range(3):
    prehistory = combs[e][0]
    forecast_size = combs[e][1]

    folder = f'arctic(res3)_{prehistory}_{forecast_size}'
    '''images_folder = f'{folder}/images'
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)'''

    start = '20170501'
    end = '20180531'

    target_ts, target_dates = IceLoader('osisaf').load_sea('Arctic', (start, end))
    target_ts = target_ts[:forecast_size]
    target_dates = target_dates[:forecast_size]
    target_ts = resize(target_ts, (target_ts.shape[0], target_ts.shape[1]//3, target_ts.shape[2]//3))

    features_ts, features_dates = IceLoader('osisaf').load_sea('Arctic', ((datetime.strptime(start, '%Y%m%d')-relativedelta(weeks=prehistory)).strftime('%Y%m%d'),
                                                                          start
                                                                          ))
    features_ts = features_ts[-prehistory:]
    features_dates = features_dates[-prehistory:]
    features_ts = resize(features_ts, (features_ts.shape[0], features_ts.shape[1]//3, features_ts.shape[2]//3))
    test_dataset = TensorDataset(torch.tensor(np.expand_dims(features_ts, axis=0)).float(),
                                 torch.tensor(np.expand_dims(target_ts, axis=0)))
    dims = (features_ts.shape[1], features_ts.shape[2])
    print('Dataset loaded')

    model_path = f'{folder}/{dims}_arctic_4999.pt'

    model = ForecasterBase(input_size=dims,
                           in_time_points=prehistory,
                           out_time_points=forecast_size,
                           n_layers=5,
                           finish_activation_function=nn.ReLU())
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    model = model.to(device)

    s=0
    for features, target in test_dataset:
        s = s + 1
        features = features.to(device)
        target = target.numpy()
        prediction = model(features).cpu().detach().numpy()


        prediction[prediction<0] = 0
        prediction[prediction >1] = 1

        mae_per_frame = []
        for im in range(prediction.shape[0]):
            m = round(float(np.mean(abs(prediction[im] - target[im]))), 3)
            if m > 0.1:
                m = mae_per_frame[-1]
            mae_per_frame.append(m)
        df['dates'] = target_dates
        df[str(combs[e])] = mae_per_frame
        '''plt.plot(target_dates, mae_per_frame)
        plt.show()'''

        #### small im with steps
        fig, axs = plt.subplots(2, len(range(0, forecast_size, 8)), figsize=(11, 4.5))
        kk = 0

        sc_dates = []
        sc_n = []

        for k in range(0, forecast_size, 8):
            axs[0][kk].imshow(target[k], cmap='Greys_r')
            axs[0][kk].set_title(target_dates[k].strftime('%Y/%m/%d'))
            axs[1][kk].imshow(prediction[k], cmap='Greys_r')
            axs[1][kk].set_title(f'MAE={mae_per_frame[k]}')

            sc_dates.append(target_dates[k])
            sc_n.append(k)
    
            axs[0][kk].axes.xaxis.set_ticks([])
            axs[1][kk].axes.xaxis.set_ticks([])
            axs[0][kk].axes.yaxis.set_ticks([])
            axs[1][kk].axes.yaxis.set_ticks([])
            kk = kk + 1
        plt.tight_layout()
        #plt.suptitle()
        #plt.savefig(f'{images_folder}/{combs[e]}.png', dpi=300)
        plt.show()
        #########



plt.rcParams['figure.figsize'] = (6, 4)
plt.plot(df['dates'], df['(156, 52)'], label='input frames=156', c='black')
plt.plot(df['dates'], df['(104, 52)'], label='input frames=104', c='black', linestyle='-')
plt.plot(df['dates'], df['(26, 52)'], label='input frames=26', c='black', linestyle='--')
for s in range(len(sc_n)-2):
    plt.scatter(sc_dates[s], df['(26, 52)'][sc_n[s]], c='black', s=40)
    plt.scatter(sc_dates[s], df['(104, 52)'][sc_n[s]], c='black', s=40)
    plt.scatter(sc_dates[s], df['(156, 52)'][sc_n[s]], c='black', s=40)
plt.legend()
plt.ylabel('Mean Absolute Error')
plt.title('MAE dynamics by frames')
plt.tight_layout()
plt.show()