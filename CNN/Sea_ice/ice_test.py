import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_msssim import ssim
from skimage.transform import resize
from torch import nn

from examples.ice_example.Loader import IceLoader
from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


test_ts, dates = IceLoader('osisaf').load_sea('Arctic', ('20130101', '20201231'))
test_ts = resize(test_ts, (test_ts.shape[0], test_ts.shape[1]//3, test_ts.shape[2]//3))

prehistory = 156
forecast_size = 52
ep = 4999

test_dataset = multi_output_tensor(data=test_ts, forecast_len=forecast_size, pre_history_len=prehistory)
dim = (test_ts.shape[1], test_ts.shape[2])

file = f'arctic_{prehistory}_{forecast_size}/{dim}_arctic_{ep}.pt'
df = pd.read_csv(f'arctic(res3)_{prehistory}_{forecast_size}/{dim}_arctic_{ep}.csv')
plt.plot(df['epoch'], df['train_loss'], label='train')
plt.plot(df['epoch'], df['val_losses'], label='validation')
plt.legend()
plt.yscale('log')
plt.show()

device = 'cuda'
print(f'Calculation on device: {device}')
model = ForecasterBase(input_size=dim,
                       in_time_points=prehistory,
                       out_time_points=forecast_size,
                       n_layers=5,
                       finish_activation_function=nn.ReLU())
model.load_state_dict(torch.load(file, weights_only=True))
model.eval()
model = model.to(device)

mae_list = []
mse_list = []
ssim_list = []
psnr_list = []


for features, target in test_dataset:
    features = features.to(device)
    target = target.numpy()
    prediction = model(features).cpu().detach().numpy()
    prediction[prediction < 0] = 0
    prediction[prediction > 1] = 1

    mae_v = round(np.mean(abs(prediction - target)).astype(float), 5)
    mse_v = round(np.mean((prediction - target)**2).astype(float), 5)
    ssim_v = round(ssim(torch.Tensor(np.expand_dims(prediction, axis=0)),
                        torch.Tensor(np.expand_dims(target, axis=0)), data_range=1).item(), 5)
    psnr_v = round(calculate_psnr(prediction, target), 5)

    mae_list.append(mae_v)
    mse_list.append(mse_v)
    ssim_list.append(ssim_v)
    psnr_list.append(psnr_v)


    #fig, axs = plt.subplots(3, forecast_size, figsize=(50, 5))
    fig, axs = plt.subplots(3, forecast_size, figsize=(5, 6))
    for i in range(forecast_size):
        axs[0][i].imshow(prediction[i], cmap='Greys_r')
        axs[1][i].imshow(target[i], cmap='Greys_r')
        axs[2][i].imshow(abs(prediction[i] - target[i]), cmap='Reds', vmin=0, vmax=1)

        axs[0][i].axes.xaxis.set_ticks([])
        axs[1][i].axes.xaxis.set_ticks([])
        axs[2][i].axes.xaxis.set_ticks([])
        axs[0][i].axes.yaxis.set_ticks([])
        axs[1][i].axes.yaxis.set_ticks([])
        axs[2][i].axes.yaxis.set_ticks([])

    plt.tight_layout()
    plt.suptitle(f'MAE={mae_v}, SSIM={ssim_v}, PSNR={psnr_v}')
    plt.show()

print(f'Mean MAE for test set = {np.mean(mae_list)}')
print(f'Mean MSE for test set = {np.mean(mse_list)}')
print(f'Mean SSIM for test set = {np.mean(ssim_list)}')
print(f'Mean PSNR for test set = {np.mean(psnr_list)}')