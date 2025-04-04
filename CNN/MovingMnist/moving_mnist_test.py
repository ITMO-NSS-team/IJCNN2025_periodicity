import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim, tensor
from torch.utils.data import TensorDataset, DataLoader
from torchcnnbuilder.models import ForecasterBase

from pytorch_msssim import ssim
from skimage.transform import resize


def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


data = np.load('moving_mnist.npy').astype(np.float32)/255
test_set = data[:, 9000:, :, :]

prehistory = 5
forecast_size = 10
ep = 100000
dim = 45

test_set = resize(test_set, (test_set.shape[0], test_set.shape[1], dim, dim))

test_features = test_set[:prehistory, :, :, :]
test_features = np.swapaxes(test_features, 0, 1)
test_target = test_set[prehistory:prehistory+forecast_size, :, :, :]
test_target = np.swapaxes(test_target, 0, 1)
print('Data loaded')

device = 'cuda'
print(f'Calculation on device: {device}')
model = ForecasterBase(input_size=[dim, dim],
                       in_time_points=prehistory,
                       out_time_points=forecast_size,
                       n_layers=5,
                       finish_activation_function=nn.ReLU())


file = f'moving_mnist_{prehistory}_{forecast_size}/45_mnist_{ep}.pt'
df = pd.read_csv(f'moving_mnist_{prehistory}_{forecast_size}/45_mnist_{ep}.csv')
plt.plot(df['epoch'], df['train_loss'], label='train')
plt.plot(df['epoch'], df['val_losses'], label='validation')
plt.legend()
plt.show()

model.load_state_dict(torch.load(file, weights_only=True))
model.eval()
model = model.to(device)


l1_errors = []
ssim_errors = []
psnr_errors = []
mse_errors = []

for s in range(test_features.shape[0]):
    features = tensor(test_features[s]).to(device)
    prediction = model(features).detach().cpu().numpy()
    target = test_target[s]
    mae = np.mean(abs(prediction - target))
    l1_errors.append(mae)
    mse = np.mean((prediction - target)**2)
    mse_errors.append(mse)
    ssim_errors.append(ssim(torch.Tensor(np.expand_dims(prediction, axis=0)), torch.Tensor(np.expand_dims(target, axis=0)), data_range=1))
    psnr_errors.append(round(calculate_psnr(prediction, target), 5))
print(f'Mean MAE for test set = {np.mean(l1_errors)}')
print(f'Mean MSE for test set = {np.mean(mse_errors)}')
print(f'Mean SSIM for test set = {np.mean(ssim_errors)}')
print(f'Mean PSNR for test set = {np.mean(psnr_errors)}')

for s in range(5):
    tensor_features = tensor(test_features[s]).to(device)
    prediction = model(tensor_features).detach().cpu().numpy()

    mae_v = round(np.mean(abs(prediction - test_target[s])).astype(float), 5)
    ssim_v = round(ssim(torch.Tensor(np.expand_dims(prediction, axis=0)), torch.Tensor(np.expand_dims(test_target[s], axis=0)), data_range=1).item(), 5)
    psnr_v = round(calculate_psnr(prediction, test_target[s]), 5)

    #plt.rcParams["figure.figsize"] = (12, 4)
    fig, axs = plt.subplots(3, forecast_size)
    for i in range(forecast_size):
        axs[0][i].imshow(prediction[i], cmap='Greys_r')
        axs[1][i].imshow(test_target[s][i], cmap='Greys_r')
        axs[2][i].imshow(abs(prediction[i] - test_target[s][i]), cmap='Reds')

        axs[0][i].axes.xaxis.set_ticks([])
        axs[1][i].axes.xaxis.set_ticks([])
        axs[2][i].axes.xaxis.set_ticks([])
        axs[0][i].axes.yaxis.set_ticks([])
        axs[1][i].axes.yaxis.set_ticks([])
        axs[2][i].axes.yaxis.set_ticks([])

    plt.tight_layout()
    plt.suptitle(f'Sample {s}\nMAE={mae_v}, SSIM={ssim_v}, PSNR={psnr_v}')
    plt.show()