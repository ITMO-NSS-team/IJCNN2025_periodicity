import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from pytorch_msssim import ssim
from skimage.transform import resize
from torch import nn, optim
from torch.utils.data import DataLoader

from examples.ice_example.Loader import IceLoader
from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

def calculate_psnr(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


ts, dates = IceLoader('osisaf').load_sea('Arctic', ('20160101', '20201231'))
ts = resize(ts, (ts.shape[0], ts.shape[1]//3, ts.shape[2]//3))

prediction = []

for date in dates:
    print(f'Calc {date}')
    dates_to_mean = [date-relativedelta(years=y) for y in range(1, 6)]
    matrices_to_mean = []
    for d in dates_to_mean:
        matrix, _ = IceLoader('osisaf').load_sea('Arctic', (d.strftime('%Y%m%d'), d.strftime('%Y%m%d')))
        matrix = resize(matrix, (matrix.shape[0], matrix.shape[1] // 3, matrix.shape[2] // 3))[0]
        matrices_to_mean.append(matrix)
    prediction.append(np.mean(matrices_to_mean, axis=0))
prediction = np.array(prediction)

mae_ts = abs(prediction-ts)
mae = np.mean(multi_output_tensor(mae_ts, 52, 52).tensors[1].numpy())
mse_ts = (prediction-ts)**2
mse = np.mean(multi_output_tensor(mse_ts, 52, 52).tensors[1].numpy())
psnr_ts = []
for i in range(ts.shape[0]):
    psnr_ts.append(calculate_psnr(prediction[i], ts[i]).astype(float))
psnr = np.mean(multi_output_tensor(psnr_ts, 52, 52).tensors[1].numpy())

print(f'Mean MAE for test set = {mae}')
print(f'Mean MSE for test set = {np.mean(mse)}')
print(f'Mean PSNR for test set = {np.mean(psnr)}')

ssim_errors = []
for i in range(ts.shape[0]):
    ssim_errors.append(ssim(torch.Tensor(np.expand_dims(np.expand_dims(prediction[i], axis=0), axis=0)),
                            torch.Tensor(np.expand_dims(np.expand_dims(ts[i], axis=0), axis=0)),
         data_range=1).item())

ssim_v = np.mean(multi_output_tensor(ssim_errors, 52, 52).tensors[1].numpy())

print(f'Mean SSIM for test set = {ssim_v}')