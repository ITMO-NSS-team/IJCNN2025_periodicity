import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.transform import resize
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from examples.ice_example.Loader import IceLoader
from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

ts, dates = IceLoader('osisaf').load_sea('Arctic', ('20000101', '20151231'))
val_ts, val_dates = IceLoader('osisaf').load_sea('Arctic', ('20130101', '20181231'))
ts = resize(ts, (ts.shape[0], ts.shape[1]//3, ts.shape[2]//3))
val_ts = resize(val_ts, (val_ts.shape[0], val_ts.shape[1]//3, val_ts.shape[2]//3))

device = 'cuda'
prehistory = 156
forecast_size = 52
folder = f'arctic_{prehistory}_{forecast_size}'
if not os.path.exists(folder):
    os.makedirs(folder)


train_dataset = multi_output_tensor(data=ts, forecast_len=forecast_size, pre_history_len=prehistory)
validation_dataset = multi_output_tensor(data=val_ts, forecast_len=forecast_size, pre_history_len=prehistory)
dim = (ts.shape[1], ts.shape[2])
print(len(train_dataset))

device = 'cuda'
print(f'Calculation on device: {device}')
model = ForecasterBase(input_size=dim,
                       in_time_points=prehistory,
                       out_time_points=forecast_size,
                       n_layers=5,
                       finish_activation_function=nn.ReLU())
print(model)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)


epochs = np.arange(0, 5000)
batch_size = 50
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()

losses = []
val_losses = []
epoches = []

start = time.time()

for epoch in epochs:
    loss = 0
    for train_features, train_targets in dataloader:
        train_features = train_features.to(device)
        train_targets = train_targets.to(device)
        optimizer.zero_grad()
        outputs = model(train_features)
        train_loss = criterion(outputs, train_targets)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    loss = loss / len(dataloader)

    val_loss_value = 0
    for v_train_features, v_train_targets in val_dataloader:
        v_train_features = v_train_features.to(device)
        v_train_targets = v_train_targets.to(device)
        optimizer.zero_grad()
        outputs = model(v_train_features)
        val_loss = criterion(outputs, v_train_targets)
        val_loss.backward()
        optimizer.step()
        val_loss_value += val_loss.item()
    val_loss_value = val_loss_value / len(val_dataloader)

    print(f'epoch {epoch}, loss={np.round(loss, 5)}, validation_loss={np.round(val_loss_value, 5)}')
    #print(f'epoch {epoch}, loss={np.round(loss, 5)}')

    losses.append(loss)
    val_losses.append(val_loss_value)
    epoches.append(epoch)
    if epoch % 100 == 0:
        torch.save(model.state_dict(), f'{folder}/{dim}_arctic_{epoch}.pt')
        torch.save(optimizer.state_dict(), f'{folder}/optim_{dim}_arctic_{epoch}.pt')
        df = pd.DataFrame()
        df['epoch'] = epoches
        df['train_loss'] = losses
        df['val_losses'] = val_losses
        df.to_csv(f'{folder}/{dim}_arctic_{epoch}.csv', index=False)
        plt.plot(epoches, losses, label='train')
        plt.plot(epoches, val_losses, label='validation')
        plt.legend()
        plt.yscale('log')
        plt.show()
end = time.time()
print(f'Finished {(end-start)/60} min')
with open(f'{folder}/{dim}_arctic_{epoch}.txt', 'w') as log:
    log.write(f'Finished {(end-start)/60} min')
torch.save(model.state_dict(), f'{folder}/{dim}_arctic_{epoch}.pt')