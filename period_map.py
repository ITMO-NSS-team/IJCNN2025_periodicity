import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from CNN.Sea_ice.Loader import IceLoader


def extract_period(timeseries):
    timeseries = timeseries - np.mean(timeseries)
    num_points = len(timeseries)
    T = (1 / np.argsort(np.abs(np.fft.fft(timeseries))[:num_points // 2]) * num_points)
    T = np.round(T[-1])
    return T


def get_arctic_period_map():
    ts, dates = IceLoader('osisaf').load_sea('Arctic', ('19900101', '20151231'))
    ts = resize(ts, (ts.shape[0], ts.shape[1] // 3, ts.shape[2] // 3))
    plt.rcParams['figure.figsize'] = (6, 3)
    plt.plot(ts[:, 80, 100][-400:-200])
    plt.show()

    template = np.zeros((ts.shape[1], ts.shape[2]))
    for i in range(ts.shape[1]):
        print(f'{i}/{ts.shape[1]}')
        for j in range(ts.shape[2]):
            print(f'{j}/{ts.shape[2]}')
            t = ts[:, i, j]
            t_period = extract_period(t)
            template[i, j] = t_period


    values, counts = np.unique(template, return_counts=True)
    ind = np.argmax(counts)
    print(values[ind])

    template[template == np.inf] = np.nan
    plt.rcParams['figure.figsize'] = (6, 6)
    plt.imshow(template, cmap='Reds')
    plt.colorbar()
    plt.show()


def media_period_map(plot_ts=False):
    matrix = np.load('media/leg.npy')
    if plot_ts:
        plt.rcParams['figure.figsize'] = (6, 3)
        plt.plot(matrix[:, 35, 40])
        plt.savefig('media/ts3.png', dpi=400)
        plt.show()
        plt.rcParams['figure.figsize'] = (6, 3)
        plt.plot(matrix[:, 30, 40])
        plt.savefig('media/ts2.png', dpi=400)
        plt.show()
        plt.rcParams['figure.figsize'] = (6, 3)
        plt.plot(matrix[:, 45, 40])
        plt.savefig('media/ts1.png', dpi=400)
        plt.show()

    template = np.zeros((matrix.shape[1], matrix.shape[2]))
    for i in range(matrix.shape[1]):
        print(f'{i}/{matrix.shape[1]}')
        for j in range(matrix.shape[2]):
            print(f'{j}/{matrix.shape[2]}')
            t = matrix[:, i, j]
            t_period = extract_period(t)
            template[i, j] = t_period

    values, counts = np.unique(template, return_counts=True)
    ind = np.argmax(counts)
    print(values[ind])

    template[template == np.inf] = np.nan
    template[template == matrix.shape[0]] = np.nan
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.imshow(template, cmap='Reds')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.savefig('media/map.png', dpi=400)
    plt.show()

