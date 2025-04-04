from datetime import datetime

import numpy as np
import pandas as pd


class IceLoader:
    def __init__(self, source: str, root: str = None):
        if root is None:
            self.root = 'ICE_DATA_MODULE'
        else:
            self.root = root
        self._init_dataset_name(source)
        self.source = source

    def _init_dataset_name(self, name: str):
        """
        Function to set path of files depending on source and its name format
        :param name: source of data name - 'masisaf', 'osisaf', 'masie', 'oras', 'mean'
        """
        names = {'masisaf': 'OSISAF_MASIE_HYBRID',
                 'osisaf': 'OSISAF',
                 'masie': 'MASIE',
                 'oras': 'ORAS',
                 'mean': 'meanyears5'
                 }
        formats = {'masisaf': 'osi_masie_%Y%m%d.npy',
                   'osisaf': 'osi_%Y%m%d.npy',
                   'masie': 'masie_%Y%m%d.npy',
                   'oras': 'oras_%Y%m.npy',
                   'mean': 'osi_%Y%m%d.npy'}
        self.filespath = f'{self.root}/{names[name]}'
        self.filesformat = formats[name]

    def _validate_period(self, period: tuple[str, str]):
        periods = {'masisaf': (datetime(2006, 1, 1), datetime(2024, 8, 31)),
                   'osisaf': (datetime(1979, 1, 1), datetime(2024, 8, 31)),
                   'masie': (datetime(2006, 1, 1), datetime(2024, 8, 31)),
                   'oras': (datetime(1990, 1, 1), datetime(2024, 5, 1)),
                   'mean': (datetime(1984, 1, 1), datetime(2024, 12, 31))}
        if (datetime.strptime(period[0], '%Y%m%d') < periods[self.source][0] or
                datetime.strptime(period[1], '%Y%m%d') > periods[self.source][1]):
            raise Exception(f'Period is invalid, for source {self.source} available periods {periods[self.source]}')

    def load_sea(self, sea_name: str, period: tuple[str, str], time_discret=7):
        """
        Function for loading in array files by sea and dates
        :param time_discret: step of discretization by time (in days)
        :param sea_name: 'kara', 'barents', 'laptev', 'eastsib', 'chukchi'
        :param period: tuple with start and end dates of period in format %Y%m%d
        :return: array with images
        """
        self._validate_period(period)
        print(f'Loading files period {period[0]} - {period[1]}')
        period = pd.date_range(period[0], period[1], freq=f'{time_discret}D')
        names = [t.strftime(self.filesformat) for t in period]
        array = []
        for file in names:
            matrix = np.load(f'{self.filespath}/{sea_name}/{file}')
            array.append(matrix)
        array = np.array(array)
        return array, period
