import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms

from config import ConfigV1

class Datasetv1(data.Dataset):
    """### Datasetv1 dataset for only kaggle spectrogram
    
    """
    def __init__(self, df:pd.DataFrame, config: ConfigV1) -> None:
        super().__init__()
        self.config = config
        self.df = df
        # self.cache = dict()
        self._transform = transforms.Compose([
            transforms.Resize(size=self.config.data.image_size),
        ])
    
    def transform(self, spec):
        #TODO scaling spectrogram require
        return self._transform(spec)

    def scales_spectrogram(self, spectrogram):
        _spec = np.clip(spectrogram, np.exp(-4), np.exp(8))
        return np.log(_spec)
    
    def prepare_x(self, row):
        spectrogram_id = row['spectrogram_id']
        spectrogram_label_offset_seconds = int(row['spectrogram_label_offset_seconds'])
        
        # optimize this block store in cache
        # if spectrogram_id in self.cache:
        #     spec_df = self.cache[spectrogram_id]
        # else:
        
        spec_df = pd.read_parquet(f"{self.config.data.data_prefix}/{self.config.data.kaggle_spec_folder}/{spectrogram_id}.parquet")
        spec_df.fillna(-1, inplace=True)
        
        spec = spec_df.loc[(spec_df.time>=spectrogram_label_offset_seconds)
                     &(spec_df.time<spectrogram_label_offset_seconds+600)].iloc[:, 1:].T.values
        
        spec = self.scales_spectrogram(spec)
        
        # there are stack 4 spectrogram in single one
        kaggle_spec_n_bins = 100
        bins, time = spec.shape
        spec = spec.reshape(bins//kaggle_spec_n_bins, kaggle_spec_n_bins, time)
        
        return torch.from_numpy(spec)
    
    def prepare_y(self, row):
        if self.config.inference:
            return torch.randn(len(self.config.class_columns))
        labels = torch.from_numpy(row[self.config.class_columns].values.astype(np.int32))
        try:
            return labels/labels.sum()
        except ZeroDivisionError:
            return labels/(labels.sum() + 1e-6)

    def __getitem__(self,index):
        row = self.df.iloc[index]
        x: torch.Tensor = self.prepare_x(row)
        y: torch.Tensor = self.prepare_y(row)
        return x, y

    def __len__(self):
        return self.df.shape[0]

class InferDatasetv1(data.Dataset):
    """### Datasetv1 dataset for only kaggle spectrogram
    
    """
    def __init__(self, df:pd.DataFrame, config: ConfigV1) -> None:
        super().__init__()
        self.config = config
        self.df = df
        # self.cache = dict()
        self._transform = transforms.Compose([
            transforms.Resize(size=self.config.data.image_size),
        ])
    
    def transform(self, spec):
        #TODO scaling spectrogram require
        return self._transform(spec)

    def scales_spectrogram(self, spectrogram):
        _spec = np.clip(spectrogram, np.exp(-4), np.exp(8))
        return np.log(_spec)
    
    def prepare_x(self, row):
        spectrogram_id = row['spectrogram_id']
        spectrogram_label_offset_seconds = 0
        
        # optimize this block store in cache
        # if spectrogram_id in self.cache:
        #     spec_df = self.cache[spectrogram_id]
        # else:
        
        spec_df = pd.read_parquet(f"{self.config.data.data_prefix}/{self.config.data.kaggle_test_spec_folder}/{spectrogram_id}.parquet")
        spec_df.fillna(-1, inplace=True)
        
        spec = spec_df.loc[(spec_df.time>=spectrogram_label_offset_seconds)
                     &(spec_df.time<spectrogram_label_offset_seconds+600)].iloc[:, 1:].T.values
        
        spec = self.scales_spectrogram(spec)
        
        # there are stack 4 spectrogram in single one
        kaggle_spec_n_bins = 100
        bins, time = spec.shape
        spec = spec.reshape(bins//kaggle_spec_n_bins, kaggle_spec_n_bins, time)
        
        return torch.from_numpy(spec)

    def __getitem__(self,index):
        row = self.df.iloc[index]
        x: torch.Tensor = self.prepare_x(row)
        return x, torch.randn(1)

    def __len__(self):
        return self.df.shape[0]