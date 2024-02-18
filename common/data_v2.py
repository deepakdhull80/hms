import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms

from config import ConfigV1

class Datasetv2(data.Dataset):
    """### Datasetv2 dataset for kaggle spectrogram and the eeg data
    
    """
    def __init__(self, df:pd.DataFrame, config: ConfigV1, spectrograms, mode="train") -> None:
        super().__init__()
        self.config = config
        self.df = df
        self.spectograms = spectrograms
        
        self._transform = transforms.Compose([
            transforms.Resize(size=self.config.data.image_size),
        ])
        self.mode = mode
    
    def transform(self, spec):
        return self._transform(spec)

    def scales_spectrogram(self, spectrogram):
        _spec = np.clip(spectrogram, np.exp(-4), np.exp(8))
        return np.log(_spec)
    
    def prepare_x(self, row):
        spectrogram_id = row['spec_id']
        # spectrogram_label_offset_seconds = int(row['spectrogram_label_offset_seconds'])
        spec: np.ndarray = self.spectograms[str(spectrogram_id)]
        if self.mode=='train':
            # RANDOM CROPS FOR TRAIN
            r = np.random.randint(row['min'], row['max']+1)//2
        else:
            r = int( (row['min'] + row['max'])//4 )
        
        spec = spec[r:r+300,1:].T
        
        spec = spec.reshape(4, -1, 300)
        spec = self.scales_spectrogram(spec)
        
        #Scale spectrogram
        ep = 1e-6
        m = np.nanmean(spec.reshape(4, -1), axis=1).reshape(4, 1, 1)
        s = np.nanstd(spec.reshape(4, -1), axis=1).reshape(4, 1, 1)
        spec = (spec-m)/(s+ep)
        spec = np.nan_to_num(spec, nan=0.0)
        
        ## CROP TO 256 TIME STEPS
        spec = spec[:, :, 22:-22]
        
        spec = np.pad(spec, ((0,0), (14, 14), (0,0)), constant_values=0)
        spec = torch.from_numpy(spec)
        k, bins, t = spec.shape
        spec = spec.view(1, -1, t)
        spec = torch.concat([spec, spec, spec], dim=0)
        return spec
    
    def prepare_y(self, row):
        if self.config.inference:
            return torch.randn(len(self.config.class_columns))
        # kl_divergence
        labels = torch.from_numpy(row[self.config.class_columns].values.astype(np.int32))
        
        # cross-entropy
        labels = torch.tensor(self.config.class_dict[row['target'].lower()]).long()
        
        try:
            return labels/labels.sum() if len(labels.size()) >= 1 else labels
        except ZeroDivisionError:
            return labels/(labels.sum() + 1e-6)

    def __getitem__(self,index):
        row = self.df.iloc[index]
        x: torch.Tensor = self.prepare_x(row)
        y: torch.Tensor = self.prepare_y(row)
        return x.float(), y

    def __len__(self):
        return self.df.shape[0]

class InferDatasetv2(data.Dataset):
    """### Datasetv2 dataset for kaggle spectrogram and the eeg data
    
    """
    def __init__(self, df:pd.DataFrame, config: ConfigV1) -> None:
        super().__init__()
        self.config = config
        self.df = df
        self.spectograms = np.load(os.path.join(self.config.data.data_prefix, 'spectrograms.npz'), allow_pickle=True)
    
    def scales_spectrogram(self, spectrogram):
        _spec = np.clip(spectrogram, np.exp(-4), np.exp(8))
        return np.log(_spec)
    
    def prepare_x(self, row):
        spectrogram_id = row['spec_id']
        # spectrogram_label_offset_seconds = int(row['spectrogram_label_offset_seconds'])
        spec: np.ndarray = self.spectograms[spectrogram_id]
        r = 0# inference
        spec = spec[r:r+300,1:].T
        
        spec = spec.reshape(4, -1, 300)
        spec = self.scales_spectrogram(spec)
        
        #Scale spectrogram
        ep = 1e-6
        m = np.nanmean(spec.reshape(4, -1), axis=1).reshape(4, 1, 1)
        s = np.nanstd(spec.reshape(4, -1), axis=1).reshape(4, 1, 1)
        spec = (spec-m)/(s+ep)
        spec = np.nan_to_num(spec, nan=0.0)
        
        ## CROP TO 256 TIME STEPS
        spec = spec[:, :, 22:-22]
        
        spec = np.pad(spec, ((0,0), (14, 14), (0,0)), constant_values=0)
        return torch.from_numpy(spec)
    
    def prepare_y(self, row):
        if self.config.inference:
            return torch.randn(len(self.config.class_columns))
        # kl_divergence
        labels = torch.from_numpy(row[self.config.class_columns].values.astype(np.int32))
        
        # cross-entropy
        labels = torch.tensor(self.config.class_dict[row['target'].lower()]).long()
        
        try:
            return labels/labels.sum() if len(labels.size()) >= 1 else labels
        except ZeroDivisionError:
            return labels/(labels.sum() + 1e-6)

    def __getitem__(self,index):
        row = self.df.iloc[index]
        x: torch.Tensor = self.prepare_x(row)
        y: torch.Tensor = self.prepare_y(row)
        return x, y

    def __len__(self):
        return self.df.shape[0]