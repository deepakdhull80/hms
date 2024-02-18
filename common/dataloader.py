import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config import ConfigV1
from common.data import Datasetv1, InferDatasetv1
from common.data_v2 import Datasetv2, InferDatasetv2

def getDataLoader(config: ConfigV1, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> list[DataLoader, DataLoader]:
    spectrograms = np.load(os.path.join(config.data.data_prefix, 'spectrograms.npz'), allow_pickle=True)
    
    train_ds = Datasetv1(train_df, config) if config.version == 'v1' else Datasetv2(train_df, config, spectrograms, mode='train')
    val_ds = None
    if val_df is not None:
        val_ds = Datasetv1(val_df, config) if config.version == 'v1' else Datasetv2(val_df, config, spectrograms, mode='val')
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=config.trainer_config.batch_size,
        shuffle=True, 
        num_workers=config.trainer_config.num_workers,
        pin_memory=config.trainer_config.pin_memory,
        multiprocessing_context="fork"
    )
    
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(
            val_ds, 
            batch_size=config.trainer_config.batch_size,
            shuffle=True, 
            num_workers=config.trainer_config.num_workers,
            pin_memory=config.trainer_config.pin_memory,
            multiprocessing_context="fork"
        )
    
    return train_dl, val_dl


def getInferDataLoader(config: ConfigV1, infer_df: pd.DataFrame) -> DataLoader:
    infer_ds = InferDatasetv1(infer_df, config) if config.version == 'v1' else InferDatasetv2(infer_df, config)
    
    infer_dl = DataLoader(
        infer_ds, 
        batch_size=config.trainer_config.batch_size,
        num_workers=config.trainer_config.num_workers,
        pin_memory=config.trainer_config.pin_memory 
    )
    
    return infer_dl