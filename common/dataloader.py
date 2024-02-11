import pandas as pd
from torch.utils.data import DataLoader
from config.config import ConfigV1
from common.data import Datasetv1

def getDataLoader(config: ConfigV1, train_df: pd.DataFrame, val_df: pd.DataFrame) -> list[DataLoader, DataLoader]:
    train_ds = Datasetv1(train_df, config)
    val_ds = Datasetv1(val_df, config)
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=config.trainer_config.batch_size,
        shuffle=True, 
        num_workers=config.trainer_config.num_workers,
        pin_memory=config.trainer_config.pin_memory 
    )
    
    val_dl = DataLoader(
        val_ds, 
        batch_size=config.trainer_config.batch_size,
        shuffle=True, 
        num_workers=config.trainer_config.num_workers,
        pin_memory=config.trainer_config.pin_memory 
    )
    
    return train_dl, val_dl