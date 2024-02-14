import pandas as pd
from torch.utils.data import DataLoader
from config.config import ConfigV1
from common.data import Datasetv1, InferDatasetv1

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


def getInferDataLoader(config: ConfigV1, infer_df: pd.DataFrame) -> DataLoader:
    infer_ds = InferDatasetv1(infer_df, config)
    
    infer_dl = DataLoader(
        infer_ds, 
        batch_size=config.trainer_config.batch_size,
        num_workers=config.trainer_config.num_workers,
        pin_memory=config.trainer_config.pin_memory 
    )
    
    return infer_dl