import torch
class DataConfig:
    data_prefix: str = "/Users/deepak.dhull/practice/kaggle/HMS/data"
    kaggle_spec_folder: str = "train_spectrograms"
    eeg_folder: str = "train_eegs"
    meta_file_name: str = "train.csv"
    image_size: tuple = (512, 512)

class ModelConfig:
    conv_in_channels: int = 4
    model_clazz: str = "models.backbone.EfficientNet_S"

class TrainerConfig:
    batch_size: int = 32
    num_workers: int = 1
    pin_memory: bool = False
    optim_clazz: str = "torch.optim.Adam"
    lr: float = 0.001
    epoch: int = 4
    step_perc_v: int = 10
    tqdm: bool = True
    train_size: float = 0.8
    

class ConfigV1:
    data = DataConfig
    model = ModelConfig
    trainer_config = TrainerConfig
    device: torch.device = torch.device("cpu")
    cache: int = 20
    inference: bool = False
    class_columns: list = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']