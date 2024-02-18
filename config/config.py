import torch
class DataConfig:
    data_prefix: str = "/Users/deepak.dhull/practice/kaggle/HMS/data"
    kaggle_spec_folder: str = "train_spectrograms"
    kaggle_test_spec_folder: str = "test_spectrograms"
    eeg_folder: str = "train_eegs"
    eeg_test_folder: str = "test_eegs"
    meta_file_name: str = "train.csv"
    test_meta_file_name: str = "test.csv"
    image_size: tuple = (512, 512)

class ModelConfig:
    conv_in_channels: int = 3
    model_clazz: str = "models.backbone.EfficientNet_M"
    load_pretrained_weights: bool = True

class TrainerConfig:
    batch_size: int = 64
    num_workers: int = 1
    pin_memory: bool = False
    optim_clazz: str = "torch.optim.Adam"
    lr: float = 0.001
    precision: torch.dtype = torch.float16
    epoch: int = 5
    step_perc_v: int = 5
    tqdm: bool = True
    train_size: float = 0.8
    

class ConfigV1:
    version: str = "v2"
    k_folds: int = 5
    random_state_seed: int = 1997
    data = DataConfig
    model = ModelConfig
    trainer_config = TrainerConfig
    device: torch.device = torch.device("cpu")
    cache: int = 20
    inference: bool = False
    class_columns: list = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    class_dict: dict = {'seizure':0, 'lpd':1, 'gpd':2, 'lrda':3, 'grda':4, 'other':5}
    submission_export_path: str = "data"
    model_state_dict_path: str = "checkpoints/model.ckpt"
