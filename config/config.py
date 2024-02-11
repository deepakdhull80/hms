from abc import ABC

class DataConfig:
    data_prefix: str = "/Users/deepak.dhull/practice/kaggle/HMS/data"
    kaggle_spec_folder: str = "train_spectrograms"
    eeg_folder: str = "train_eegs"
    meta_file_name: str = "train.csv"


class ConfigV1:
    data = DataConfig
    cache: int = 20
    inference: bool = False
    class_columns: list = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

