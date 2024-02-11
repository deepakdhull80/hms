import pandas as pd

from common.data import Datasetv1
from config.config import ConfigV1


if __name__ == "__main__":
    df = pd.read_csv(ConfigV1.data.data_prefix + "/" + ConfigV1.data.meta_file_name)
    ds = Datasetv1(df, ConfigV1)
    row = next(iter(ds))
    print(row)