import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from config.config import ConfigV1

if len(sys.argv) > 1:
    ConfigV1.data.data_prefix = sys.argv[1]

folder = f"{ConfigV1.data.data_prefix}/train_spectrograms/"
paths = os.listdir(folder)

cache = {}
for path in tqdm(paths, total=len(paths)):
    _id = path.split("/")[-1].replace(".parquet", "")
    spec = pd.read_parquet(f"{folder}{path}")
    cache[_id] = spec.to_numpy()

np.savez(f"{ConfigV1.data.data_prefix}/spectrograms.npz", **cache)