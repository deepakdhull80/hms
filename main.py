import os
import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import ConfigV1
from models.base_model import Classifier
from common.dataloader import getDataLoader
from common.trainer import per_epoch

config = ConfigV1

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch", default=4)
    parser.add_argument("--tqdm", type=bool, default=True)
    return parser.parse_args()

def training_env(train_df, val_df, env_no=1):
    dirs = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(dirs, "runs", "checkpoints", f"env_{env_no}")
    os.makedirs(checkpoint_dir, mode=777, exist_ok=True)
    
    train_dl, val_dl = getDataLoader(config, train_df, val_df)
    model = Classifier(config)
    model.to(config.device)
    optimizer = model.get_optim()
    global_loss = 1e3
    for epoch in range(config.trainer_config.epoch):
        print("################################")
        print("Epoch: {}".format(epoch))
        train_loss = per_epoch(config, model, optimizer, train_dl, train=True, enable_tqdm=config.trainer_config.tqdm)
        val_loss = per_epoch(config, model, optimizer, val_dl, train=False, enable_tqdm=config.trainer_config.tqdm)
        
        if val_loss < global_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.ckpt"))
            print(f"*** Model Checkpoint Saved. ***\ncur_loss = {val_loss}\nglobal_loss = {global_loss}")
            global_loss = val_loss

def run(args):
    config.data.data_prefix = args.data_path if args.data_path else config.data.data_prefix
    config.device = torch.device(args.device)
    config.trainer_config.tqdm = args.tqdm
    config.trainer_config.epoch = args.epoch
    
    df = pd.read_csv(os.path.join(config.data.data_prefix, config.data.meta_file_name))
    train, val = train_test_split(df, train_size=config.trainer_config.train_size)    
    training_env(train, val, env_no=0)

if __name__ == '__main__':
    args = parser()
    run(args)