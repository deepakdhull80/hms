import os
import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import ConfigV1
from models.base_model import Classifier
from common.dataloader import getDataLoader, getInferDataLoader
from common.trainer import per_epoch, inference
from common.utils import score

config = ConfigV1

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch", default=4)
    parser.add_argument("--tqdm", action='store_true')
    parser.add_argument("--batch_size", type=int, default=32)
    
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
            config.model_state_dict_path = os.path.join(checkpoint_dir, "model.ckpt")
            print(f"*** Model Checkpoint Saved. ***\ncur_loss = {val_loss}\nglobal_loss = {global_loss}")
            global_loss = val_loss

def infer_env(infer_df: pd.DataFrame):
    solution_df = infer_df[['eeg_id'] + config.class_column].copy()
    dirs = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(dirs, "runs", "checkpoints", f"infer")
    os.makedirs(checkpoint_dir, mode=777, exist_ok=True)
    ds = getInferDataLoader(config, infer_df)
    model = Classifier(config)
    model.load_weights(config.model_state_dict_path)
    model.to(config.device)
    result = inference(config, model, ds, enable_tqdm=config.trainer_config.tqdm)
    infer_df[config.class_columns] = result
    submission_df = infer_df[['eeg_id'] + config.class_columns].copy()
    final_score = score(solution_df, submission_df, "eeg_id")
    print("********************************")
    print("submission_score: ", final_score)
    print("********************************")

def run(args):
    config.data.data_prefix = args.data_path if args.data_path else config.data.data_prefix
    config.device = torch.device(args.device)
    config.trainer_config.tqdm = args.tqdm
    config.trainer_config.epoch = args.epoch
    config.trainer_config.batch_size = args.batch_size
    
    df = pd.read_csv(os.path.join(config.data.data_prefix, config.data.meta_file_name))
    train, val = train_test_split(df, train_size=config.trainer_config.train_size, random_state=config.random_state_seed)
    training_env(train, val, env_no=0)
    infer_env(val)

if __name__ == '__main__':
    args = parser()
    run(args)