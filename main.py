import os
import argparse
import torch
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold

from config import ConfigV1
from models.base_model import Classifier
from common.dataloader import getDataLoader
from common.trainer import per_epoch, inference
from common.utils import score

config = ConfigV1

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch", default=config.trainer_config.epoch, type=int)
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
    chkpt_path = os.path.join(checkpoint_dir, "model.ckpt")
    for epoch in range(config.trainer_config.epoch):
        print("################################")
        print("Epoch: {}".format(epoch))
        train_loss = per_epoch(config, model, optimizer, train_dl, train=True, enable_tqdm=config.trainer_config.tqdm)
        # val_loss = per_epoch(config, model, optimizer, val_dl, train=False, enable_tqdm=config.trainer_config.tqdm)
        val_loss = infer_env(val_df, env_no, verbose=False)
        print(f"********* CV: {val_loss} **************")
        if val_loss < global_loss:
            torch.save(model.state_dict(), chkpt_path)
            config.model_state_dict_path = chkpt_path
            print(f"*** Model Checkpoint Saved. ***\ncur_loss = {val_loss}\nglobal_loss = {global_loss}")
            global_loss = val_loss
    return global_loss, chkpt_path

def infer_env(infer_df: pd.DataFrame, env_no, verbose:bool=True):
    solution_df = infer_df[['eeg_id'] + config.class_columns].copy()
    solution_df[config.class_columns] = torch.tensor(solution_df[config.class_columns].values).float().softmax(dim=-1).numpy()
    ds, _ = getDataLoader(config, infer_df)
    model = Classifier(config)
    
    dirs = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(dirs, "runs", "checkpoints", f"env_{env_no}")
    model.load_weights(os.path.join(checkpoint_dir, "model.ckpt"))
    model.to(config.device)
    result = inference(config, model, ds, total_samples=infer_df.shape[0], verbose=verbose)
    infer_df[config.class_columns] = result
    submission_df = infer_df[['eeg_id'] + config.class_columns].copy()
    final_score = score(solution_df, submission_df, "eeg_id")
    print("********************************")
    print("submission_score: ", final_score)
    print("********************************")
    return final_score

def run(args):
    config.data.data_prefix = args.data_path if args.data_path else config.data.data_prefix
    config.device = torch.device(args.device)
    config.trainer_config.tqdm = args.tqdm
    config.trainer_config.epoch = args.epoch
    config.trainer_config.batch_size = args.batch_size
    
    df = pd.read_csv(os.path.join(config.data.data_prefix, config.data.meta_file_name))

    stratify_split = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.random_state_seed,)
    train_logs = []
    for i, (train_idx, val_idx) in enumerate(stratify_split.split(df, df['expert_consensus'])):
        print(f"#Fold: {i}#")
        train, val = df.iloc[train_idx], df.iloc[val_idx]
        _loss, model_ckpt_path = training_env(train, val, env_no=i)
        kl_loss = infer_env(val, env_no=i)
        
        train_logs.append({
            'fold': i,
            'kl_loss': kl_loss,
            'model_path': model_ckpt_path
        })
    json.dump(train_logs, open(os.path.join(config.data.data_prefix, "runs/runner.logs"), 'w'))
    print(train_logs)

if __name__ == '__main__':
    args = parser()
    run(args)