import os
import argparse
import torch
import numpy as np
import pandas as pd

from config.config import ConfigV1
from models.base_model import Classifier
from common.dataloader import getInferDataLoader
from common.trainer import inference

config = ConfigV1

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch", default=4)
    parser.add_argument("--tqdm", action='store_true')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--submission_export_path")
    parser.add_argument("--model_state_dict_path")
    
    return parser.parse_args()

def infer_env(infer_df, model_state_dict_path):
    ds = getInferDataLoader(config, infer_df)
    model = Classifier(config)
    model.load_weights(model_state_dict_path)
    model.to(config.device)
    result = inference(config, model, ds,total_samples=infer_df.shape[0])
    return result

def esemble_infer(infer_df: pd.DataFrame):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    results = []
    for k in range(config.k_folds):
        checkpoint_path = os.path.join(cur_dir, "runs", "checkpoints", f"env_{k}", "model.ckpt")
        result = infer_env(infer_df.copy(), checkpoint_path)
        results.append(result)
    result = np.stack(results).mean(axis=0)
    infer_df[config.class_columns] = result
    return infer_df[['eeg_id'] + config.class_columns]

def run(args):
    config.data.data_prefix = args.data_path if args.data_path else config.data.data_prefix
    config.device = torch.device(args.device)
    config.trainer_config.tqdm = args.tqdm
    config.trainer_config.epoch = args.epoch
    config.trainer_config.batch_size = args.batch_size
    
    if args.submission_export_path:
        config.submission_export_path = args.submission_export_path
    if args.model_state_dict_path:
        config.model_state_dict_path = args.model_state_dict_path
        config.model.load_pretrained_weights = False
    
    df = pd.read_csv(os.path.join(config.data.data_prefix, config.data.test_meta_file_name))
    
    result_df = esemble_infer(df)
    result_df.to_csv(f"{config.submission_export_path}/submission.csv", index=False)

if __name__ == '__main__':
    args = parser()
    run(args)