import numpy as np
import torch
from tqdm import tqdm
from config.config import ConfigV1
from time import time
from torch.utils.data import Dataset, DataLoader

def per_epoch(config: ConfigV1, model: torch.nn.Module, optimizer:torch.optim.Optimizer, dl: DataLoader, train=True, enable_tqdm=True):
    total_batches = len(dl)
    _iter = tqdm(enumerate(dl), total=total_batches) if enable_tqdm else enumerate(dl)
    model.train(train)
    total_loss = 0
    c = 1
    for i, batch in _iter:
        strt_time = time()
        step_prc = int(i/(total_batches) * 100)
        x, y = batch[0].to(config.device), batch[1].to(config.device)
        if train:
            optimizer.zero_grad()
            loss: torch.Tensor = model(x, y)
            loss.backward()
            optimizer.step()
        else:
            loss: torch.Tensor = model(x, y)
        
        loss = loss.detach().cpu().item()
        total_loss += loss
        
        log = f"loss: {loss: .4f}, avg_loss: {total_loss/(i+1): .4f}"
        if enable_tqdm:
            _iter.set_description(log)
        else:
            if step_prc != c and step_prc % config.trainer_config.step_perc_v == 0:
                print(f"step({step_prc}%) :{log}, step time: {time() - strt_time}")
                c = step_prc
    
    return total_loss/len(dl)

def inference(config: ConfigV1, model: torch.nn.Module, ds: DataLoader, enable_tqdm=True) -> np.ndarray:
    
    total_batches = len(ds)
    _iter = enumerate(ds)
    model.train(False)
    result = np.zeros((total_batches, len(config.class_columns)), dtype=np.float32)
    with torch.no_grad():
        for i, batch in _iter:
            strt_time = time()
            x = batch[0].to(config.device)
            prob = model(x).softmax(dim=-1)
            result[i*config.trainer_config.batch_size: (i+1)*config.trainer_config.batch_size, :] = prob.cpu().numpy()
            print(f"Batch: {i} and size: {config.trainer_config.batch_size} completed, time taken: {time()-strt_time}")
    return result