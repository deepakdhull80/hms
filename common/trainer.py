import torch
from tqdm import tqdm
from config.config import ConfigV1

def per_epoch(config: ConfigV1, model: torch.nn.Module, optimizer:torch.optim.Optimizer, dl, train=True, enable_tqdm=True):
    total_batches = len(dl)
    _iter = tqdm(enumerate(dl), total=total_batches) if enable_tqdm else enumerate(dl)
    model.train(train)
    total_loss = 0
    for i, batch in _iter:
        step_prc = int(i/(total_batches) * 100)
        x, y = batch[0].to(config.device), batch[1].to(config.device)
        if train:
            optimizer.zero_grad()
            loss: torch.Tensor = model(x, y)
            loss.backward()
            optimizer.step()
        else:
            loss: torch.Tensor = model(x, y).cpu().item()
        
        loss = loss.detach().cpu().item()
        total_loss += loss
        
        log = f"loss: {loss: .4f}, avg_loss: {total_loss/(i+1): .4f}"
        if enable_tqdm:
            _iter.set_description(log)
        else:
            if step_prc % config.trainer_config.step_perc_v == 0:
                print(f"step({step_prc}%) :{log}")
    
    return total_loss/len(dl)