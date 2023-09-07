import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def get_scores(model, loader, leave=False, device='cpu'):
    model.eval()

    score_list = []
    target_list = []
    weight_list = []
    t = tqdm(enumerate(loader), total=len(loader), leave=leave)
    for i, batch in t:
        data = batch[0]
        target_list.append(data[1])
        weight_list.append(data[2])
        sample_indices = batch[1]
        x = [x_i.to(device) for x_i in data[0]]
        sample_indices.to(device)
        batch_score = model(*x, sample_indices)
        score_list.append(batch_score)
        t.refresh()  # to show immediately the update

    return torch.cat(score_list).cpu().numpy().flatten(), torch.cat(target_list).cpu().numpy().flatten(), torch.cat(weight_list).cpu().numpy().flatten()


@torch.no_grad()
def get_r_hats(model, loader, leave=False, device='cpu'):
    model.eval()

    r_hat_list = []
    t = tqdm(enumerate(loader), total=len(loader), leave=leave)
    for i, batch in t:
        data = batch[0]
        sample_indices = batch[1]
        x = [x_i.to(device) for x_i in data[0]]
        sample_indices.to(device)
        batch_output = model(*x, sample_indices)
        r_hat = batch_output / (1 - batch_output)
        r_hat_list.append(r_hat)
        t.refresh()  # to show immediately the update

    return torch.cat(r_hat_list).cpu().numpy().flatten()


def get_calibrated_r_hats(model, calib_fn, loader, leave=False, device='cpu'):
    model.eval()

    r_hat_list = []
    t = tqdm(enumerate(loader), total=len(loader), leave=leave)
    for i, batch in t:
        data = batch[0]
        sample_indices = batch[1]
        x = [x_i.to(device) for x_i in data[0]]
        sample_indices.to(device)
        batch_output = calib_fn(model(*x, sample_indices).cpu().detach().numpy())
        r_hat = batch_output / (1 - batch_output)
        r_hat_list.append(r_hat)
        t.refresh()  # to show immediately the update

    return np.concatenate(r_hat_list).flatten()
