import torch
import numpy as np
from tqdm import tqdm


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

@torch.no_grad()
def get_scores(model, loader, leave=False, device='cpu', model_type='pytorch'):
    """
    <model_type> can be "pytorch" or "onnx"
    """
    model_type = model_type.lower()
    if model_type == "pytorch":
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
        x.append(sample_indices.to(device))
        if model_type == "pytorch":
            # Compute PyTorch output prediction
            batch_score = model(*x)
        elif model_type == "onnx":
            # Compute ONNX Runtime output prediction
            ort_inputs = {ort_input.name: to_numpy(x[i]) for i, ort_input in enumerate(model.get_inputs())}
            batch_score = torch.from_numpy(model.run(None, ort_inputs)[0])
        score_list.append(batch_score)
        t.refresh()  # to show immediately the update

    return to_numpy(torch.cat(score_list)).flatten(), to_numpy(torch.cat(target_list)).flatten(), to_numpy(torch.cat(weight_list)).flatten()


@torch.no_grad()
def get_r_hats(model, loader, leave=False, device='cpu', model_type='pytorch'):
    model_type = model_type.lower()
    if model_type == "pytorch":
        model.eval()

    r_hat_list = []
    t = tqdm(enumerate(loader), total=len(loader), leave=leave)
    for i, batch in t:
        data = batch[0]
        sample_indices = batch[1]
        x = [x_i.to(device) for x_i in data[0]]
        x.append(sample_indices.to(device))
        if model_type == "pytorch":
            # Compute PyTorch output prediction
            batch_output = model(*x)
        elif model_type == "onnx":
            # Compute ONNX Runtime output prediction
            ort_inputs = {ort_input.name: to_numpy(x[i]) for i, ort_input in enumerate(model.get_inputs())}
            batch_output = torch.from_numpy(model.run(None, ort_inputs)[0])
        r_hat = batch_output / (1 - batch_output)
        r_hat_list.append(r_hat)
        t.refresh()  # to show immediately the update

    return torch.cat(r_hat_list).cpu().numpy().flatten()


def get_calibrated_r_hats(model, calib_fn, loader, leave=False, device='cpu', model_type='pytorch'):
    model_type = model_type.lower()
    if model_type == "pytorch":
        model.eval()

    r_hat_list = []
    t = tqdm(enumerate(loader), total=len(loader), leave=leave)
    for i, batch in t:
        data = batch[0]
        sample_indices = batch[1]
        x = [x_i.to(device) for x_i in data[0]]
        x.append(sample_indices.to(device))
        if model_type == "pytorch":
            # Compute PyTorch output prediction
            batch_output = calib_fn(to_numpy(model(*x)))
        elif model_type == "onnx":
            # Compute ONNX Runtime output prediction
            ort_inputs = {ort_input.name: to_numpy(x[i]) for i, ort_input in enumerate(model.get_inputs())}
            batch_output = calib_fn(model.run(None, ort_inputs)[0])
        r_hat = batch_output / (1 - batch_output)
        r_hat_list.append(r_hat)
        t.refresh()  # to show immediately the update

    return np.concatenate(r_hat_list).flatten()
