import torch
import torch.nn.functional as F

import zipfile
import yaml
import os
import os.path as osp
import collections
from tqdm import tqdm

from . import preprocessing as carl_preprocessing
from . import models as carl_models


@torch.no_grad()
def test(model, loader, leave=False, device='cpu'):
    model.eval()

    sum_loss = 0.0
    t = tqdm(enumerate(loader), total=len(loader), position=0, leave=leave)
    for i, batch in t:
        data = batch[0]
        sample_indices = batch[1]
        x = [x_i.to(device) for x_i in data[0]]
        y = data[1].to(device)
        w = data[2].to(device)
        sample_indices.to(device)
        batch_output = model(*x, sample_indices)
        batch_loss_item = F.binary_cross_entropy(batch_output, y, weight=w).cpu().item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh()  # to show immediately the update

    return sum_loss / (i + 1)


def train_epoch(model, optimizer, loader, leave=False, device='cpu'):
    model.train()

    sum_loss = 0.0
    t = tqdm(enumerate(loader), total=len(loader), position=0, leave=leave)
    for i, batch in t:
        data = batch[0]
        sample_indices = batch[1]
        optimizer.zero_grad()
        x = [x_i.to(device) for x_i in data[0]]
        y = data[1].to(device)
        w = data[2].to(device)
        sample_indices.to(device)
        batch_output = model(*x, sample_indices)
        batch_loss = F.binary_cross_entropy(batch_output, y, weight=w)
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("loss = %.5f" % batch_loss_item)
        t.refresh()  # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()

    return sum_loss / (i + 1)


def train_loop(model, optimizer, train_loader, validation_loader, n_epochs, patience=5, return_best_model=True, device='cpu', saveAs="deepsets_model", model_metadata=None):
    stale_epochs = 0
    best_valid_loss = 99999
    
    t = tqdm(range(0, n_epochs), leave=False)

    train_losses = []
    val_losses = []

    for epoch in t:
        loss = train_epoch(
            model,
            optimizer,
            train_loader,
            leave=bool(epoch == n_epochs - 1),
            device=device
        )
        valid_loss = test(
            model,
            validation_loader,
            leave=bool(epoch == n_epochs - 1),
            device=device
        )
        train_losses.append(loss)
        val_losses.append(valid_loss)
        print("Epoch: {:02d}, Training Loss:   {:.4f}".format(epoch+1, loss))
        print("           Validation Loss: {:.4f}".format(valid_loss))
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if model_metadata is not None:
                save_model_data(model, model_metadata, name=saveAs, device=device)
                print("New best model saved to: {}.zip".format(saveAs))
            else:
                modpath = osp.join("{}.pth".format(saveAs))
                model.save(modpath)
                print("New best model saved to:", modpath)
            stale_epochs = 0
        else:
            print("Stale epoch")
            stale_epochs += 1
        if stale_epochs >= patience:
            print("Early stopping after %i stale epochs" % patience)
            break

    if return_best_model is True:
        if model_metadata is None:
            model.load(osp.join("{}.pth".format(saveAs)))
        else:
            model = carl_models.load_model("{}.zip".format(saveAs))
    return model, (train_losses, val_losses)


def train(model_settings, training_dataset, validation_dataset, optimizer="Adam", learning_rate=1e-2, batch_size=128, n_epochs=10, patience=5, return_best_model=True, device='cpu', saveAs="deepsets_model", **kwargs):
    training_settings = {
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "patience": patience,
        "device": device
    }
    training_settings.update(kwargs)

    features = collections.OrderedDict(sorted(model_settings["features"].items()))

    print("Constructing the model")
    model = carl_models.DeepSetsEnsemble(features, model_settings["phi"], model_settings["mlp"])
    model = model.to(device)
    if optimizer.lower() == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)    
    else:
        print("{} is not implemented yet as an optimizer".format(optimizer))

    print("Loading the input data scaling")
    # Do some data preprocessing for standardized inputs and weights
    X_scalers, weight_norm = carl_preprocessing.get_scaling(training_dataset, features)

    # Prepare the data for training
    train_loader = carl_preprocessing.get_training_DataLoader(training_dataset, features,
                                                              X_scalers, weight_norm=weight_norm,
                                                              batch_size=batch_size, shuffle=True)
    valid_loader = carl_preprocessing.get_training_DataLoader(validation_dataset, features,
                                                              X_scalers, weight_norm=weight_norm,
                                                              batch_size=batch_size, shuffle=False)
    
    model_metadata = get_model_metadata(training_settings, model, X_scalers, weight_norm)
    print("Training the model")
    model, losses = train_loop(
        model,
        optim,
        train_loader,
        valid_loader,
        n_epochs,
        patience=patience,
        device=device,
        return_best_model=return_best_model,
        saveAs=saveAs,
        model_metadata=model_metadata
    )
    print("Finished training")
    return model, losses


def get_model_metadata(training_settings, model, input_scalers, weight_scale):
    model_settings = {
        "features": model.features,
        "phi": model._phi_nodes,
        "mlp": model._mlp_nodes
    }

    scaling_settings = {
        "inputs": [{
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "var": scaler.var_
        } for scaler in input_scalers],
        "weights": weight_scale
    }
    
    metadata = {
        "model": model_settings,
        "training": training_settings,
        "scaling": scaling_settings        
    }
    return metadata    


def save_model_data(model, metadata, name="deepsets_model", save_onnx=True, device='cpu'):
    yaml.dump(metadata, open("deepsets_metadata.yaml", 'w'))
    model.save("deepsets_ensemble_best.pth")
    if save_onnx is True:
        model = model.to('cpu')
        model.eval()
        test_input = []
        input_names = []
        for k in model.features:
            test_input.append(torch.randn(1, model.features[k]["size"], requires_grad=True).T[None, :])
            input_names.append(k)
        test_input.append(torch.ones(len(model.features), 1, dtype=int))
        input_names.append("sample_indices")
        test_input = tuple(test_input)
        torch.onnx.export(model,
                          test_input,
                          "deepsets_model.onnx",
                          export_params=True,
                          opset_version=11,
                          input_names=input_names,
                          output_names=["output"])
        model = model.to(device)
        
    
    with zipfile.ZipFile("{}.zip".format(name), mode='w') as zipf:
        zipf.write("deepsets_metadata.yaml")
        zipf.write("deepsets_ensemble_best.pth")
        zipf.write("deepsets_model.onnx")

    os.remove("deepsets_metadata.yaml")
    os.remove("deepsets_ensemble_best.pth")
    os.remove("deepsets_model.onnx")
    return None


def load_training_settings(path_to_zip):
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        return yaml.load(zf.read("deepsets_metadata.yaml"), Loader=yaml.CLoader)["training"]