import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    BatchNorm1d,
    AvgPool1d,
    Sigmoid,
    Conv1d,
)
import onnx, onnxruntime

import zipfile
import yaml
import collections
import io
from typing import List



class PhiNet(torch.nn.Module):
    def __init__(self, input_size, phi_nodes):
        super(PhiNet, self).__init__()
        phi_layers = [Conv1d(input_size, phi_nodes[0], 1),
                      ReLU()]
        for i in range(len(phi_nodes) - 1):
            phi_layers.append(Conv1d(phi_nodes[i], phi_nodes[i+1], 1))
            phi_layers.append(ReLU())
        self.phi = Seq(*phi_layers)

    def forward(self, x, sample_indices):
        rval = self.phi(x)
        return torch.cat([torch.mean(rval[:, :, sample_indices[j]:sample_indices[j+1]], dim=2) for j in range(sample_indices.size(0)-1)], dim=0)


class Reshape(torch.nn.Module):
    def __init__(self, input_size):
        super(Reshape, self).__init__()
        self.input_size = input_size
        
    def forward(self, x, sample_indices):
        return x.reshape(-1, self.input_size)


class DeepSetsEnsemble(torch.nn.Module):
    def __init__(self, features, phi_nodes, mlp_nodes, outputs=1):
        super(DeepSetsEnsemble, self).__init__()
        self.features = collections.OrderedDict(sorted(features.items()))
        self._num_features = len(features)
        self._phi_nodes = phi_nodes
        self._mlp_nodes = mlp_nodes
        self.phi = []
        for feat in self.features:
            # Create an embedding layer for the DeepSets style model
            if self.features[feat]["set"] is True:
                self.phi.append(torch.jit.script(PhiNet(self.features[feat]["size"], phi_nodes)))
            else:
                self.phi.append(torch.jit.trace(Reshape(self.features[feat]["size"])))
        self.phi = nn.ModuleList(self.phi)

        # Then have one large MLP that takes in the representations and standard inputs
        total_repr_size = 0
        for feat in self.features:
            if self.features[feat]["set"] is True:
                total_repr_size += phi_nodes[-1]
            else:
                total_repr_size += self.features[feat]["size"]
                
        mlp_layers = [Lin(total_repr_size, mlp_nodes[0]), BatchNorm1d(mlp_nodes[0]), ReLU()]
        for i in range(len(mlp_nodes) - 1):
            mlp_layers.append(Lin(mlp_nodes[i], mlp_nodes[i+1]))
            mlp_layers.append(BatchNorm1d(mlp_nodes[i+1]))
            mlp_layers.append(ReLU())
        mlp_layers.append(Lin(mlp_nodes[-1], outputs))
        mlp_layers.append(Sigmoid())
        self.mlp = Seq(*mlp_layers)


    def forward(self, *args: List[torch.Tensor]): #x: ModelArgs, sample_indices: torch.Tensor):
        x = args[:-1]
        sample_indices = args[-1]
        zero_pad = torch.zeros(sample_indices.size(0), 1, dtype=torch.int64).to(sample_indices.device)
        sample_indices = torch.cat([zero_pad, sample_indices], dim=1)
        sample_indices = torch.cumsum(sample_indices, dim=1)
        repr_values = []
        for i, phi in enumerate(self.phi):
            repr_values.append(phi(x[i], sample_indices[i]))
        z = torch.cat(repr_values, dim=1)
        return self.mlp(z)
        

    def save(self, path):
        torch.save(self.state_dict(), path)
        


class DeepSetsEnsemble2(torch.nn.Module):
    def __init__(self, features, phi_nodes, mlp_nodes, outputs=1):
        super(DeepSetsEnsemble, self).__init__()
        self.features = collections.OrderedDict(sorted(features.items()))
        #self._feat_is_set = torch.tensor([self.features[feat]["set"] for feat in self.features], dtype=int)
        self._feat_is_set = [int(self.features[feat]["set"]) for feat in self.features]
        self._phi_counter = torch.cumsum(torch.tensor(self._feat_is_set), dim=0) - 1
        self._feat_size = torch.tensor([self.features[feat]["size"] for feat in self.features])
        self._num_features = len(features)
        self._phi_nodes = phi_nodes
        self._mlp_nodes = mlp_nodes
        self.phi = []
        for feat in self.features:
            # Create an embedding layer for the DeepSets style model
            if self.features[feat]["set"] is True:
                phi_layers = [Conv1d(self.features[feat]["size"], phi_nodes[0], 1),
                              ReLU()]
                for i in range(len(phi_nodes) - 1):
                    phi_layers.append(Conv1d(phi_nodes[i], phi_nodes[i+1], 1))
                    phi_layers.append(ReLU())
                self.phi.append(Seq(*phi_layers))
        self.phi = nn.ModuleList(self.phi)

        # Then have one large MLP that takes in the representations and standard inputs
        total_repr_size = 0
        for feat in self.features:
            if self.features[feat]["set"] is True:
                total_repr_size += phi_nodes[-1]
            else:
                total_repr_size += self.features[feat]["size"]
                
        mlp_layers = [Lin(total_repr_size, mlp_nodes[0]), BatchNorm1d(mlp_nodes[0]), ReLU()]
        for i in range(len(mlp_nodes) - 1):
            mlp_layers.append(Lin(mlp_nodes[i], mlp_nodes[i+1]))
            mlp_layers.append(BatchNorm1d(mlp_nodes[i+1]))
            mlp_layers.append(ReLU())
        mlp_layers.append(Lin(mlp_nodes[-1], outputs))
        mlp_layers.append(Sigmoid())
        self.mlp = Seq(*mlp_layers)


    def encoding_layer(self, i, x, sample_indices):
        if self._feat_is_set[i] == 1:
            rval = self.phi[self._phi_counter[i]](x)
            rval = torch.cat([torch.mean(rval[:, :, sample_indices[j]:sample_indices[j+1]], dim=2) for j in range(sample_indices.size(0)-1)], dim=0)
        else:
            rval = x.reshape(-1, self._feat_size[i])
        return rval


    def forward(self, *args):
        x = args[:-1]
        sample_indices = args[-1]
        zero_pad = torch.zeros(sample_indices.size(0), 1, dtype=int).to(sample_indices.device)
        sample_indices = torch.cat([zero_pad, sample_indices], dim=1)
        sample_indices = torch.cumsum(sample_indices, dim=1)
        #repr_values = []
        #phi_counter = torch.cumsum(self._feat_is_set, dim=0) - 1
        #for i in range(self._num_features):
        #    if self._feat_is_set[i] == 1:
        #        out_i = self.phi[phi_counter[i]](x[i])
        #        out_i = torch.cat([torch.mean(out_i[:, :, sample_indices[i,j]:sample_indices[i,j+1]], dim=2) for j in range(sample_indices.size(1)-1)], dim=0)
        #        repr_values.append(out_i)
        #    else:
        #        repr_values.append(x[i].reshape(-1, self._feat_size[i]))
        repr_values = [self.encoding_layer(i, x[i], sample_indices[i]) for i in range(self._num_features)]
        z = torch.cat(repr_values, dim=1)
        return self.mlp(z)
        

    def save(self, path):
        torch.save(self.state_dict(), path)
        

class OldDeepSetsEnsemble(torch.nn.Module):
    def __init__(self, features, phi_nodes, mlp_nodes, outputs=1):
        super(DeepSetsEnsemble, self).__init__()
        self.features = collections.OrderedDict(sorted(features.items()))
        self._feat_is_set = torch.tensor([self.features[feat]["set"] for feat in self.features], dtype=int)
        self._phi_counter = torch.cumsum(self._feat_is_set, dim=0) - 1
        self._feat_size = torch.tensor([self.features[feat]["size"] for feat in self.features])
        self._num_features = len(features)
        self._phi_nodes = phi_nodes
        self._mlp_nodes = mlp_nodes
        self.phi = []
        for feat in self.features:
            # Create an embedding layer for the DeepSets style model
            if self.features[feat]["set"] is True:
                phi_layers = [Conv1d(self.features[feat]["size"], phi_nodes[0], 1),
                              ReLU()]
                for i in range(len(phi_nodes) - 1):
                    phi_layers.append(Conv1d(phi_nodes[i], phi_nodes[i+1], 1))
                    phi_layers.append(ReLU())
                self.phi.append(Seq(*phi_layers))

        # Then have one large MLP that takes in the representations and standard inputs
        total_repr_size = 0
        for feat in self.features:
            if self.features[feat]["set"] is True:
                total_repr_size += phi_nodes[-1]
            else:
                total_repr_size += self.features[feat]["size"]
                
        mlp_layers = [Lin(total_repr_size, mlp_nodes[0]), BatchNorm1d(mlp_nodes[0]), ReLU()]
        for i in range(len(mlp_nodes) - 1):
            mlp_layers.append(Lin(mlp_nodes[i], mlp_nodes[i+1]))
            mlp_layers.append(BatchNorm1d(mlp_nodes[i+1]))
            mlp_layers.append(ReLU())
        mlp_layers.append(Lin(mlp_nodes[-1], outputs))
        mlp_layers.append(Sigmoid())
        self.mlp = Seq(*mlp_layers)


    # override
    def __repr__(self):
        stringified = self._get_name() + '(\n'
        for i, p in enumerate(self.phi):
            stringified += '  (phi{}):'.format(i) + '\n'.join(['  ' + s for s in p.__str__().split('\n')]) + '\n'
        stringified += '  (mlp):' + '\n'.join(['  ' + s for s in self.mlp.__str__().split('\n')])
        stringified += '\n)'
        return stringified


    def encoding_layer(self, i, x, sample_indices):
        if self._feat_is_set[i] == 1:
            rval = self.phi[self._phi_counter[i]](x)
            rval = torch.cat([torch.mean(rval[:, :, sample_indices[j]:sample_indices[j+1]], dim=2) for j in range(sample_indices.size(0)-1)], dim=0)
        else:
            rval = x.reshape(-1, self._feat_size[i])
        return rval


    def forward(self, *args):
        x = args[:-1]
        sample_indices = args[-1]
        sample_indices = torch.cat([torch.zeros(sample_indices.size(0), 1, dtype=int), sample_indices], dim=1)
        sample_indices = torch.cumsum(sample_indices, dim=1)
        #repr_values = []
        #phi_counter = torch.cumsum(self._feat_is_set, dim=0) - 1
        #for i in range(self._num_features):
        #    if self._feat_is_set[i] == 1:
        #        out_i = self.phi[phi_counter[i]](x[i])
        #        out_i = torch.cat([torch.mean(out_i[:, :, sample_indices[i,j]:sample_indices[i,j+1]], dim=2) for j in range(sample_indices.size(1)-1)], dim=0)
        #        repr_values.append(out_i)
        #    else:
        #        repr_values.append(x[i].reshape(-1, self._feat_size[i]))
        repr_values = [self.encoding_layer(i, x[i], sample_indices[i]) for i in range(self._num_features)]
        z = torch.cat(repr_values, dim=1)
        return self.mlp(z)

    #  override
    def parameters(self):
        for p in self.phi:
            for q in p.parameters():
                yield q
        for p in self.mlp.parameters():
            yield p

    # override
    def to(self, device, *args, **kwargs):
        self.mlp = self.mlp.to(device, *args, **kwargs)
        self.phi = [p.to(device, *args, **kwargs) for p in self.phi]
        return self

    # override
    def state_dict(self, **kwargs):
        state_dict = collections.OrderedDict()
        for i, p in enumerate(self.phi):
            for key, value in p.state_dict(**kwargs).items():
                state_dict["phi{}.{}".format(i,key)] = value
        for key, value in self.mlp.state_dict(**kwargs).items():
            state_dict["mlp.{}".format(key)] = value
        return state_dict

    # override
    def load_state_dict(self, state_dict):
        mlp_state_dict = collections.OrderedDict()
        phi_state_dicts = [collections.OrderedDict() for i in range(len(self.phi))]
        for key, params in state_dict.items():
            split_key = key.split('.')
            network, param_name = split_key[0], '.'.join(split_key[1:])
            if network == 'mlp':
                mlp_state_dict[param_name] = params
            elif network.startswith("phi"):
                ix = int(network[3:])
                phi_state_dicts[ix][param_name] = params

        self.mlp.load_state_dict(mlp_state_dict)
        for i in range(len(self.phi)):
            self.phi[i].load_state_dict(phi_state_dicts[i])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, is_zip=False, device='cpu'):
        if is_zip is False:
            self.load_state_dict(torch.load(path, map_location=device))
        return self


def load_model(path_to_zip, device='cpu'):
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        model_metadata = yaml.load(zf.read("deepsets_metadata.yaml"), Loader=yaml.CLoader)["model"]
        features = collections.OrderedDict(sorted(model_metadata["features"].items()))
        model = DeepSetsEnsemble(model_metadata["features"], model_metadata["phi"], model_metadata["mlp"])
        model.load_state_dict(torch.load(io.BytesIO(zf.read("deepsets_ensemble_best.pth")), map_location=device))
    return model

def load_onnx_model(path_to_zip, device='cpu'):
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        onnx_model = onnx.load(io.BytesIO(zf.read("deepsets_model.onnx")))
    return onnx_model

def load_onnx_session(path_to_zip, device='cpu'):
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        onnx_model = onnxruntime.InferenceSession(zf.read("deepsets_model.onnx"),
                                                  providers=['TensorrtExecutionProvider',
                                                             ('CUDAExecutionProvider', {"cudnn_conv_algo_search": "DEFAULT"}),
                                                             'CPUExecutionProvider'])
    return onnx_model