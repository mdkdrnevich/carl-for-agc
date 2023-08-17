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

import zipfile
import yaml
import collections
import io


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


    def forward(self, x, sample_indices):
        sample_indices = torch.cat([torch.zeros(sample_indices.size(0), 1, dtype=int), sample_indices], dim=1)
        sample_indices = torch.cumsum(sample_indices, dim=1)
        repr_values = []
        phi_counter = 0
        for i, feat in enumerate(self.features):
            if self.features[feat]["set"] is True:
                out_i = self.phi[phi_counter](x[i])
                out_i = torch.cat([torch.mean(out_i[:, :, sample_indices[i,j]:sample_indices[i,j+1]], dim=2) for j in range(sample_indices.size(1)-1)], dim=0)
                repr_values.append(out_i)
                phi_counter += 1
            else:
                #print(x[i].size())
                repr_values.append(x[i].reshape(-1, self.features[feat]["size"]))
        #[print(_.size()) for _ in repr_values]
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
    def state_dict(self):
        state_dict = {}
        for i, p in enumerate(self.phi):
            state_dict["phi.{}".format(i)] = p.state_dict()
        state_dict["mlp"] = self.mlp.state_dict()
        return state_dict

    # override
    def load_state_dict(self, state_dict):
        self.mlp.load_state_dict(state_dict["mlp"])
        for key in state_dict:
            if key.startswith("phi"):
                ix = int(key.split('.')[-1])
                self.phi[ix].load_state_dict(state_dict[key])

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