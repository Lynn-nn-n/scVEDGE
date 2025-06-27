import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import collections
from typing import Iterable
from torch.nn import init

class MLP(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
        use_spectral_norm: bool = True,  
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        self.use_spectral_norm = use_spectral_norm  
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        cat_dim = sum(self.n_cat_list)

        layers = collections.OrderedDict()
        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            linear_layer = nn.Linear(n_in + cat_dim * self.inject_into_layer(i), n_out, bias=bias)
            
            # 应用 Spectral Normalization
            if self.use_spectral_norm:
                linear_layer = spectral_norm(linear_layer)

            layer_list = [linear_layer]

            if use_batch_norm:
                layer_list.append(nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001))
            if use_layer_norm:
                layer_list.append(nn.LayerNorm(n_out, elementwise_affine=False))
            if use_activation:
                layer_list.append(activation_fn())
            if dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))

            layers[f"Layer {i}"] = nn.Sequential(*layer_list)

        self.mlp = nn.Sequential(layers)
        self.apply(self._init_weights)  
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

    def inject_into_layer(self, layer_num) -> bool:
        return layer_num == 0 or (layer_num > 0 and self.inject_covariates)

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        one_hot_cat_list = []
        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")

        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat
                one_hot_cat_list.append(one_hot_cat)

        for i, layers in enumerate(self.mlp):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, torch.nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [layer(slice_x).unsqueeze(0) if slice_x.shape[0] > 1 else slice_x.clone().detach().unsqueeze(0)
                                 for slice_x in x], dim=0
                            )
                        else:
                            if x.shape[0] == 1:
                                x = x.clone().detach()
                            else:
                                x = layer(x)
                    else:
                        if isinstance(layer, torch.nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.unsqueeze(1).long(), 1) 
    return onehot.float()
