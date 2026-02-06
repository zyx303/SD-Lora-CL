# SD-Adapter: Applying the SD-LoRA spectral decomposition approach with bottleneck adapters
# Adapted from backbone/lora.py
#
# Instead of low-rank A*B on Q/V attention projections (LoRA),
# this inserts bottleneck adapters (down -> ReLU -> up) in the MLP path
# of each transformer block, using the same SD scaling/normalization framework.

import math
import copy
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch import Tensor
from torch.nn.parameter import Parameter
from timm.models.vision_transformer import VisionTransformer as timm_ViT

from backbone.linears import SimpleLinear


class ParameterWrapper(nn.Module):
    """Wraps a scalar parameter and multiplies it with the input."""
    def __init__(self, param):
        super(ParameterWrapper, self).__init__()
        self.param = param

    def forward(self, x):
        return x * self.param


class _Adapter_MLP_train(nn.Module):
    """Wraps the original MLP and adds SD-Adapter contributions during training.

    For previous tasks: normalized adapter output (direction) scaled by learned coefficients.
    For current task: raw adapter output scaled by learned coefficient.
    """
    def __init__(self, mlp, adapter_down, adapter_up,
                 task_id, saved_downs, saved_ups, t_layer_i, rank, dim,
                 scaling_factor, scaling_factor_prev, per_layer_scaling=False):
        super().__init__()
        self.mlp = mlp
        self.adapter_down = adapter_down.cuda()
        self.adapter_up = adapter_up.cuda()

        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()

        self.task_id = task_id
        self.saved_downs = saved_downs
        self.saved_ups = saved_ups
        self.t_layer_i = t_layer_i
        self.rank = rank
        self.dim = dim
        self.per_layer_scaling = per_layer_scaling

    def forward(self, x):
        # Original MLP output
        mlp_output = self.mlp(x)

        temp_down = nn.Linear(self.dim, self.rank, bias=False)
        temp_up = nn.Linear(self.rank, self.dim, bias=False)

        new_adapter = 0
        for i in range(self.task_id):
            saved_down_i = self.saved_downs['saved_down_' + str(i)]
            saved_up_i = self.saved_ups['saved_up_' + str(i)]

            down_layer = saved_down_i[self.t_layer_i]
            up_layer = saved_up_i[self.t_layer_i]

            temp_down.weight = Parameter(down_layer.weight)
            temp_down.weight.requires_grad = False
            temp_down.to(x.device)
            temp_up.weight = Parameter(up_layer.weight)
            temp_up.weight.requires_grad = False
            temp_up.to(x.device)

            # Normalized adapter output for previous tasks (direction)
            adapter_out = temp_up(F.relu(temp_down(x)))
            norm_factor = torch.norm(temp_down.weight) * torch.norm(temp_up.weight)
            adapter_out = adapter_out / norm_factor

            scale_idx = self.t_layer_i if self.per_layer_scaling else 0
            if i == 0:
                new_adapter = self.scaling_factor_prev[i][scale_idx](adapter_out)
            else:
                new_adapter += self.scaling_factor_prev[i][scale_idx](adapter_out)

        # Current task adapter (not normalized, raw magnitude)
        scale_idx = self.t_layer_i if self.per_layer_scaling else 0
        new_adapter += self.scaling_factor[scale_idx](
            self.adapter_up(F.relu(self.adapter_down(x)))
        )

        return mlp_output + new_adapter


class _Adapter_MLP_eval(nn.Module):
    """Wraps the original MLP and adds SD-Adapter contributions during evaluation."""
    def __init__(self, task_id, mlp, saved_downs, saved_ups, t_layer_i, rank, dim,
                 scaling_factor, scaling_factor_prev, save_file, per_layer_scaling=False):
        super().__init__()
        self.task_id = task_id
        self.mlp = mlp
        self.saved_downs = saved_downs
        self.saved_ups = saved_ups
        self.t_layer_i = t_layer_i
        self.rank = rank
        self.dim = dim
        self.save_file = save_file
        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()
        self.per_layer_scaling = per_layer_scaling

    def forward(self, x):
        mlp_output = self.mlp(x)
        new_adapter = 0

        temp_down = nn.Linear(self.dim, self.rank, bias=False)
        temp_up = nn.Linear(self.rank, self.dim, bias=False)

        file_path = self.save_file + 'scaling_factor' + str(self.task_id - 1) + '.pt'
        scaling_param = torch.load(file_path)

        for i in range(self.task_id):
            saved_down_i = self.saved_downs['saved_down_' + str(i)]
            saved_up_i = self.saved_ups['saved_up_' + str(i)]

            down_layer = saved_down_i[self.t_layer_i]
            up_layer = saved_up_i[self.t_layer_i]

            temp_down.weight = Parameter(down_layer.weight)
            temp_up.weight = Parameter(up_layer.weight)

            # Normalized adapter output (direction)
            adapter_out = temp_up(F.relu(temp_down(x)))
            norm_factor = torch.norm(temp_down.weight) * torch.norm(temp_up.weight)
            adapter_out = adapter_out / norm_factor

            scale_idx = self.t_layer_i if self.per_layer_scaling else 0
            if i == 0:
                new_adapter = self.scaling_factor_prev[i][scale_idx](adapter_out)
            else:
                new_adapter += self.scaling_factor_prev[i][scale_idx](adapter_out)

        # Last task contribution with current scaling factor
        scale_idx = self.t_layer_i if self.per_layer_scaling else 0
        new_adapter = self.scaling_factor[scale_idx](
            temp_up(F.relu(temp_down(x)))
        )

        return mlp_output + new_adapter


class SDAdapter_ViT_timm(nn.Module):
    """SD-Adapter for Vision Transformer (timm version).

    Applies the SD-LoRA spectral decomposition approach using bottleneck adapters
    (down_proj -> ReLU -> up_proj) inserted in the MLP path of each transformer block.

    Args:
        vit_model: a timm vision transformer model
        r: bottleneck dimension of the adapter
        num_classes: number of output classes
        increment: incremental class step size
        filepath: path for saving/loading parameters
        lora_layer: which layers to apply adapter (default: all)
        eval: whether in evaluation mode
        index: whether to initialize task_id
        cur_task_index: current task index for loading
        per_layer_scaling: whether to use per-layer scaling factors
    """
    def __init__(self, vit_model: timm_ViT, r: int, num_classes: int = 0,
                 increment=10, filepath='./', lora_layer=None, eval=False,
                 index=True, cur_task_index=None, per_layer_scaling=False):
        super(SDAdapter_ViT_timm, self).__init__()

        assert r > 0
        self.rank = r
        self.base_vit = copy.deepcopy(vit_model)
        self.per_layer_scaling = per_layer_scaling

        if not eval:
            self.save_file = filepath
            self.increment = increment
            print('save_file', self.save_file)

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        # Storage for current task adapter parameters
        self.adapter_downs = []  # Down-projection layers (dim -> r)
        self.adapter_ups = []    # Up-projection layers (r -> dim)

        if index:
            print('Initialize task-id and curtask id')
            self.task_id, self.cur_id = 0, 0

        if cur_task_index is not None:
            self.task_id = cur_task_index

        # Freeze the saved base model
        for param in self.base_vit.parameters():
            param.requires_grad = False

        # Freeze the working model
        for param in vit_model.parameters():
            param.requires_grad = False

        # Load saved adapter parameters from previous tasks
        saved_adapter_down, saved_adapter_up = {}, {}
        for i in range(self.task_id):
            file_path = self.save_file + 'adapter_down_' + str(i) + '.pt'
            saved_adapter_down['saved_down_' + str(i)] = torch.load(file_path)
            file_path = self.save_file + 'adapter_up_' + str(i) + '.pt'
            saved_adapter_up['saved_up_' + str(i)] = torch.load(file_path)

        # Number of layers with adapters
        self.num_adapter_layers = len(self.lora_layer)

        # Create scaling factors: per-layer or global
        if self.per_layer_scaling:
            self.wrapped_param = nn.ModuleList([
                ParameterWrapper(nn.Parameter(torch.Tensor([0.8])))
                for _ in range(self.num_adapter_layers)
            ])
            self.wrapped_param_prev = nn.ModuleList([
                nn.ModuleList([
                    ParameterWrapper(nn.Parameter(torch.Tensor([0.8])))
                    for _ in range(self.num_adapter_layers)
                ])
                for _ in range(20)
            ])
        else:
            scaling_factor = nn.Parameter(torch.Tensor([0.8]))
            self.wrapped_param = nn.ModuleList([ParameterWrapper(scaling_factor)])
            self.wrapped_param_prev = nn.ModuleList([
                ParameterWrapper(nn.Parameter(torch.Tensor([0.8])))
                for _ in range(20)
            ])

        # Surgery: replace blk.mlp with adapter-wrapped version
        for t_layer_i, blk in enumerate(vit_model.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_mlp = blk.mlp
            self.dim = blk.attn.qkv.in_features

            adapter_down = nn.Linear(self.dim, r, bias=False)
            adapter_up = nn.Linear(r, self.dim, bias=False)

            self.adapter_downs.append(adapter_down)
            self.adapter_ups.append(adapter_up)

            if not eval:
                blk.mlp = _Adapter_MLP_train(
                    w_mlp, adapter_down, adapter_up,
                    self.task_id, saved_adapter_down, saved_adapter_up,
                    t_layer_i, self.rank, self.dim,
                    self.wrapped_param, self.wrapped_param_prev,
                    per_layer_scaling=self.per_layer_scaling
                )
            else:
                blk.mlp = _Adapter_MLP_eval(
                    self.task_id, w_mlp, saved_adapter_down, saved_adapter_up,
                    t_layer_i, self.rank, self.dim,
                    self.wrapped_param, self.wrapped_param_prev,
                    self.save_file, per_layer_scaling=self.per_layer_scaling
                )

        self.reset_parameters()
        self.adapter_vit = vit_model
        if not eval:
            self.adapter_vit.head = torch.nn.Identity()
        else:
            self.reset_adapter_vit_head()

    def reset_adapter_vit_head(self):
        task_incremental = self.increment
        self.adapter_vit.head = self.generate_fc(768, (self.task_id) * task_incremental).cuda()
        temp_weights = torch.load(self.save_file + 'CLs_weight' + str(self.task_id - 1) + '.pt')
        temp_bias = torch.load(self.save_file + 'CLs_bias' + str(self.task_id - 1) + '.pt')
        self.adapter_vit.head.weight.data = temp_weights.data.cuda()
        self.adapter_vit.head.bias.data = temp_bias.data.cuda()

    def reset(self, eval=False):
        self.__init__(self.base_vit, self.rank, lora_layer=None, eval=eval, index=False)

    def reset_parameters(self) -> None:
        for down in self.adapter_downs:
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        for up in self.adapter_ups:
            nn.init.zeros_(up.weight)

    def save_wrap_param(self, filename):
        if self.per_layer_scaling:
            if self.task_id == 1:
                scaling_param = torch.zeros(self.num_adapter_layers, 20, 20)
            else:
                scaling_param = torch.load(filename + 'scaling_factor' + str(self.task_id - 2) + '.pt')

            i = self.task_id - 1
            for layer_idx in range(self.num_adapter_layers):
                for j in range(i + 1):
                    if j == i:
                        scaling_param[layer_idx][i][j] = self.wrapped_param[layer_idx].param.item()
                    else:
                        scaling_param[layer_idx][i][j] = self.wrapped_param_prev[j][layer_idx].param.item()
            torch.save(scaling_param, filename + 'scaling_factor' + str(self.task_id - 1) + '.pt')
        else:
            if self.task_id == 1:
                scaling_param = torch.zeros(20, 20)
            else:
                scaling_param = torch.load(filename + 'scaling_factor' + str(self.task_id - 2) + '.pt')
            i = self.task_id - 1
            for j in range(i + 1):
                if j == i:
                    scaling_param[i][j] = self.wrapped_param[0].param.item()
                else:
                    scaling_param[i][j] = self.wrapped_param_prev[j].param.item()
            torch.save(scaling_param, filename + 'scaling_factor' + str(self.task_id - 1) + '.pt')

    def save_adapter_parameters(self, filename: str, task_id) -> None:
        """Save current task's adapter parameters and increment task counter."""
        self.task_id += 1
        if not os.path.exists(filename):
            os.makedirs(filename)
        torch.save(self.adapter_downs, filename + 'adapter_down_' + str(task_id) + '.pt')
        torch.save(self.adapter_ups, filename + 'adapter_up_' + str(task_id) + '.pt')

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def load_eval_vit(self):
        """Load the evaluation ViT with all saved adapters."""
        self.adapter_vit = copy.deepcopy(self.base_vit)
        saved_adapter_down, saved_adapter_up = {}, {}
        for i in range(self.task_id):
            file_path = self.save_file + 'adapter_down_' + str(i) + '.pt'
            saved_adapter_down['saved_down_' + str(i)] = torch.load(file_path)
            file_path = self.save_file + 'adapter_up_' + str(i) + '.pt'
            saved_adapter_up['saved_up_' + str(i)] = torch.load(file_path)

        for param in self.adapter_vit.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(self.adapter_vit.blocks):
            w_mlp = blk.mlp
            self.dim = blk.attn.qkv.in_features
            blk.mlp = _Adapter_MLP_eval(
                self.task_id, w_mlp, saved_adapter_down, saved_adapter_up,
                t_layer_i, self.rank, self.dim,
                self.wrapped_param, self.wrapped_param_prev,
                self.save_file, per_layer_scaling=self.per_layer_scaling
            )
        self.reset_adapter_vit_head()

    def compute_ortho_loss(self):
        """Compute orthogonal regularization loss between current and previous adapters."""
        loss = torch.tensor(0).float().cuda()
        for i in range(self.task_id):
            file_path = self.save_file + 'adapter_down_' + str(i) + '.pt'
            if os.path.exists(file_path):
                prev_downs = torch.load(file_path)
                num_layer = len(self.adapter_downs)
                for j in range(num_layer):
                    temp = torch.matmul(
                        prev_downs[j].weight.to(self.adapter_downs[j].weight.device),
                        self.adapter_downs[j].weight.t()
                    )
                    temp = torch.sum(torch.square(temp))
                    loss = loss.to(self.adapter_downs[j].weight.device)
                    loss += temp
        return loss

    def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
        if eval:
            self.reset(eval=True)
            return self.adapter_vit(x)
        elif loss:
            ortho_loss = self.compute_ortho_loss()
            return self.adapter_vit(x), ortho_loss
        else:
            return self.adapter_vit(x)
