import os
from pathlib import Path
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from adabelief_pytorch import AdaBelief
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Tuple
import random
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision import transforms
import pandas as pd


def test_gpu_cuda():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    print()


def prepare_tensorboard(run_id, model, dataloader):
    """
    Initialize tensorboard writers for capturing training and validation stats.
    :return: a writer for training and a writer for validation
    """
    Path('runs').mkdir(parents=True, exist_ok=True)
    train_path = os.path.join('runs', 'train')
    val_path = os.path.join('runs', 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, run_id)
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, run_id)
    val_writer = SummaryWriter(val_log_path)

    train_writer.add_graph(model.to('cpu'), torch.rand((1, 3, 32, 32)))

    # display a sample of images from train_dataloader
    images = None
    for images, labels in dataloader:
        break
    add_image_to_tensorboard(images, train_writer, 'training images (no mixup)')

    return train_writer, val_writer


# adopted from pytorch.org
def matplotlib_imshow(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)).astype('uint8'))


def reverse_normalization(images):
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    un_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return un_normalize(images)


def add_image_to_tensorboard(images, writer, description='images'):
    images = reverse_normalization(images)
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid)
    writer.add_image(description, img_grid)


def inference_display(dataset, inputs, true_labels, predicted_labels, description):
    sample_idx = random.sample(list(range(len(inputs))), 16)
    images = np.array(
        [np.transpose(reverse_normalization(torch.tensor(inputs[i])).numpy(), (1, 2, 0)) for i in sample_idx])

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(4, 4),
                     axes_pad=0.4,
                     )
    for idx, (ax, im) in enumerate(zip(grid, images)):
        ax.set_title('T:' + dataset.idx_to_class[true_labels[sample_idx[idx]]] + ' - P:' + dataset.idx_to_class[
            predicted_labels[sample_idx[idx]]], fontsize=8)
        ax.set_axis_off()
        ax.imshow(im)
    plt.title(description)
    plt.show()


def get_optimizer(eps, gamma, learning_rate, min_lr, model, opt_name, sam_option, warmup, weight_decay,
                  weight_decouple, train_dataloader, scheduler_final_epoch):
    if sam_option:
        if opt_name == 'AdaBelief':
            base_opt = AdaBelief
            opt = SAM(model.parameters(), base_opt, adaptive=True, lr=learning_rate, eps=eps, weight_decay=weight_decay,
                      weight_decouple=weight_decouple, rectify=False)
        elif opt_name == 'SGD':
            base_opt = torch.optim.SGD
            opt = SAM(model.parameters(), base_opt, adaptive=True, lr=learning_rate, weight_decay=weight_decay,
                      momentum=0.9,
                      dampening=0, nesterov=True)
        else:
            base_opt = eval('torch.optim.' + opt_name)
            opt = SAM(model.parameters(), base_opt, adaptive=True, lr=learning_rate, eps=eps, weight_decay=weight_decay)

    else:
        if opt_name == 'AdaBelief':
            opt = AdaBelief(model.parameters(), lr=learning_rate, eps=eps,
                            weight_decouple=weight_decouple, rectify=False)
        elif opt_name == 'SGD':
            opt = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9,
                                  dampening=0, nesterov=True)
        else:
            opt = eval('torch.optim.' + opt_name)(model.parameters(), lr=learning_rate, eps=eps,
                                                  weight_decay=weight_decay)

    scheduler = CosineAnnealingWarmupRestarts(opt.base_optimizer if sam_option else opt,
                                              first_cycle_steps=scheduler_final_epoch * len(train_dataloader),
                                              cycle_mult=0.0,
                                              max_lr=learning_rate, min_lr=min_lr,
                                              warmup_steps=warmup * len(train_dataloader), gamma=gamma)

    return opt, scheduler


# adopted from https://github.com/moskomule/mixup.pytorch
def partial_mixup(input_tensors: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input_tensors.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input_tensors[indices]
    return input_tensors.mul(gamma).add(perm_input, alpha=1 - gamma)


# adopted from https://github.com/moskomule/mixup.pytorch
def mixup(input_tensors: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input_tensors.size(0), device=input_tensors.device, dtype=torch.long)
    return partial_mixup(input_tensors, gamma, indices), partial_mixup(target, gamma, indices)


# related to sam_function()
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


# related to sam_function()
def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


# adopted from https://github.com/davda54/sam/issues/7
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False, mixed_precision=False):
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"

            if zero_grad:
                self.zero_grad()

    @torch.no_grad()
    def second_step(self, mixed_precision=False):
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step(closure)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# adopted from https://github.com/davda54/sam/issues/7
def sam_function(model, tensors, labels, criterion, scaler, mixed_precision,
                 sam_optimizer, grad_clip):
    enable_running_stats(model)

    with torch.cuda.amp.autocast(enabled=mixed_precision):
        preds_first = model(tensors)
        loss = criterion(preds_first, labels)

    scaler.scale(loss).backward()
    scaler.unscale_(sam_optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if mixed_precision:
        optimizer_state = scaler._per_optimizer_states[id(sam_optimizer)]
        inf_grad_cnt = sum(v.item() for v in optimizer_state["found_inf_per_device"].values())
        if inf_grad_cnt == 0:
            sam_optimizer.first_step(zero_grad=True, mixed_precision=mixed_precision)
            sam_first_step_applied = True
        else:
            sam_optimizer.zero_grad()
            sam_first_step_applied = False
    else:
        sam_optimizer.first_step(zero_grad=True, mixed_precision=mixed_precision)
        sam_first_step_applied = True

    scaler.update()

    disable_running_stats(model)

    with torch.cuda.amp.autocast(enabled=mixed_precision):
        outputs = model(tensors)
        second_loss = loss = criterion(outputs, labels)

    scaler.scale(second_loss).backward()
    scaler.unscale_(sam_optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if sam_first_step_applied:  # If sam_first_step was applied, apply the 2nd step
        sam_optimizer.second_step(mixed_precision=mixed_precision)
    scaler.step(sam_optimizer)
    scaler.update()

    return loss, outputs


class Record:
    def __init__(self, tools, run_id, train_batch_size, valid_batch_size, num_epoch, input_size, opt_name,
                 learning_rate, weight_decouple, weight_decay, eps, warmup, gamma, min_lr, rand_aug, rand_aug_num_ops,
                 rand_aug_magnitude, imagenet_normalize, model, path='Training Records.csv'):
        self.path = path
        try:
            self.df = pd.read_csv(self.path)
        except FileNotFoundError:
            self.df = pd.DataFrame(data={
                'date': run_id,
                'pretrained_weights': [],
                'train_batch_size': [],
                'valid_batch_Size': [],
                'num_epoch': [],
                'input_size': [],
                'mixed_precision': [],
                'optimizer': [],
                'max_lr': [],
                'min_lr': [],
                'sam': [],
                'weight_decouple': [],
                'weight_decay': [],
                'eps': [],
                'grad_clip_norm': [],
                'warmup': [],
                'gamma': [],
                'parameters': [],
                'mixup': [],
                'mixup_param': [],
                'rand_aug': [],
                'rand_aug_num_ops': [],
                'rand_aug_magnitude': [],
                'imagenet_normalize': [],
                'maximum_val_acc': [],
                'maximum_val_acc_epoch': [],
            })
        new_row = {
            'date': run_id,
            'pretrained_weights': tools['pretrained_weights'],
            'train_batch_size': train_batch_size,
            'valid_batch_Size': valid_batch_size,
            'num_epoch': num_epoch,
            'input_size': input_size,
            'mixed_precision': tools['mixed_precision'],
            'optimizer': opt_name,
            'max_lr': learning_rate,
            'sam': tools['sam_option'],
            'weight_decouple': weight_decouple,
            'weight_decay': weight_decay,
            'eps': eps,
            'grad_clip_norm': tools['grad_clip'],
            'warmup': warmup,
            'min_lr': min_lr,
            'gamma': gamma,
            'mixup': tools['mix_up'],
            'mixup_param': tools['mix_up_param'],
            'rand_aug': rand_aug,
            'rand_aug_num_ops': rand_aug_num_ops,
            'rand_aug_magnitude': rand_aug_magnitude,
            'imagenet_normalize': imagenet_normalize,
            'parameters': sum(p.numel() for p in model.parameters()),
            'maximum_val_acc': 0.0,
            'maximum_val_acc_epoch': 0
        }
        self.df.loc[len(self.df)] = new_row
        self.save_df()

    def save_df(self):
        self.df.to_csv(self.path, index=False)

    def load_df(self):
        self.df = pd.read_csv(self.path)

    def update_df(self, col, value):
        self.load_df()
        self.df.loc[len(self.df) - 1, col] = value
        self.save_df()

    def update_df_max_value(self, col, value, epoch_col, epoch):
        self.load_df()
        if value > self.df.loc[len(self.df) - 1][col]:
            self.df.loc[len(self.df) - 1, col] = value
            self.df.loc[len(self.df) - 1, epoch_col] = epoch
            self.save_df()


# just for test :)
if __name__ == '__main__':
    pass
