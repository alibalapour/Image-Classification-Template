import time
import datetime
import os
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

from utils import Record
from utils import test_gpu_cuda, prepare_tensorboard, add_image_to_tensorboard, get_optimizer, sam_function, mixup
from model import prepare_model
from dataset import CustomDataset


def train(model, tools, epoch, logs):
    train_dataloader = tools['train_dataloader']
    train_device = tools['train_device']
    opt = tools['opt']
    mix_up = tools['mix_up']
    mix_up_param = tools['mix_up_param']
    criterion = tools['criterion']
    train_writer = tools['train_writer']
    grad_clip = tools['grad_clip']
    tensorboard_log = tools['tensorboard_log']
    scheduler = tools['scheduler']
    checkpoints_every = tools['checkpoints_every']
    result_path = tools['result_path']
    mixed_precision = tools['mixed_precision']
    scaler = tools['scaler']
    sam_option = tools['sam_option']
    train_accuracy = tools['train_accuracy']
    min_lr = tools['min_lr']
    scheduler_final_epoch = tools['scheduler_final_epoch']
    mixup_off_epoch = tools['mixup_off_epoch']

    model.train()
    train_loss = []
    running_loss = 0.0
    step = 0

    for tensors, labels in train_dataloader:  # training loop
        tensors = tensors.to(train_device)
        labels = labels.to(train_device)
        opt.zero_grad()

        tensors, labels = mixup(tensors, labels, torch.distributions.beta.Beta(mix_up_param,
                                                                               mix_up_param).sample().item()) if mix_up and epoch < mixup_off_epoch else (
            tensors, labels)

        # to save some samples from mixup augmentation
        if mix_up and step == 0 and epoch == 0:
            add_image_to_tensorboard(tensors.cpu(), train_writer, 'training images (with mixup)')

        if sam_option:
            loss, outputs = sam_function(model, tensors, labels, criterion, scaler,
                                         mixed_precision, opt, grad_clip)
        else:
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(tensors)
                loss = criterion(outputs, labels)

        if not sam_option:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

        train_loss.append(loss.item())
        train_accuracy.update(outputs.argmax(dim=1), labels.argmax(dim=1))
        running_loss += loss.item()

        step += 1
        if tensorboard_log and step % 5 == 0:
            train_writer.add_scalar('Training Step loss',
                                    loss.item(),
                                    epoch * len(train_dataloader) + step,
                                    new_style=True)
            train_writer.add_scalar('LR',
                                    scheduler.optimizer.param_groups[0]['lr'],
                                    epoch * len(train_dataloader) + step,
                                    new_style=True)

        # to stop scheduler after specific epoch
        if epoch < scheduler_final_epoch:
            scheduler.step()
        else:
            opt.param_groups[0]['lr'] = min_lr

    accuracy = train_accuracy.compute().item() * 100
    logs['train_loss'] = sum(train_loss) / len(train_loss)
    logs['train_acc'] = accuracy
    logs['lr'] = scheduler.optimizer.param_groups[0]['lr']

    if tensorboard_log:
        train_writer.add_scalar('Loss',
                                sum(train_loss) / len(train_loss),
                                epoch,
                                new_style=True)
        train_writer.add_scalar('ACC',
                                accuracy,
                                epoch,
                                new_style=True)
    if epoch % checkpoints_every == 0:
        torch.save(model.state_dict(), os.path.join(result_path, 'model_epoch_' + str(epoch) + '.pth'))
        torch.save({
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch
        }, os.path.join(result_path, 'model_epoch_' + str(epoch) + '.pth'))

    train_accuracy.reset()


def valid(model, tools, epoch, logs, recorder):
    valid_dataloader = tools['valid_dataloader']
    valid_device = tools['valid_device']
    opt = tools['opt']
    criterion = tools['criterion']
    tensorboard_log = tools['tensorboard_log']
    valid_writer = tools['valid_writer']
    valid_accuracy = tools['valid_accuracy']
    valid_precision = tools['valid_precision']
    valid_recall = tools['valid_recall']
    valid_f1 = tools['valid_f1']

    valid_loss = []
    model.eval()
    for tensors, labels in valid_dataloader:
        tensors = tensors.to(valid_device)
        labels = labels.to(valid_device)
        opt.zero_grad()

        outputs = model(tensors)
        loss = criterion(outputs, labels)
        predicted_labels = outputs.argmax(dim=1)
        true_labels = labels.argmax(dim=1)

        valid_accuracy.update(predicted_labels, true_labels)
        valid_precision.update(predicted_labels, true_labels)
        valid_recall.update(predicted_labels, true_labels)
        valid_f1.update(predicted_labels, true_labels)
        valid_loss.append(loss.item())

    accuracy = valid_accuracy.compute().item() * 100
    precision = valid_precision.compute().item() * 100
    recall = valid_recall.compute().item() * 100
    f1 = valid_f1.compute().item() * 100

    logs['valid_acc'] = accuracy
    logs['valid_loss'] = sum(valid_loss) / len(valid_loss)
    recorder.update_df_max_value('maximum_val_acc', accuracy, 'maximum_val_acc_epoch', epoch)

    if tensorboard_log:
        valid_writer.add_scalar('ACC',
                                accuracy,
                                epoch,
                                new_style=True)
        valid_writer.add_scalar('Validation Precision',
                                precision,
                                epoch,
                                new_style=True)
        valid_writer.add_scalar('Validation Recall',
                                recall,
                                epoch,
                                new_style=True)
        valid_writer.add_scalar('Validation F1',
                                f1,
                                epoch,
                                new_style=True)
        valid_writer.add_scalar('Loss',
                                sum(valid_loss) / len(valid_loss),
                                epoch,
                                new_style=True)

    valid_accuracy.reset()
    valid_precision.reset()
    valid_recall.reset()
    valid_f1.reset()


def main(config):
    test_gpu_cuda()

    pretrained_weights = config['pretrain']['pretrained_weights']
    pretrained_weights_path = config['pretrain']['pretrained_weights_path']
    resume = config['resume']['resume']
    resume_path = config['resume']['resume_path']
    fix_seed = config['fix_seed']
    result_path = config['result_path']
    checkpoints_every = config['checkpoints_every']
    tensorboard_log = config['tensorboard_log']

    train_path = config['train_settings']['train_path']
    train_batch_size = config['train_settings']['train_batch_size']
    num_epoch = config['train_settings']['num_epochs']
    train_shuffle = config['train_settings']['shuffle']
    target_size = (config['train_settings']['w_input'], config['train_settings']['h_input'])
    mixed_precision = config['train_settings']['mixed_precision']
    train_device = str(config['train_settings']['device'])

    valid_path = str(config['valid_settings']['valid_path'])
    valid_batch_size = config['valid_settings']['valid_batch_Size']
    valid_every = config['valid_settings']['do_every']
    valid_device = str(config['valid_settings']['device'])

    model_name = str(config['model']['model_name'])

    opt_name = str(config['optimizer']['name'])
    learning_rate = float(config['optimizer']['lr'])
    sam_option = config['optimizer']['sam']
    weight_decay = float(config['optimizer']['weight_decay'])
    weight_decouple = config['optimizer']['weight_decouple']  # ada-belief ???
    eps = float(config['optimizer']['eps'])
    grad_clip = float(config['optimizer']['grad_clip_norm'])

    warmup = int(config['optimizer']['decay']['warmup'])
    min_lr = float(config['optimizer']['decay']['min_lr'])
    gamma = float(config['optimizer']['decay']['gamma'])
    scheduler_final_epoch = int(config['optimizer']['decay']['final_epoch'])

    imagenet_normalize = config['augmentation']['imagenet_normalize']
    mix_up = config['augmentation']['mixup']['mixup']
    mix_up_param = float(config['augmentation']['mixup']['mixup_param'])
    mixup_off_epoch_ratio = float(config['augmentation']['mixup']['mixup_off_epoch_ratio'])
    rand_aug = int(config['augmentation']['rand_aug']['rand_aug'])
    rand_aug_num_ops = int(config['augmentation']['rand_aug']['rand_aug_num_ops'])
    rand_aug_magnitude = int(config['augmentation']['rand_aug']['rand_aug_magnitude'])
    crop_scale = float(config['augmentation']['others']['crop_scale'])
    jitter_param = float(config['augmentation']['others']['jitter_param'])
    erasing_prob = float(config['augmentation']['others']['erasing_prob'])

    # initialization
    if fix_seed:
        torch.manual_seed(0)
    Path(result_path).mkdir(parents=True, exist_ok=True)

    # to save tensorboard logs and models in a same folder. Also, we use this id in records dataframe
    run_id = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    result_path = os.path.join(result_path, run_id)
    Path(result_path).mkdir(parents=True, exist_ok=True)

    # dataset pipeline (in dataset.py)
    train_dataset = CustomDataset(train_path, target_size, is_train=True, rand_aug=rand_aug,
                                  rand_aug_num_ops=rand_aug_num_ops, rand_aug_magnitude=rand_aug_magnitude,
                                  imagenet_normalize=imagenet_normalize, crop_scale=crop_scale,
                                  jitter_param=jitter_param, erasing_prob=erasing_prob)
    valid_dataset = CustomDataset(valid_path, target_size, is_train=False, rand_aug=False)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle, num_workers=2,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=2,
                                  pin_memory=True)

    # preparing model (in model.py)
    model = prepare_model(num_classes=len(train_dataset.idx_to_class), model_name=model_name)
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights_path)['model'])
        print('Model is loaded!')

    model.train()
    criterion = nn.CrossEntropyLoss()
    opt, scheduler = get_optimizer(eps, gamma, learning_rate, min_lr, model, opt_name, sam_option,
                                   warmup, weight_decay, weight_decouple, train_dataloader, scheduler_final_epoch)

    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            opt.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        print('Model is loaded to resume training!')

    # initialize tensorboards
    train_writer, valid_writer = None, None
    if tensorboard_log:
        train_writer, valid_writer = prepare_tensorboard(run_id, model, train_dataloader)

    # preparing training loop (here in train.py)
    model.to(train_device)
    tools = {
        'pretrained_weights': pretrained_weights,
        'train_dataloader': train_dataloader,
        'train_device': train_device,
        'opt': opt,
        'mix_up': mix_up,
        'mixup_off_epoch': mixup_off_epoch_ratio * num_epoch,
        'mix_up_param': mix_up_param,
        'criterion': criterion,
        'grad_clip': grad_clip,
        'tensorboard_log': tensorboard_log,
        'scheduler': scheduler,
        'scheduler_final_epoch': scheduler_final_epoch,
        'train_dataset': train_dataset,
        'checkpoints_every': checkpoints_every,
        'result_path': result_path,
        'valid_dataloader': valid_dataloader,
        'valid_device': valid_device,
        'train_writer': train_writer,
        'valid_writer': valid_writer,
        'mixed_precision': mixed_precision,
        'sam_option': sam_option,
        'train_accuracy': Accuracy().to(train_device),
        'valid_accuracy': Accuracy().to(valid_device),
        'valid_precision': Precision(num_classes=len(valid_dataset.idx_to_class), average='macro').to(valid_device),
        'valid_recall': Recall(num_classes=len(valid_dataset.idx_to_class), average='macro').to(valid_device),
        'valid_f1': F1Score(num_classes=len(valid_dataset.idx_to_class), average='macro').to(valid_device),
        'scaler': torch.cuda.amp.GradScaler(enabled=mixed_precision),
        'num_epoch': num_epoch,
        'min_lr': min_lr
    }

    recorder = Record(tools, run_id, train_batch_size, valid_batch_size, num_epoch, target_size[0], opt_name,
                      learning_rate, weight_decouple, weight_decay, eps, warmup, gamma, min_lr, rand_aug,
                      rand_aug_num_ops, rand_aug_magnitude, imagenet_normalize, model)

    print("Number of parameters :", sum(p.numel() for p in model.parameters()))
    logs = {}
    for epoch in range(start_epoch, num_epoch):  # main loop
        epoch_start_time = time.time()
        train(model, tools, epoch=epoch, logs=logs)
        if epoch % valid_every == 0:
            valid(model, tools, epoch, logs, recorder)
        logs['time'] = time.time() - epoch_start_time

        print(
            f'Epoch: {epoch} | Train Loss: {logs["train_loss"]:.4f}, Train ACC: {logs["train_acc"]:.4f}, Validation Loss: {logs["valid_loss"]:.4f},'
            f' Validation ACC: {logs["valid_acc"]:.4f}, Time: {logs["time"]:.1f}, lr: {logs["lr"]:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train the CURL neural network on image pairs")

    parser.add_argument(
        "--config_path", "-c", help="The location of curl config file", default='./config.yaml')

    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file)
