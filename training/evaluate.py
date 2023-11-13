import time
import argparse
from torch.utils.data import DataLoader
import torch
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import numpy as np

from utils import inference_display
from model import prepare_model
from dataset import CustomDataset


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mixed-precision', default=True, type=bool)

    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--target-size', default=32, type=int)
    parser.add_argument('--test-path', default='../../cifar10/test', type=str)

    parser.add_argument('--model-path', default='./model.pth', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model-name', default='ResNet', type=str)
    parser.add_argument('--conv-mixer-dim', default=256, type=str)
    parser.add_argument('--conv-mixer-depth', default=8, type=str)
    parser.add_argument('--conv-mixer-kernel-size', default=9, type=str)
    parser.add_argument('--conv-mixer-patch-size', default=2, type=str)

    arguments = parser.parse_args()
    return arguments


def main(args):
    test_dataset = CustomDataset(args.test_path, args.target_size, is_train=False, rand_aug=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                 pin_memory=True)

    conv_mixer_params = {}
    if args.model_name == 'ConvMixer':
        conv_mixer_params.conv_mixer_dim = args.conv_mixer_dim
        conv_mixer_params.conv_mixer_depth = args.conv_mixer_depth
        conv_mixer_params.conv_mixer_kernel_size = args.conv_mixer_kernel_size
        conv_mixer_params.conv_mixer_patch_size = args.conv_mixer_patch_size

    model = prepare_model(num_classes=len(test_dataset.idx_to_class), model_name=args.model_name)
    model.load_state_dict(torch.load(args.model_path)['model'])
    model.to(args.device)

    valid_accuracy = Accuracy().to(args.device)
    valid_precision = Precision(num_classes=len(test_dataset.idx_to_class), average='macro').to(args.device)
    valid_recall = Recall(num_classes=len(test_dataset.idx_to_class), average='macro').to(args.device)
    valid_f1 = F1Score(num_classes=len(test_dataset.idx_to_class), average='macro').to(args.device)

    whole_inputs = []
    whole_true_labels = []
    whole_predicted_labels = []

    start_time = time.time()
    model.eval()
    for tensors, labels in test_dataloader:
        tensors = tensors.to(args.device)
        labels = labels.to(args.device)

        outputs = model(tensors)

        predicted_labels = outputs.argmax(dim=1)
        true_labels = labels.argmax(dim=1)

        valid_accuracy.update(predicted_labels, true_labels)
        valid_precision.update(predicted_labels, true_labels)
        valid_recall.update(predicted_labels, true_labels)
        valid_f1.update(predicted_labels, true_labels)

        whole_inputs += tensors.cpu().numpy().tolist()
        whole_true_labels += true_labels.cpu().numpy().tolist()
        whole_predicted_labels += predicted_labels.cpu().numpy().tolist()

    end_time = time.time()

    whole_inputs = np.array(whole_inputs)
    whole_true_labels = np.array(whole_true_labels)
    whole_predicted_labels = np.array(whole_predicted_labels)

    print(
        f'Inference | Test ACC: {valid_accuracy.compute().item() * 100:.4f}, Test Precision: {valid_precision.compute().item() * 100:.4f}'
        f', Test Recall: {valid_recall.compute().item() * 100:.4f}, Test f1: {valid_f1.compute().item() * 100:.4f}, Time: {end_time - start_time:.1f}')

    inference_display(test_dataset, whole_inputs, whole_true_labels, whole_predicted_labels,
                      description='Samples of test data (T is true labels & P is predicted ones)')
    inference_display(test_dataset, whole_inputs[whole_true_labels != whole_predicted_labels],
                      whole_true_labels[whole_true_labels != whole_predicted_labels],
                      whole_predicted_labels[whole_true_labels != whole_predicted_labels],
                      description='Samples of misclassified test data (T is true labels & P is predicted ones)')

    valid_accuracy.reset()
    valid_precision.reset()
    valid_recall.reset()
    valid_f1.reset()


if __name__ == '__main__':
    args = get_parser()

    main(args)
