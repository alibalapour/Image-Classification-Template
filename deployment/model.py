import torch
from utils import ResNet, get_transform, idx_to_classes


class ResNetInterface:
    def __init__(self, device, input_size, model_path='model.pth', num_classes=10):
        self.device = device
        self.input_size = input_size
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = self.load_model()

    def load_model(self):
        model = ResNet(3, num_classes=self.num_classes, n=1)
        model.load_state_dict(torch.load(self.model_path)['model'])
        model.to(self.device)
        return model

    def inference(self, img):
        """
        get an RGB image (h*w*c) and returns its predicted class
        :param img: input image
        :return p_label: predicted label
        """

        transform = get_transform(image_size=self.input_size, imagenet_normalize=True)
        tensor = transform(img)
        tensor = tensor[None, :, :, :].to(self.device)

        self.model.to(self.device)
        self.model.eval()
        output = self.model(tensor).argmax(dim=1).item()
        p_label = idx_to_classes[output]

        return p_label
