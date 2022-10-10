from torchvision import transforms
from torchvision import models
import torch
import numpy as np


def image_preprocessing(input_image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    return preprocess(input_image)


def get_prediction(input_image, image_model):
    if image_model=='densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    else:
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.eval()
    input_tensor = image_preprocessing(input_image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    from urllib.request import urlopen
    response = urlopen('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt')
    data = response.read()
    txt_str = str(data)
    lines = txt_str.split("\\n")
    top_prob, top_catig = torch.topk(probabilities, 1)
    return np.round(float(top_prob.item()), 4), lines[top_catig]