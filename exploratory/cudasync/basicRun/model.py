"""Forward-Forward model implementation."""

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


def overlay_y_on_x(x, y, max_value=10.0, is_color=False):
    x_ = x.clone()
    if is_color:
        pixels_per_channel = x_.shape[1] // 3
        x_[:, :10] *= 0.0
        x_[range(x.shape[0]), y] = max_value
        x_[:, pixels_per_channel:pixels_per_channel+10] *= 0.0
        x_[range(x.shape[0]), pixels_per_channel + y] = max_value
        x_[:, 2*pixels_per_channel:2*pixels_per_channel+10] *= 0.0
        x_[range(x.shape[0]), 2*pixels_per_channel + y] = max_value
    else:
        x_[:, :10] *= 0.0
        x_[range(x.shape[0]), y] = max_value
    return x_


def overlay_on_x_neutral(x, is_color=False):
    x_ = x.clone()
    if is_color:
        pixels_per_channel = x_.shape[1] // 3
        x_[:, :10] *= 0.0
        x_[:, :10] = 0.1
        x_[:, pixels_per_channel:pixels_per_channel+10] *= 0.0
        x_[:, pixels_per_channel:pixels_per_channel+10] = 0.1
        x_[:, 2*pixels_per_channel:2*pixels_per_channel+10] *= 0.0
        x_[:, 2*pixels_per_channel:2*pixels_per_channel+10] = 0.1
    else:
        x_[:, :10] *= 0.0
        x_[:, :10] = 0.1
    return x_


class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_iterations = 1

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for _ in range(self.num_iterations):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([ -g_pos + self.threshold, g_neg - self.threshold ]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class SoftmaxLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.softmax_l = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.softmax_l.weight)
        self.softmax = torch.nn.Softmax(dim=1)
        self.opt = Adam(self.parameters(), lr=0.03)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        output_l = self.softmax_l(x)
        output = self.softmax(output_l)
        return output_l, output

    @staticmethod
    def to_categorical(y, num_classes):
        return np.eye(num_classes, dtype='uint8')[y]

    def train(self, x, y):
        self.opt.zero_grad()
        yhat, _ = self.forward(x)
        loss = self.criterion(yhat, y)
        loss.backward()
        self.opt.step()


class Net(torch.nn.Module):
    def __init__(self, dims, device=None, onehot_max_value=10.0, is_color=False):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.onehot_max_value = onehot_max_value
        self.is_color = is_color
        self.layers = nn.ModuleList([Layer(dims[d], dims[d + 1]).to(self.device) for d in range(len(dims) - 1)])
        self.softmax_layers = nn.ModuleList([
            SoftmaxLayer(in_features=sum(dims[1:(d + 2)]), out_features=10).to(self.device)
            for d in range(len(dims) - 1)
        ])

    def predict_one_pass(self, x, batch_size):
        x = x.to(self.device)
        h = overlay_on_x_neutral(x, self.is_color)
        softmax_layer_input = None
        for i, (layer, softmax_layer) in enumerate(zip(self.layers, self.softmax_layers), start=0):
            h = layer(h)
            if softmax_layer_input is None:
                softmax_layer_input = h
            else:
                softmax_layer_input = torch.cat((softmax_layer_input, h), 1)
            if i == len(self.layers) - 1:
                _, softmax_layer_output = softmax_layer(softmax_layer_input)
        return softmax_layer_output.argmax(1)

    def check_confidence(self, layer_num, confidence_mean_vec, confidence_std_vec, softmax_layer_output_l):
        threshold = confidence_mean_vec[layer_num] - confidence_std_vec[layer_num]
        return torch.max(softmax_layer_output_l) > threshold

    def light_predict_one_sample(self, x, confidence_mean_vec, confidence_std_vec):
        x = x.to(self.device)
        h = overlay_on_x_neutral(x, self.is_color)
        confidence_flag = False
        predicted_with_layers_up_to = 0
        softmax_layer_input = None
        for i, (layer, softmax_layer) in enumerate(zip(self.layers, self.softmax_layers), start=0):
            if not confidence_flag:
                predicted_with_layers_up_to += 1
                h = layer(h)
                if softmax_layer_input is None:
                    softmax_layer_input = h
                else:
                    softmax_layer_input = torch.cat((softmax_layer_input, h), 1)
                softmax_layer_output_l, softmax_layer_output = softmax_layer(softmax_layer_input)
                confidence_flag = self.check_confidence(
                    layer_num=i,
                    confidence_mean_vec=confidence_mean_vec,
                    confidence_std_vec=confidence_std_vec,
                    softmax_layer_output_l=softmax_layer_output_l,
                )
        return softmax_layer_output.argmax(1), predicted_with_layers_up_to

    def light_predict_analysis(self, x, num_layers):
        x = x.to(self.device)
        num_samples = x.shape[0]
        y_predicted_on_layer = np.zeros((num_layers, num_samples))
        cumulative_goodness_on_layer = np.zeros((num_layers, num_samples))
        softmax_output_on_layer = np.zeros((num_layers, num_samples, 10))
        h = overlay_on_x_neutral(x, self.is_color)
        softmax_layer_input = None
        for i, (layer, softmax_layer) in enumerate(zip(self.layers, self.softmax_layers), start=0):
            h = layer(h)
            if softmax_layer_input is None:
                softmax_layer_input = h
            else:
                softmax_layer_input = torch.cat((softmax_layer_input, h), 1)
            for j in range(i, num_layers):
                cumulative_goodness_on_layer[j, :] += h.pow(2).mean(1).detach().cpu().numpy()
            softmax_layer_output_l, softmax_layer_output = softmax_layer(softmax_layer_input)
            y_predicted_on_layer[i, :] = softmax_layer_output.argmax(1).cpu().numpy()
            softmax_output_on_layer[i, :, :] = softmax_layer_output_l.detach().cpu().numpy()
        return y_predicted_on_layer, cumulative_goodness_on_layer, softmax_output_on_layer

    def train(self, x_pos, x_neg):
        x_pos = x_pos.to(self.device)
        x_neg = x_neg.to(self.device)
        h_pos, h_neg = x_pos, x_neg
        for layer in self.layers:
            h_pos, h_neg = layer.train(h_pos, h_neg)

    def train_softmax_layer(self, x_neutral_label, y, batch_size, dims):
        x_neutral_label = x_neutral_label.to(self.device)
        y = y.to(self.device)
        for d, softmax_layer in enumerate(self.softmax_layers, start=0):
            h_neutral_label = x_neutral_label
            num_input_features = sum(dims[1:(d + 2)])
            softmax_layer_input = torch.empty((batch_size, num_input_features), device=self.device)
            for i, layer in islice(enumerate(self.layers), 0, (d + 1)):
                h_neutral_label = layer.forward(h_neutral_label)
                index_start = sum(dims[1:(i + 1)])
                index_end = index_start + dims[i + 1]
                softmax_layer_input[:, index_start:index_end] = h_neutral_label
            softmax_layer.train(softmax_layer_input, y)
