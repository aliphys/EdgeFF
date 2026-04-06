import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from basicRun.data import get_dataset
from basicRun.model import Net, overlay_y_on_x, overlay_on_x_neutral


def dataset_loaders(dataset_name, train_batch_size, test_batch_size, val_size, seed):
    full_train, is_color = get_dataset(dataset_name, train=True)
    test_dataset, _ = get_dataset(dataset_name, train=False)

    if val_size <= 0 or val_size >= len(full_train):
        raise ValueError('val_size must be positive and smaller than the training dataset size')

    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, is_color


def make_negative_labels(targets, num_classes=10):
    neg_labels = targets.clone()
    for i, label in enumerate(targets):
        choices = list(range(num_classes))
        choices.remove(int(label))
        neg_labels[i] = random.choice(choices)
    return neg_labels


def train_representation_epoch(model, train_loader, onehot_max_value, is_color, device, log_interval=10):
    total_train = len(train_loader.dataset)
    for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        x_pos_batch = overlay_y_on_x(batch_inputs, batch_targets, onehot_max_value, is_color)
        y_neg_batch = make_negative_labels(batch_targets)
        x_neg_batch = overlay_y_on_x(batch_inputs, y_neg_batch, onehot_max_value, is_color)

        model.train(x_pos_batch, x_neg_batch)

        if batch_idx % log_interval == 0:
            processed = (batch_idx + 1) * batch_inputs.shape[0]
            pct = processed / total_train
            print(f'Rep Epoch [{processed}/{total_train} ({pct:.0%})]')


def train_softmax_epoch(model, train_loader, is_color, device, layers, log_interval=10):
    total_train = len(train_loader.dataset)
    for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        x_neutral_batch = overlay_on_x_neutral(batch_inputs, is_color)
        model.train_softmax_layer(x_neutral_batch, batch_targets, batch_inputs.shape[0], layers)

        if batch_idx % log_interval == 0:
            processed = (batch_idx + 1) * batch_inputs.shape[0]
            pct = processed / total_train
            print(f'Softmax Epoch [{processed}/{total_train} ({pct:.0%})]')


def quick_accuracy(model, x, y):
    with torch.no_grad():
        preds = model.predict_one_pass(x, batch_size=min(5000, x.shape[0]))
    return (preds.cpu() == y.cpu()).float().mean().item()


def save_model(model, output_dir, name='temp_'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / name
    torch.save(model, model_path)
    return model_path


def build_model(x_pos, x_neg, x_neutral, targets, layers, wandb_run=None, eval_inputs=None, eval_targets=None):
    dims = layers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dims, device=device)
    x_pos = x_pos.to(device)
    x_neg = x_neg.to(device)
    x_neutral = x_neutral.to(device)
    targets = targets.to(device)

    num_epochs = 100
    representation_batch_size = 5000
    softmax_batch_size = 500
    num_train_samples_repr = 50000
    num_train_samples_softmax = 50000

    if eval_inputs is not None and eval_targets is not None:
        eval_subset_inputs = eval_inputs[:1000].to(device)
        eval_subset_targets = eval_targets[:1000].to(device)
    else:
        eval_subset_inputs = None
        eval_subset_targets = None

    for epoch in tqdm(range(num_epochs), desc="Train representation layers"):
        train_data_record_indices = list(range(0, num_train_samples_repr))
        train_data_record_indices_shuffled = shuffle(train_data_record_indices)
        num_batches = int(num_train_samples_repr / representation_batch_size)
        chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)
        for i in range(num_batches):
            x_pos_, x_neg_ = x_pos[chunk_indices[i]], x_neg[chunk_indices[i]]
            model.train(x_pos_, x_neg_)
        if wandb_run and eval_subset_inputs is not None:
            with torch.no_grad():
                preds = model.predict_one_pass(eval_subset_inputs, batch_size=len(eval_subset_inputs))
                sample_val_acc = (preds == eval_subset_targets).float().mean().item()
            wandb_run.log({
                'epoch': epoch,
                'phase': 'representation',
                'sample_val_accuracy': sample_val_acc,
            })

    for epoch in tqdm(range(num_epochs), desc="Train softmax layers"):
        train_data_record_indices = list(range(0, num_train_samples_softmax))
        train_data_record_indices_shuffled = shuffle(train_data_record_indices)
        num_batches = int(num_train_samples_softmax / softmax_batch_size)
        chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)
        for i in range(num_batches):
            x_neutral_, targets_ = x_neutral[chunk_indices[i]], targets[chunk_indices[i]]
            model.train_softmax_layer(x_neutral_, targets_, softmax_batch_size, dims)
        if wandb_run and eval_subset_inputs is not None:
            with torch.no_grad():
                preds = model.predict_one_pass(eval_subset_inputs, batch_size=len(eval_subset_inputs))
                sample_val_acc = (preds == eval_subset_targets).float().mean().item()
            wandb_run.log({
                'epoch': epoch,
                'phase': 'softmax',
                'sample_val_accuracy': sample_val_acc,
            })

    name = 'temp_'
    model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, name)
    torch.save(model, model_path)

    if wandb_run:
        artifact = wandb.Artifact('ff-model', type='model')
        artifact.add_file(model_path)
        wandb_run.log_artifact(artifact)

    return model
