import numpy as np
import torch


def calculate_goodness_distributions(matrix, y_predicted_on_layer, targets):
    mean_all = 0.0
    std_all = 0.0
    mean_all_incorrect_labels = 0.0
    std_all_incorrect_labels = 0.0

    for col_index in range(10):
        indices_correct = np.where((targets == col_index) & (targets == y_predicted_on_layer))
        if indices_correct[0].size == 0:
            continue

        for row_index in range(10):
            values = matrix[indices_correct, row_index][0]
            if values.size == 0:
                continue
            if row_index == col_index:
                mean_all += np.mean(values)
                std_all += np.std(values)
            else:
                mean_all_incorrect_labels += np.mean(values)
                std_all_incorrect_labels += np.std(values)

    mean_all /= 10
    std_all /= 10
    mean_all_incorrect_labels /= 90
    std_all_incorrect_labels /= 90

    return mean_all, std_all, mean_all_incorrect_labels, std_all_incorrect_labels


def analysis_val_set(model, inputs, targets, batch_size=1000):
    num_layers = len(model.layers)
    num_val_samples = inputs.shape[0]
    test_data_record_indices = range(0, num_val_samples)
    num_batches = int(np.ceil(num_val_samples / batch_size))
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)

    y_predicted_on_layer = np.zeros((num_layers, num_val_samples), dtype=np.int64)
    cumulative_goodness_on_layer = np.zeros((num_layers, num_val_samples), dtype=np.float32)
    softmax_output_on_layer = np.zeros((num_layers, num_val_samples, 10), dtype=np.float32)

    for batch_idx in range(num_batches):
        chunk = chunk_indices_validation[batch_idx]
        x_ = inputs[chunk]
        temp_y_predicted_on_layer, temp_cumulative_goodness_on_layer, temp_softmax_output_on_layer = \
            model.light_predict_analysis(x=x_, num_layers=num_layers)

        y_predicted_on_layer[:, chunk] = temp_y_predicted_on_layer
        cumulative_goodness_on_layer[:, chunk] = temp_cumulative_goodness_on_layer
        softmax_output_on_layer[:, chunk, :] = temp_softmax_output_on_layer

    mean = []
    std = []
    target_values = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    for i in range(num_layers):
        temp_mean, temp_std, _, _ = calculate_goodness_distributions(
            softmax_output_on_layer[i, :, :],
            y_predicted_on_layer[i, :],
            target_values,
        )
        mean.append(temp_mean)
        std.append(temp_std)

    return mean, std
