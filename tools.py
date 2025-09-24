import numpy as np

import torch
from Evaluation import print_results


def calculate_goodness_distributions(matrix, y_predicted_on_layer, targets, num_layers):  # matrix is softmax_output_on_layer

    mean_all = 0
    std_all = 0
    valid_correct_count = 0

    mean_all_incorrect_labels = 0
    std_all_incorrect_labels = 0
    valid_incorrect_count = 0

    # Get unique classes present in the dataset
    unique_classes = np.unique(targets)

    for col_index in unique_classes:  # Only iterate over classes present in the dataset
        for row_index in range(10):

            indices_correct = np.where((targets == col_index) & (targets == y_predicted_on_layer))

            indices = indices_correct

            # Check if we have any samples for this combination
            if len(indices[0]) > 0:
                data_slice = matrix[indices, row_index][0]
                if len(data_slice) > 0:  # Additional check for non-empty data
                    if row_index == col_index:
                        mean_all += np.mean(data_slice)
                        std_all += np.std(data_slice)
                        valid_correct_count += 1
                    else:
                        mean_all_incorrect_labels += np.mean(data_slice)
                        std_all_incorrect_labels += np.std(data_slice)
                        valid_incorrect_count += 1

    # Only divide if we have valid samples
    if valid_correct_count > 0:
        mean_all /= valid_correct_count
        std_all /= valid_correct_count
    else:
        mean_all = 0
        std_all = 0

    if valid_incorrect_count > 0:
        mean_all_incorrect_labels /= valid_incorrect_count
        std_all_incorrect_labels /= valid_incorrect_count
    else:
        mean_all_incorrect_labels = 0
        std_all_incorrect_labels = 0

    print("Averaged mean: ", mean_all)
    print("Averaged std: ", std_all)

    print("Averaged mean_all_incorrect_labels: ", mean_all_incorrect_labels)
    print("Averaged std_all_incorrect_labels: ", std_all_incorrect_labels)
    return mean_all, std_all



def analysis_val_set(model, inputs, targets):
    num_layers = len(model.layers)
    # val set - use actual number of samples instead of fixed 10000
    num_val_samples = len(inputs)
    test_data_record_indices = range(0, num_val_samples)

    batch_size = min(1000, num_val_samples)  # Ensure batch size doesn't exceed sample count
    num_batches = int(np.ceil(num_val_samples / batch_size))  # Use ceil to handle remaining samples
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)
    y_predicted_on_layer = np.zeros((num_layers, num_val_samples))
    cumulative_goodness_on_layer = np.zeros((num_layers, num_val_samples))
    softmax_output_on_layer = np.zeros((num_layers, num_val_samples, 10))  # 10 is the number of softmax neurons

    for i in range(num_batches):
        x_ = inputs[chunk_indices_validation[i]]

        temp_y_predicted_on_layer, temp_cumulative_goodness_on_layer, temp_softmax_output_on_layer = \
            model.light_predict_analysis(x=x_, num_layers=num_layers)

        y_predicted_on_layer[:, chunk_indices_validation[i]] = temp_y_predicted_on_layer#.detach().cpu().numpy()

        cumulative_goodness_on_layer[:, chunk_indices_validation[i]] = \
            temp_cumulative_goodness_on_layer#.detach().cpu().numpy()

        softmax_output_on_layer[:, chunk_indices_validation[i], :] = temp_softmax_output_on_layer

    mean = []
    std =  []
    # t = targets.detach().cpu().numpy()
    for i in range(num_layers):
        temp_mean, temp_std = calculate_goodness_distributions(softmax_output_on_layer[i, :, :], y_predicted_on_layer[i, :], targets.detach().cpu().numpy(), num_layers)
        mean.append(temp_mean)
        std.append(temp_std)

    return mean, std
    #exit()
