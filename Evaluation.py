import torch

from sklearn.metrics import f1_score, accuracy_score
import numpy as np

from tqdm import tqdm


def print_results(labels_vec, predictions_vec):
    f1_performance = f1_score(labels_vec, predictions_vec, average='macro')
    acc_performance = accuracy_score(labels_vec, predictions_vec)

    print("\tF1-score: ", f1_performance)
    print("\tAccuracy: ", acc_performance)


def eval_train_set(model, inputs, targets):
    # train set - use actual number of samples instead of fixed 50000
    num_train_samples = len(inputs)
    train_data_record_indices = range(0, num_train_samples)

    batch_size = min(5000, num_train_samples)  # Ensure batch size doesn't exceed sample count
    num_batches = int(np.ceil(num_train_samples / batch_size))  # Use ceil to handle remaining samples
    chunk_indices = np.array_split(train_data_record_indices, num_batches)

    y_predicted = np.zeros(num_train_samples)
    for i in range(num_batches):
        # x_pos_ = inputs[chunk_indices[i]]
        # y_predicted[chunk_indices[i]] = model.predict(x_pos_).detach().cpu().numpy()
        x_ = inputs[chunk_indices[i]]
        y_predicted[chunk_indices[i]] = model.predict_one_pass(x_, batch_size=len(chunk_indices[i])).detach().cpu().numpy()

    print("\nResults for the {}TRAIN{} set: ".format('\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())


def eval_test_set(model, inputs, targets):
    # test set - use actual number of samples instead of fixed 10000
    num_test_samples = len(inputs)
    test_data_record_indices = range(0, num_test_samples)

    batch_size = min(5000, num_test_samples)  # Ensure batch size doesn't exceed sample count
    num_batches = int(np.ceil(num_test_samples / batch_size))  # Use ceil to handle remaining samples
    chunk_indices_test = np.array_split(test_data_record_indices, num_batches)
    y_predicted = np.zeros(num_test_samples)
    for i in range(num_batches):
        # x_pos_ = inputs[chunk_indices_test[i]]
        # y_predicted[chunk_indices_test[i]] = model.predict(x_pos_).detach().cpu().numpy()
        x_ = inputs[chunk_indices_test[i]]
        y_predicted[chunk_indices_test[i]] = model.predict_one_pass(x_, batch_size=len(chunk_indices_test[i])).detach().cpu().numpy()

    print("\nResults for the {}TEST{} set: ".format('\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())


def eval_val_set(model, inputs, targets):
    # validation set - use actual number of samples instead of fixed 10000
    num_test_samples = len(inputs)
    test_data_record_indices = range(0, num_test_samples)

    batch_size = min(5000, num_test_samples)  # Ensure batch size doesn't exceed sample count
    num_batches = int(np.ceil(num_test_samples / batch_size))  # Use ceil to handle remaining samples
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)
    y_predicted = np.zeros(num_test_samples)
    for i in range(num_batches):
        # x_pos_ = inputs[chunk_indices_validation[i]]
        # y_predicted[chunk_indices_validation[i]] = model.predict(x_pos_).detach().cpu().numpy()
        x_ = inputs[chunk_indices_validation[i]]
        y_predicted[chunk_indices_validation[i]] = model.predict_one_pass(x_,
                                                                          batch_size=len(chunk_indices_validation[i])).detach().cpu().numpy()

    print("\nResults for the {}VALIDATION{} set: ".format('\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())


def eval_val_set_light(model, inputs, targets, confidence_mean_vec, confidence_std_vec):
    # validation set - use actual number of samples instead of fixed 10000
    num_test_samples = len(inputs)
    test_data_record_indices = range(0, num_test_samples)

    batch_size = 1
    num_batches = int(num_test_samples / batch_size)
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)
    y_predicted = np.zeros(num_test_samples)
    predicted_with_layers_up_to = np.zeros(num_test_samples)
    for i in tqdm(range(num_batches)):
        x_ = inputs[chunk_indices_validation[i]]
        y_predicted[chunk_indices_validation[i]], predicted_with_layers_up_to[chunk_indices_validation[i]] = \
            model.light_predict_one_sample(x_, confidence_mean_vec, confidence_std_vec)

    print("\nResults for the {}VALIDATION{} set based on light inference: ".format('\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())
    print("mean number of layers used: ", np.mean(predicted_with_layers_up_to))
    values, counts = np.unique(predicted_with_layers_up_to, return_counts=True)
    print("percentage for layers_up_to ", values, " : ", counts/num_test_samples)





