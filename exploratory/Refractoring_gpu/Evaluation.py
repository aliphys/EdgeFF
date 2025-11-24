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
    # train set
    num_train_samples = 50000
    train_data_record_indices = range(0, num_train_samples)

    batch_size = 5000
    num_batches = int(num_train_samples / batch_size)
    chunk_indices = np.array_split(train_data_record_indices, num_batches)

    y_predicted = np.zeros(num_train_samples)
    for i in range(num_batches):
        # x_pos_ = inputs[chunk_indices[i]]
        # y_predicted[chunk_indices[i]] = model.predict(x_pos_).detach().cpu().numpy()
        x_ = inputs[chunk_indices[i]]
        y_predicted[chunk_indices[i]] = model.predict_one_pass(x_, batch_size=batch_size).detach().cpu().numpy()

    print("\nResults for the {}TRAIN{} set: ".format('\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())


def eval_test_set(model, inputs, targets):
    # test set
    num_test_samples = 10000
    test_data_record_indices = range(0, num_test_samples)

    batch_size = 5000
    num_batches = int(num_test_samples / batch_size)
    chunk_indices_test = np.array_split(test_data_record_indices, num_batches)
    y_predicted = np.zeros(num_test_samples)
    for i in range(num_batches):
        # x_pos_ = inputs[chunk_indices_test[i]]
        # y_predicted[chunk_indices_test[i]] = model.predict(x_pos_).detach().cpu().numpy()
        x_ = inputs[chunk_indices_test[i]]
        y_predicted[chunk_indices_test[i]] = model.predict_one_pass(x_, batch_size=batch_size).detach().cpu().numpy()

    print("\nResults for the {}TEST{} set: ".format('\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())


def eval_val_set(model, inputs, targets):
    # test set
    num_test_samples = 10000
    test_data_record_indices = range(0, num_test_samples)

    batch_size = 5000
    num_batches = int(num_test_samples / batch_size)
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)
    y_predicted = np.zeros(num_test_samples)
    for i in range(num_batches):
        # x_pos_ = inputs[chunk_indices_validation[i]]
        # y_predicted[chunk_indices_validation[i]] = model.predict(x_pos_).detach().cpu().numpy()
        x_ = inputs[chunk_indices_validation[i]]
        y_predicted[chunk_indices_validation[i]] = model.predict_one_pass(x_,
                                                                          batch_size=batch_size).detach().cpu().numpy()

    print("\nResults for the {}VALIDATION{} set: ".format('\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())


def eval_val_set_light(model, inputs, targets, confidence_mean_vec, confidence_std_vec):
    # test set
    num_test_samples = 10000
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


def eval_with_inference_measurement(model, inputs, targets, hw_monitor=None, set_name='test'):
    """
    Evaluate model and measure inference energy/latency if hw_monitor is provided.
    
    Args:
        model: The model to evaluate
        inputs: Input tensors
        targets: Target labels
        hw_monitor: TegratsMonitor instance (optional)
        set_name: Name of the dataset (for logging)
    
    Returns:
        dict: Contains accuracy, error, and inference metrics (if hw_monitor provided)
    """
    num_samples = inputs.shape[0]
    batch_size = min(5000, num_samples)
    num_batches = int(np.ceil(num_samples / batch_size))
    
    y_predicted = np.zeros(num_samples)
    all_inference_metrics = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        actual_batch_size = end_idx - start_idx
        
        x_batch = inputs[start_idx:end_idx]
        
        # Measure inference if monitor available
        if hw_monitor:
            hw_monitor.start_inference_measurement()
        
        predictions = model.predict_one_pass(x_batch, batch_size=actual_batch_size)
        
        if hw_monitor:
            batch_metrics = hw_monitor.stop_inference_measurement(actual_batch_size)
            if batch_metrics:
                all_inference_metrics.append(batch_metrics)
        
        y_predicted[start_idx:end_idx] = predictions.detach().cpu().numpy()
    
    # Calculate accuracy metrics
    accuracy = accuracy_score(targets.detach().cpu().numpy(), y_predicted)
    f1 = f1_score(targets.detach().cpu().numpy(), y_predicted, average='macro')
    error = 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item()
    
    print(f"\nResults for the {set_name.upper()} set:")
    print(f"\tF1-score: {f1}")
    print(f"\tAccuracy: {accuracy}")
    print(f"\tError: {error}")
    
    results = {
        f'{set_name}/accuracy': accuracy,
        f'{set_name}/f1_score': f1,
        f'{set_name}/error': error,
    }
    
    # Average inference metrics across batches
    if all_inference_metrics:
        avg_metrics = {
            f'{set_name}/inference_latency_per_sample_ms': np.mean([m['inference/latency_per_sample_ms'] for m in all_inference_metrics]),
            f'{set_name}/inference_energy_per_sample_mj': np.mean([m['inference/energy_per_sample_mj'] for m in all_inference_metrics]),
            f'{set_name}/inference_avg_power_mw': np.mean([m['inference/avg_power_during_inference_mw'] for m in all_inference_metrics]),
            f'{set_name}/inference_total_latency_ms': sum([m['inference/total_batch_latency_ms'] for m in all_inference_metrics]),
            f'{set_name}/inference_total_energy_mj': sum([m['inference/total_batch_energy_mj'] for m in all_inference_metrics]),
        }
        results.update(avg_metrics)
        
        print(f"\nInference Metrics for {set_name.upper()} set:")
        print(f"\tLatency per sample: {avg_metrics[f'{set_name}/inference_latency_per_sample_ms']:.4f} ms")
        print(f"\tEnergy per sample: {avg_metrics[f'{set_name}/inference_energy_per_sample_mj']:.4f} mJ")
        print(f"\tAverage power: {avg_metrics[f'{set_name}/inference_avg_power_mw']:.2f} mW")
    
    return results


