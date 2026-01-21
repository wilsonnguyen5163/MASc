import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score, f1_score
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf
import ast
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_SMD(group, group_index):
    """
    Loads SMD dataset based on machine-group-group_index
    NOTE: SMD is not scaled

    Args:
        group (int): server number
        group_index (int): machine number
    """
    path = "Dataset\MTS datasets\ServerMachineDataset"
    machine = "machine-{}-{}.csv".format(group, group_index)
    X_train_full = pd.read_csv(os.path.join(
        path, 'train', machine), header=None)
    X_test_full = pd.read_csv(os.path.join(path, 'test', machine), header=None)
    y_test_full = pd.read_csv(os.path.join(
        path, 'test_label', machine), header=None)

    assert not X_train_full.isna().any().any() and not X_test_full.isna().any().any()
    return X_train_full, X_test_full, y_test_full


def load_PSM():
    path = "Dataset\MTS datasets\PooledServerMetrics"
    X_train_full = pd.read_csv(os.path.join(path, 'train.csv'), header=0).drop(
        columns=['timestamp_(min)'])
    X_test_full = pd.read_csv(os.path.join(path, 'test.csv'), header=0).drop(
        columns=['timestamp_(min)'])
    y_test_full = pd.read_csv(os.path.join(path, 'test_label.csv'), header=0).drop(
        columns=['timestamp_(min)'])

    X_train_full = X_train_full.interpolate(
        method='linear', limit_direction='both')
    X_test_full = X_test_full.interpolate(
        method='linear', limit_direction='both')

    assert not X_train_full.isna().any().any() and not X_test_full.isna().any().any()
    return X_train_full, X_test_full, y_test_full


def load_MSL_SMAP(entity, entity_num):
    """
    Loads MSL / SMAP dataset based on entity-entity_num
    NOTE: Dataset comes prescaled [-1, 1] based on test data

    Args:
        group (int): server number
        group_index (int): machine number
    """
    path = "Dataset\MTS datasets\MSL_SMAP"
    channel = "{}-{}".format(entity, entity_num)
    labeled_anomalies = pd.read_csv(os.path.join(
        path, 'labeled_anomalies.csv'), header=0)
    try:
        anomalous_sequences = ast.literal_eval(
            labeled_anomalies[labeled_anomalies['chan_id'] == channel]['anomaly_sequences'].iloc[0])
    except:
        print("Not a right channel combination")
        return None, None, None
    X_train_full = pd.DataFrame(
        np.load(os.path.join(path, 'train', channel + '.npy')))
    X_test_full = pd.DataFrame(
        np.load(os.path.join(path, 'test', channel + '.npy')))
    labels = np.zeros(X_test_full.shape[0])
    for start, end in anomalous_sequences:
        labels[start:end] = 1
    y_test_full = pd.DataFrame(labels)

    assert not X_train_full.isna().any().any() and not X_test_full.isna().any().any()
    return X_train_full, X_test_full, y_test_full


def load_WADI():
    path = "Dataset\MTS datasets\WADI"
    X_train_full = pd.read_csv(os.path.join(
        path, 'WADI_14days_new.csv'), header=0)
    X_test_full = pd.read_csv(os.path.join(
        path, 'WADI_attackdataLABLE.csv'), header=1)

    # Preprocessing (drop columns with lots of NaN)
    valid_ratio = 0.1  # keep columns with ≥10% valid data
    X_train_full = X_train_full.loc[:,
                                    X_train_full.notna().mean() > valid_ratio]
    X_test_full = X_test_full.loc[:, X_test_full.notna().mean() > valid_ratio]
    # Preprocessing: FORWARD FILL
    X_train_full = X_train_full.ffill()
    X_test_full = X_test_full.ffill()

    y_test_full = X_test_full['Attack LABLE (1:No Attack, -1:Attack)'].replace(
        1, 0).replace(-1, 1)
    X_test_full.columns = X_test_full.columns.str.strip()
    X_train_full.columns = X_train_full.columns.str.strip()
    X_test_full = X_test_full.drop(
        columns=['Attack LABLE (1:No Attack, -1:Attack)', 'Row', 'Date', 'Time'])
    X_train_full = X_train_full.drop(columns=['Row', 'Date', 'Time'])

    assert not X_train_full.isna().any().any() and not X_test_full.isna().any().any()
    assert (X_train_full.columns == X_test_full.columns).all()
    assert X_train_full.shape[1] == X_test_full.shape[1]
    assert X_test_full.shape[0] == y_test_full.shape[0]
    return X_train_full, X_test_full, y_test_full


def timestamp_labels_from(windows, window_labels):
    """
    Compute timestamp labels from provided window and its window-level label.  
    Assigns to the last timestep of the window.   
    Assumes winndows are 1-stride overlapping, and are not shuffled to maintain 
    temporal translation context
    """
    num_windows, frame_length, num_features = windows.shape
    timestamp_labels = np.zeros((num_windows * frame_length, ), dtype=int)
    for i in range(num_windows):
        timestamp_labels[i * frame_length - 1] = window_labels[i]
    return timestamp_labels


def timestamp_labels_maxpooling(window_labels, stride, n_timestamps, frame_length):
    timestamp_labels = np.zeros(n_timestamps, dtype=int)
    for i, wl in enumerate(window_labels):
        if wl == 1:
            start = i * stride
            end = min(start + frame_length, n_timestamps)
            timestamp_labels[start:end] = 1
    return timestamp_labels


def timestamp_score_maxpooling(window_scores, stride, n_timestamps, frame_length):
    timestamp_scores = np.zeros(n_timestamps, dtype=float)
    for i, score in enumerate(window_scores):
        start = i * stride
        end = min(start + frame_length, n_timestamps)
        timestamp_scores[start:end] = np.maximum(
            timestamp_scores[start:end], score)
    return timestamp_scores


def cubic_spline_interpolate(series: pd.DataFrame):
    # We dont interpolate poor quality timeseries. Interpolating poor quality data result in synthetic generation of poor data, which causes overfitting
    if (series.isna().sum()/len(series) > 0.50) or series.iloc[:300].isna().all() or series.iloc[-300:].isna().all():
        return series

    mask = ~series.isna()
    value_known = series[mask].values
    index_known = series.index[mask]

    cs = CubicSpline(index_known, value_known, extrapolate=False)
    y_interpolate = np.clip(cs(series.index), 0, a_max=None)

    return pd.Series(y_interpolate, index=series.index)


def downsample(dataset, labels, normal_label, anomaly_ratio, random_state=42):
    normals = dataset.loc[labels == normal_label]
    abnormals = dataset.loc[labels != normal_label]
    anom_count = int(np.floor(len(normals)/(1-anomaly_ratio))-len(normals))
    abnormals_sample = abnormals.sample(
        n=anom_count, random_state=random_state)
    downsampled = pd.concat([normals, abnormals_sample]).sample(
        frac=1, random_state=random_state)
    labels_downsampled = labels.loc[downsampled.index]

    return downsampled, labels_downsampled


def selective_downsample(dataset, labels, normal_label, anomaly_label, anomaly_ratio, random_state=42):
    normals = dataset.loc[labels == normal_label]
    abnormals = dataset.loc[labels == anomaly_label]
    anom_count = int(np.floor(len(normals)/(1-anomaly_ratio))-len(normals))
    if anom_count > len(abnormals):
        downsampled = pd.concat([normals, abnormals]).sample(
            frac=1, random_state=random_state)
        labels_downsampled = labels.loc[downsampled.index]
    else:
        abnormals_sample = abnormals.sample(
            n=anom_count, random_state=random_state)
        downsampled = pd.concat([normals, abnormals_sample]).sample(
            frac=1, random_state=random_state)
        labels_downsampled = labels.loc[downsampled.index]
    return downsampled, labels_downsampled


def best_f1_tune(y_tests, model_scores: dict, save=False, save_path='best_thresf1_comparison.png'):
    best_f1s = []
    models = []
    best_thresholds = {}

    for model_name, scores in model_scores.items():
        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(y_tests, scores)

        # Compute F1 for every threshold
        f1s = 2 * precision * recall / (precision + recall + 1e-8)

        # precision_recall_curve returns one extra precision/recall
        # than thresholds → ignore last point
        f1s = f1s[:-1]

        best_idx = np.argmax(f1s)
        best_f1 = f1s[best_idx]
        best_thr = thresholds[best_idx]

        best_f1s.append(best_f1)
        models.append(model_name)
        best_thresholds[model_name] = {
            "best_f1": float(best_f1),
            "best_threshold": float(best_thr)
        }

    # Plot
    plt.figure(figsize=(10, 8))
    plt.bar(models, best_f1s)
    plt.xlabel("Models")
    plt.ylabel("Best F1 (PR-tuned)")
    plt.ylim(0, 1)
    plt.title("Best F1 after Threshold Tuning (PR Curve)")
    plt.grid(axis="y")

    if save:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    return best_thresholds


def display_f1_models(y_tests, model_preds: dict, save=False, save_path='f1_comparison.png'):
    """
    Display F1 scores of each model

    Args:
        y_tests (np.ndarray): true labels
        model_preds (dict): dict{ model_keys : model's predictions}
        save (bool, optional): Whether to save graph. Defaults to False.
        save_path (str, optional): File path for graph saving. Defaults to 'f1_comparison.png'.

    Returns:
        _type_: _description_
    """
    plt.figure(figsize=(10, 8))
    f1s = []
    models = []
    for model_name, y_preds in model_preds.items():
        f1s.append(f1_score(y_tests, y_preds))
        models.append(model_name)

    plt.bar(models, f1s)
    plt.xlabel('Models')
    plt.ylabel('F1-score')
    plt.ylim(0, 1)
    plt.title('F1-score Comparison')
    plt.grid()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    return dict(zip(models, f1s))


def compare_pr_models(y_tests, model_scores: dict, title='Precision-Recall Curve Comparison', save=False, save_path='pr_comparison.png'):
    """
    Display PR curve comparison between models

    Args:
        y_tests (np.ndarray): true labels
        model_scores (dict): { model_keys \: model's anomaly scores }
        title (str, optional): Display title. Defaults to 'Precision-Recall Curve Comparison'.
        save (bool, optional): Whether to save graph. Defaults to False.
        save_path (str, optional): File path for graph saving. Defaults to 'pr_comparison.png'.
    """

    plt.figure(figsize=(10, 8))
    for model_name, y_preds in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_tests, y_preds)
        ap = average_precision_score(y_tests, y_preds)
        plt.plot(recall, precision, label=f'{model_name} (AP={ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()


def compare_roc_models(y_tests, model_scores: dict, save=False, save_path='roc_comparison.png'):
    """
    Display roc graph comparison between models

    Args:
        y_tests (np.ndarray): true labels
        model_scores (dict): { model_keys \: model's anomaly scores }
        save (bool, optional): Whether to save graph. Defaults to False.
        save_path (str, optional): File path for graph saving. Defaults to 'roc_comparison.png'.
    """
    plt.figure(figsize=(10, 8))
    scores = {}
    for model_name, y_preds in model_scores.items():
        fpr, tpr, _ = roc_curve(y_tests, y_preds)
        auc = roc_auc_score(y_tests, y_preds)
        scores[model_name] = auc
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    return scores


def compare_pr_bar_models(y_tests, model_scores: dict, save=False, save_path='ap_comparison.png'):
    """
    Display average PR score comparison between models, display in bar format

    Args:
        y_tests (np.ndarray): true labels
        model_scores (dict): { model_keys \: model's anomaly scores }
        save (bool, optional): Whether to save graph. Defaults to False.
        save_path (str, optional): File path for graph saving. Defaults to 'ap_comparison.png'.
    """

    ap_scores = {}
    for model_name, y_preds in model_scores.items():
        ap = average_precision_score(y_tests, y_preds)
        ap_scores[model_name] = ap
    plt.figure(figsize=(10, 6))
    plt.bar(ap_scores.keys(), ap_scores.values(), color='skyblue')
    plt.ylabel('Average Precision (AP) Score')
    plt.title('Average Precision Score Comparison')
    plt.ylim(0, 1)
    for i, v in enumerate(ap_scores.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)
    if save:
        plt.savefig(save_path)
    else:
        plt.show()


def compare_roc_bar_models(y_tests, model_scores: dict, save=False, save_path='auc_comparison.png'):
    """
    Display ROCAUC score comparison between models

    Args:
        y_tests (np.ndarray): true labels
        model_scores (dict): { model_keys \: model's anomaly scores }
        save (bool, optional): Whether to save graph. Defaults to False.
        save_path (str, optional): File path for graph saving. Defaults to 'auc_comparison.png'.
    """
    auc_scores = {}
    for model_name, y_preds in model_scores.items():
        auc = roc_auc_score(y_tests, y_preds)
        auc_scores[model_name] = auc
    plt.figure(figsize=(10, 6))
    plt.bar(auc_scores.keys(), auc_scores.values(), color='salmon')
    plt.ylabel('Area Under Curve (AUC) Score')
    plt.title('AUC Score Comparison')
    plt.ylim(0, 1)
    for i, v in enumerate(auc_scores.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)
    plt.show()


def visualize_model_results(avg_f1s: dict, avg_rocs: dict, save=False, save_path='model_performance_comparison.png'):
    """
    Display average PR score comparison between models

    Args:
        avg_f1s (dict):  dataset \: {model \: model's performance}
        avg_rocs (dict): dataset \: {model \: model's performance}
        save (bool, optional): Whether to save graph. Defaults to False.
        save_path (str, optional): File path for graph saving. Defaults to 'model_performance_comparison.png'.
    """
    for dataset_name, perfs in avg_f1s.items():
        models = list(perfs.keys())
        f1_scores = list(perfs.values())
        plt.figure(figsize=(12, 5))
        plt.bar(models, f1_scores, color='salmon')
        plt.ylabel('F1-score')
        plt.ylim(0, 1)
        plt.title(f'Dataset: {dataset_name} - Average F1-score Comparison')
        if save:
            plt.savefig(f'{save_path}_{dataset_name}_f1.png')
        else:
            plt.show()
    for dataset_name, perfs in avg_rocs.items():
        models = list(perfs.keys())
        roc_scores = list(perfs.values())
        plt.figure(figsize=(12, 5))
        plt.bar(models, roc_scores, color='salmon')
        plt.ylabel('ROC AUC Score')
        plt.ylim(0, 1)
        plt.title(
            f'Dataset: {dataset_name} - Average ROC AUC Score Comparison')
        if save:
            plt.savefig(f'{save_path}_{dataset_name}_roc.png')
        else:
            plt.show()


def get_time_series_between(start: datetime, end: datetime, time_column, data: pd.DataFrame) -> pd.DataFrame:
    mask = (data[time_column] >= start) & (data[time_column] <= end)
    series = data[mask].copy()
    return series


def load_preprocessed_SMD_windows(server, machine, scaler=None, window_size=10, frame_step=1,):
    """
    Loads SMD dataset, preprocess with scaler (if specified), and form sliding windows for both train and test set  

    Args:
        server (int): Server Number for SMD dataset.
        machine (_type_): Machine number.
        scaler (StandardScaler, MinMaxScaler, optional): Scaler to scale data. Defaults to None.
        window_size (int, optional): Sliding window size. Defaults to 10.
        frame_step (int, optional): Stride. Defaults to 1.
        inject (bool, optional): Toggle whether to corrupt timesteps. Defaults to False.
        inject_properties (dict, optional): injection types. Defaults to {}.   

    Returns:
        train_ds (TensorSliceDataset): Train windows dataset (N_train, W, F).  
        test_ds (TensorSliceDataset): Test windows dataset (N, W, F).  
        test_labels (Tensor): Test window labels (N, 1).  
        events (list[dict]): Corruption information.   
    """
    X_train_full, X_test_full, y_test_full = load_SMD(server, machine)
    if scaler is not None:
        X_train_full = scaler.fit_transform(X_train_full)
        X_test_full = scaler.transform(X_test_full)

    X_train_windows = tf.signal.frame(tf.convert_to_tensor(
        X_train_full, dtype=tf.float32), frame_length=window_size, frame_step=frame_step, axis=0)
    X_test_windows = tf.signal.frame(tf.convert_to_tensor(
        X_test_full, dtype=tf.float32), frame_length=window_size, frame_step=frame_step, axis=0)
    test_windows_labels = tf.signal.frame(tf.convert_to_tensor(
        y_test_full, dtype=tf.float32), frame_length=window_size, frame_step=frame_step, axis=0).numpy()
    reshaped = test_windows_labels.reshape(test_windows_labels.shape[0], -1)
    window_has_anom = np.any(reshaped == 1, axis=1).astype(np.int32)

    X_train_windows = tf.convert_to_tensor(
        X_train_windows, dtype=tf.float32)           # [N, T, F]
    X_test_windows = tf.convert_to_tensor(
        X_test_windows, dtype=tf.float32)       # [N, T, F]
    y_test_windows = tf.convert_to_tensor(
        window_has_anom, dtype=tf.int32)      # [N]
    train_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(X_train_windows, tf.float32))
    test_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(X_test_windows, tf.float32))
    test_labels = tf.cast(y_test_windows, tf.int32)

    return train_ds, test_ds, test_labels.numpy()


def load_preprocessed_PSM_windows(scaler=None, window_size=10, frame_step=1, lazy=False):
    """
    Loads PSM dataset, preprocess with scaler (if specified), and form sliding windows for both train and test set  

    Args:
        scaler (StandardScaler, MinMaxScaler, optional): Scaler to scale data. Defaults to None.
        window_size (int, optional): Sliding window size. Defaults to 10.
        frame_step (int, optional): Stride. Defaults to 1.

    Returns:
        train_ds (TensorSliceDataset): Train windows dataset (N_train, W, F).  
        test_ds (TensorSliceDataset): Test windows dataset (N, W, F).  
        test_labels (Tensor): Test window labels (N, 1).  
    """
    X_train_full, X_test_full, y_test_full = load_PSM()

    if scaler is not None:
        X_train_full = scaler.fit_transform(X_train_full)
        X_test_full = scaler.transform(X_test_full)

    if not lazy:
        X_train_windows = tf.signal.frame(tf.convert_to_tensor(
            X_train_full, dtype=tf.float32), frame_length=window_size, frame_step=frame_step, axis=0)
        X_test_windows = tf.signal.frame(tf.convert_to_tensor(
            X_test_full, dtype=tf.float32), frame_length=window_size, frame_step=frame_step, axis=0)
        test_windows_labels = tf.signal.frame(tf.convert_to_tensor(
            y_test_full, dtype=tf.float32), frame_length=window_size, frame_step=frame_step, axis=0).numpy()
        reshaped = test_windows_labels.reshape(
            test_windows_labels.shape[0], -1)
        window_has_anom = np.any(reshaped == 1, axis=1).astype(np.int32)

        X_train_windows = tf.convert_to_tensor(
            X_train_windows, dtype=tf.float32)           # [N, T, F]
        X_test_windows = tf.convert_to_tensor(
            X_test_windows, dtype=tf.float32)       # [N, T, F]
        y_test_windows = tf.convert_to_tensor(
            window_has_anom, dtype=tf.int32)      # [N]
        train_ds = tf.data.Dataset.from_tensor_slices(
            tf.cast(X_train_windows, tf.float32))
        test_ds = tf.data.Dataset.from_tensor_slices(
            tf.cast(X_test_windows, tf.float32))
        test_labels = tf.cast(y_test_windows, tf.int32)
        return train_ds, test_ds, test_labels.numpy()

    else:
        X_train_full = tf.convert_to_tensor(X_train_full, tf.float32)
        X_test_full = tf.convert_to_tensor(X_test_full, tf.float32)
        y_test_full = tf.convert_to_tensor(y_test_full, tf.int32)

        train_ds = (tf.data.Dataset.from_tensor_slices(X_train_full).window(
            window_size, shift=frame_step, drop_remainder=True).flat_map(lambda w: w.batch(window_size))).shuffle(10000)
        test_ds = (tf.data.Dataset.from_tensor_slices(X_test_full).window(
            window_size, shift=frame_step, drop_remainder=True).flat_map(lambda w: w.batch(window_size)))

        test_y_ds = (
            tf.data.Dataset.from_tensor_slices(y_test_full)
            .window(window_size, shift=frame_step, drop_remainder=True)
            .flat_map(lambda w: w.batch(window_size))
            .map(lambda w: tf.cast(tf.reduce_any(w == 1), tf.int32))
        )
        window_labels = np.array(
            [int(v) for v in test_y_ds.as_numpy_iterator()], dtype=np.int32)

        return train_ds, test_ds, window_labels


def load_preprocessed_MSL_SMAP_windows(entity, entity_num, scaler=None, window_size=10, frame_step=1):
    """
    Loads MSL / SMAP dataset, preprocess with scaler (if specified), and form sliding windows for both train and test set  

    Args:
        entity (str): server number
        entity_num (int): machine number
        scaler (StandardScaler, MinMaxScaler, optional): Scaler to scale data. Defaults to None.
        window_size (int, optional): Sliding window size. Defaults to 10.
        frame_step (int, optional): Stride. Defaults to 1.

    Returns:
        train_ds (TensorSliceDataset): Train windows dataset (N_train, W, F).  
        test_ds (TensorSliceDataset): Test windows dataset (N, W, F).  
        test_labels (Tensor): Test window labels (N, 1).  
    """
    X_train_full, X_test_full, y_test_full = load_MSL_SMAP(entity, entity_num)

    if scaler is not None:
        X_train_full = scaler.fit_transform(X_train_full)
        X_test_full = scaler.transform(X_test_full)

    X_train_windows = tf.signal.frame(tf.convert_to_tensor(
        X_train_full, dtype=tf.float32), frame_length=window_size, frame_step=frame_step, axis=0)
    X_test_windows = tf.signal.frame(tf.convert_to_tensor(
        X_test_full, dtype=tf.float32), frame_length=window_size, frame_step=frame_step, axis=0)
    test_windows_labels = tf.signal.frame(tf.convert_to_tensor(
        y_test_full, dtype=tf.float32), frame_length=window_size, frame_step=frame_step, axis=0).numpy()
    reshaped = test_windows_labels.reshape(test_windows_labels.shape[0], -1)
    window_has_anom = np.any(reshaped == 1, axis=1).astype(np.int32)

    X_train_windows = tf.convert_to_tensor(
        X_train_windows, dtype=tf.float32)           # [N, T, F]
    X_test_windows = tf.convert_to_tensor(
        X_test_windows, dtype=tf.float32)       # [N, T, F]
    y_test_windows = tf.convert_to_tensor(
        window_has_anom, dtype=tf.int32)      # [N]
    train_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(X_train_windows, tf.float32))
    test_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(X_test_windows, tf.float32))
    test_labels = tf.cast(y_test_windows, tf.int32)
    return train_ds, test_ds, test_labels.numpy()


def load_preprocessed_WADI_windows(scaler=None, window_size=10, frame_step=1, max_windows=100000):
    """
    Loads WADI dataset, preprocess with scaler (if specified), and form sliding windows for both train and test set  

    Args:
        scaler (StandardScaler, MinMaxScaler, optional): Scaler to scale data. Defaults to None.
        window_size (int, optional): Sliding window size. Defaults to 10.
        frame_step (int, optional): Stride. Defaults to 1.

    Returns:
        train_ds (TensorSliceDataset): Train windows dataset (max_windows, W, F).  
        test_ds (TensorSliceDataset): Test windows dataset (N, W, F).  
        test_labels (Tensor): Test window labels (N, 1).  
    """
    X_train_full, X_test_full, y_test_full = load_WADI()

    if scaler is not None:
        X_train_full = scaler.fit_transform(X_train_full)
        X_test_full = scaler.transform(X_test_full)

    X_train_full = tf.convert_to_tensor(X_train_full, tf.float32)
    X_test_full = tf.convert_to_tensor(X_test_full, tf.float32)
    y_test_full = tf.convert_to_tensor(y_test_full, tf.int32)

    train_ds = (tf.data.Dataset.from_tensor_slices(X_train_full).window(
        window_size, shift=frame_step, drop_remainder=True).flat_map(lambda w: w.batch(window_size))).shuffle(10000)
    test_ds = (tf.data.Dataset.from_tensor_slices(X_test_full).window(
        window_size, shift=frame_step, drop_remainder=True).flat_map(lambda w: w.batch(window_size)))

    test_y_ds = (
        tf.data.Dataset.from_tensor_slices(y_test_full)
        .window(window_size, shift=frame_step, drop_remainder=True)
        .flat_map(lambda w: w.batch(window_size))
        .map(lambda w: tf.cast(tf.reduce_any(w == 1), tf.int32))
    )
    window_labels = np.array(
        [int(v) for v in test_y_ds.as_numpy_iterator()], dtype=np.int32)

    train_ds = train_ds.take(max_windows)

    return train_ds, test_ds, window_labels


class AnomalyWindowSampler():
    @staticmethod
    def sample_(windows: np.ndarray, point_perturb=False, contextual_perturb=False, collective_perturb=False, seed=0, max_cut_ratio=0.5, return_multi_labels=False):
        rng = np.random.RandomState(seed=seed)

        batch_corrupted = windows.copy()
        B, time_n, feature_dim = batch_corrupted.shape
        cut_start = time_n - rng.randint(1, int(max_cut_ratio*time_n), size=B)
        n_cut_dim = rng.randint(1, feature_dim+1, size=B)
        cut_dim = [rng.randint(0, feature_dim, size=n_cut_dim[i])
                   for i in range(B)]
        neg_labels = np.zeros(B, dtype=int)

        n_types = 6
        if point_perturb and contextual_perturb and collective_perturb:
            flags = rng.randint(0, n_types, size=B)
        else:
            pool = rng.randint(0, n_types, size=int(1e4))
            if collective_perturb:
                pool = pool[pool < 2]
            elif contextual_perturb:
                pool = pool[(pool >= 2) & (pool < 4)]
            elif point_perturb:
                pool = pool[(pool >= 4) & (pool < 6)]
            flags = rng.choice(pool, size=B, replace=False)

        for i in range(B):
            match (flags[i] % n_types):
                # Collective Perturbation
                case 0:
                    batch_corrupted[i, cut_start[i]:, cut_dim[i]] = 0
                    neg_labels[i] = 1
                case 1:
                    batch_corrupted[i, cut_start[i]:, cut_dim[i]] = 1
                    neg_labels[i] = 1

                # Contextual Perturbation
                case 2:
                    window = batch_corrupted[i, -10:, :]
                    mean = np.mean(window[:, cut_dim[i]], axis=0)
                    batch_corrupted[i, -1, cut_dim[i]] = mean+0.5
                    neg_labels[i] = 2
                case 3:
                    window = batch_corrupted[i, -10:, :]
                    mean = np.mean(window[:, cut_dim[i]], axis=0)
                    batch_corrupted[i, -1, cut_dim[i]] = mean-0.5
                    neg_labels[i] = 2

                # Point Perturbation
                case 4:
                    batch_corrupted[i, -1, cut_dim[i]] = 2
                    neg_labels[i] = 3
                case 5:
                    batch_corrupted[i, -1, cut_dim[i]] = -2
                    neg_labels[i] = 3

        if return_multi_labels:
            return batch_corrupted, neg_labels
        else:
            return batch_corrupted, np.ones_like(neg_labels)

    @staticmethod
    def augment_amplify(windows, nv=None, p=0.5, amp_val=1.5, random_state=42):
        windows = np.copy(windows)
        rng = np.random.default_rng(random_state)
        B, timesteps, dims = np.shape(windows)
        nv = nv if nv is not None else 1
        n_steps = max(int(p*timesteps), 1)
        for b in range(B):
            feat_idx = rng.choice(dims, size=nv, replace=False)
            for idx in feat_idx:
                t_idx = rng.choice(timesteps, size=n_steps, replace=False)
                windows[b, t_idx, idx] *= amp_val
        return windows

    @staticmethod
    def augment_additive(windows, nv=None, p=0.5, bias=0.2, random_state=42):
        windows = np.copy(windows)
        rng = np.random.default_rng(random_state)
        B, timesteps, dims = np.shape(windows)
        nv = nv if nv is not None else 1
        n_steps = max(int(p*timesteps), 1)
        for b in range(B):
            feat_idx = rng.choice(dims, size=nv, replace=False)
            for idx in feat_idx:
                t_idx = rng.choice(timesteps, size=n_steps, replace=False)
                windows[b, t_idx, idx] += bias
        return windows

    @staticmethod
    def augment_dropout_point(windows, nv=None, p=0.5, random_state=42):
        windows = np.copy(windows)
        rng = np.random.default_rng(random_state)
        B, timesteps, dims = np.shape(windows)
        nv = nv if nv is not None else 1
        n_steps = max(int(p*timesteps), 1)
        for b in range(B):
            feat_idx = rng.choice(dims, size=nv, replace=False)
            for idx in feat_idx:
                t_idx = rng.choice(timesteps, size=n_steps, replace=False)
                windows[b, t_idx, idx] = 0
        return windows

    @staticmethod
    def augment_permute(windows, nv=None, p=0.5, random_state=42):
        windows = np.copy(windows)
        rng = np.random.default_rng(random_state)
        B, timesteps, dims = np.shape(windows)
        nv = nv if nv is not None else 1
        n_steps = max(int(p*timesteps), 1)
        for b in range(B):
            feat_idx = rng.choice(dims, size=nv, replace=False)
            for idx in feat_idx:
                start = rng.integers(0, timesteps - n_steps + 1)
                t_idx = np.arange(start, start + n_steps)
                windows[b, t_idx, idx] = windows[b,
                                                 t_idx, idx][rng.permutation(n_steps)]
        return windows

    @staticmethod
    def augment_reverse(windows, nv=None, p=0.5, random_state=42):
        windows = np.copy(windows)
        rng = np.random.default_rng(random_state)
        B, timesteps, dims = np.shape(windows)
        nv = nv if nv is not None else 1
        n_steps = max(int(p*timesteps), 1)
        for b in range(B):
            feat_idx = rng.choice(dims, size=nv, replace=False)
            for idx in feat_idx:
                start = rng.integers(0, timesteps - n_steps + 1)
                t_idx = np.arange(start, start + n_steps)
                windows[b, t_idx, idx] = windows[b, t_idx[::-1], idx]
        return windows

    @staticmethod
    def augment_noise(windows, nv=None, p=None, sigma=0.05, random_state=42):
        windows = np.copy(windows)
        rng = np.random.default_rng(random_state)
        B, timesteps, dims = np.shape(windows)
        nv = nv if nv is not None else 1
        n_steps = timesteps
        for b in range(B):
            feat_idx = rng.choice(dims, size=nv, replace=False)
            for idx in feat_idx:
                start = rng.integers(0, timesteps - n_steps + 1)
                t_idx = np.arange(start, start + n_steps)
                noise = rng.normal(loc=0.0, scale=sigma, size=n_steps)
                windows[b, t_idx, idx] += noise
        return windows

    @staticmethod
    def augment_dropout_segment(windows, nv=None, p=0.5, random_state=42):
        windows = np.copy(windows)
        rng = np.random.default_rng(random_state)
        B, timesteps, dims = np.shape(windows)
        nv = nv if nv is not None else 1
        n_steps = max(int(p*timesteps), 1)
        for b in range(B):
            feat_idx = rng.choice(dims, size=nv, replace=False)
            for idx in feat_idx:
                start = rng.integers(0, timesteps - n_steps + 1)
                t_idx = np.arange(start, start + n_steps)
                windows[b, t_idx, idx] = 0
        return windows


class MinMaxClippingScaler():
    def __init__(self, clip_val=4):
        self.feature_min = None
        self.feature_max = None
        self.clip_val = clip_val

    def fit(self, ds):
        ds = np.asarray(ds)
        self.feature_min = ds.min(axis=0)
        self.feature_max = ds.max(axis=0)

    def fit_transform(self, ds):
        ds = np.asarray(ds)
        self.fit(ds)
        scaled = (ds - self.feature_min) / \
            (self.feature_max - self.feature_min + 1e-8)
        return scaled

    def transform(self, ds):
        ds = np.asarray(ds)
        scaled = np.clip(ds, self.feature_min-self.clip_val,
                         self.feature_max+self.clip_val)
        scaled = (scaled - self.feature_min) / \
            (self.feature_max - self.feature_min + 1e-8)
        return scaled


def point_adjust(y_true, y_pred):
    adjusted = y_pred.copy()
    in_anomaly = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        if y_true[i] == 0 and in_anomaly:
            in_anomaly = False
            end = i

            if np.any(y_pred[start:end] == 1):
                adjusted[start:end] = 1
    if in_anomaly:
        if np.any(y_pred[start:] == 1):
            adjusted[start:] = 1
    return adjusted
