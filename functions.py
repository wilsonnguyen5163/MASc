import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC


def load_spain_electrical_dataset():
    """
    Timeseries resolution: 1 hour  

    4 years of electrical data of Spain.   
    """
    dataset_full = pd.read_csv('./Dataset/Energy Load Timeseries/SpainConsumption_hourly/dataset.csv')
    dataset_full['time'] = pd.to_datetime(dataset_full['time'], utc=True)
    dataset_full['time'] = dataset_full['time'].dt.tz_localize(None)
    X_full = dataset_full[['time', 'total load actual']].rename(columns={'time':'Timestamp', 'total load actual': "Observed Load"})
    X_train = get_time_series_between(pd.to_datetime('2015-01-01'), pd.to_datetime('2017-12-31 23:00:00'), 'Timestamp', X_full)
    X_test = get_time_series_between(pd.to_datetime('2018-01-01'), pd.to_datetime('2099-01-01'), 'Timestamp', X_full)
    return X_train, X_test, 'MW'

def load_apartment_electrical_dataset():
    """
    Timeseries resolution: 15 mins   
    
    Apartment-unit raw meter energy consumption, 2 years San Jose area
    """
    dataset_full = pd.read_csv('./Dataset/Energy Load Timeseries/ApartmentConsumption_15mins/dataset.csv')
    dataset_full['Timestamp'] = pd.to_datetime(dataset_full['DATE'] + ' ' +dataset_full['START TIME'])
    dataset_full = dataset_full[~dataset_full['Timestamp'].duplicated(keep='first')]
    dataset_full['USAGE'] = dataset_full['USAGE'] * 4
    X_full = dataset_full[['Timestamp', 'USAGE']].rename(columns={'USAGE':'Observed Load'})
    X_train = get_time_series_between(pd.to_datetime('2016'), pd.to_datetime('2017-12-31 23:45:00'), 'Timestamp', X_full)
    X_test = get_time_series_between(pd.to_datetime('2018'), pd.to_datetime('2099-01-01'), 'Timestamp', X_full)
    return X_train, X_test, 'kW'

def load_EU_aggregated_electrical_15mins_dataset():
    """
    Timeseries resolution: 15 mins      
    
    Aggregated electrical consumption data within EU and its neighbouring country.
    """
    dataset_full = pd.read_csv('./Dataset/Energy Load Timeseries/EU_AggregatedElectricalLoad/dataset_15.csv')
    dataset_full['Timestamp'] = pd.to_datetime(dataset_full['utc_timestamp'], utc=True)
    dataset_full['Timestamp'] = dataset_full['Timestamp'].dt.tz_localize(None)
    X_full = dataset_full[['Timestamp', 'NL_load_forecast_entsoe_transparency']].rename(columns={'NL_load_forecast_entsoe_transparency':'Observed Load'})
    X_train = get_time_series_between(pd.to_datetime('2015'), pd.to_datetime('2019-12-31 23:45:00'), 'Timestamp', X_full)
    X_test = get_time_series_between(pd.to_datetime('2018'), pd.to_datetime('2099-01-01'), 'Timestamp', X_full)
    return X_train, X_test, 'MW'

def load_PDB_electrical_dataset():
    """
    Timeseries resolution: 1 hour      
    
    Dataset curated for load forecasting, extracted from the PDB (power distribution box)
    """
    dataset_full = pd.read_csv('./Dataset/Energy Load Timeseries/PDB_ElectricalLoad_hourly/dataset.csv')
    dataset_full['Timestamp'] = pd.to_datetime(dataset_full['date']) +  pd.to_timedelta(dataset_full['hour'], unit='h')
    X_full = dataset_full[['Timestamp', 'demand']].rename(columns={'demand':'Observed Load'})
    X_train = get_time_series_between(pd.to_datetime('2003'), pd.to_datetime('2013-12-31 23:00:00'), 'Timestamp', X_full)
    X_test = get_time_series_between(pd.to_datetime('2014'), pd.to_datetime('2099-01-01'), 'Timestamp', X_full)
    return X_train, X_test, 'W'


def load_PJM_AEP_electrical_dataset():
    """
    Timeseries resolution: 1 hour      
    
    Hourly energy consumption over 9 years, obtained from RTO (regional Transmission Organization) in the US, responsible for electric transmission system that serves many areas of the US      
    """
    dataset_full = pd.read_csv('Dataset\Energy Load Timeseries\PJM_AEP_ElectricalGrid_hourly\dataset.csv')
    dataset_full['Timestamp'] = pd.to_datetime(dataset_full['Datetime'])
    X_full = dataset_full[['Timestamp', 'PJME_MW']].rename(columns={'PJME_MW':'Observed Load'})
    X_train = get_time_series_between(pd.to_datetime('2009'), pd.to_datetime('2016-12-31 23:00:00'), 'Timestamp', X_full)
    X_test = get_time_series_between(pd.to_datetime('2017'), pd.to_datetime('2099'), 'Timestamp', X_full)
    return X_train, X_test, 'MW'

def load_smart_building_electrical_dataset():
    """
    Timeseries resolution: 1 min  

    Smart building operational data (including electrical consumption, environmental measurements)   
    """
    dataset_full = pd.read_csv('Dataset/Energy Load Timeseries/SmartBuilding_1min/2018Floor5.csv')
    dataset_full['Timestamp'] = pd.to_datetime(dataset_full['Date'])
    dataset_full['Observed Load'] = dataset_full.select_dtypes(include='number').sum(axis=1)
    X_train = dataset_full[['Timestamp', 'Observed Load']]
    
    dataset_full_test = pd.read_csv('Dataset/Energy Load Timeseries/SmartBuilding_1min/2019Floor5.csv')
    dataset_full_test['Timestamp'] = pd.to_datetime(dataset_full_test['Date'])
    dataset_full_test['Observed Load'] = dataset_full_test.select_dtypes(include='number').sum(axis=1)
    X_test = dataset_full_test[['Timestamp', 'Observed Load']]
    return X_train, X_test, 'kW'

def load_UK_electricalgrid_dataset():
    """
    Timeseries resolution: 30 mins  

    Real-life load obtained from National Grid of Great Britain, from 2008 till present day
    """
    dataset_full = pd.read_csv('Dataset/Energy Load Timeseries/UK_ElectricalLoad_30mins/dataset.csv')
    dataset_full['Timestamp'] = pd.to_datetime(dataset_full['ELEXM_utc'], utc=True)
    dataset_full['Timestamp'] = dataset_full['Timestamp'].dt.tz_localize(None)

    X_full = dataset_full[['Timestamp', 'POWER_ESPENI_MW']].rename(columns={'POWER_ESPENI_MW':'Observed Load'})
    X_train = get_time_series_between(pd.to_datetime('2008'), pd.to_datetime('2020-12-31 23:30:00'), 'Timestamp', X_full)
    X_test = get_time_series_between(pd.to_datetime('2021'), pd.to_datetime('2099'), 'Timestamp', X_full)
    return X_train, X_test, 'MW'

def load_SGCC_dataset():
    """
    Timeseries resolution: daily  

    Electricity theft detection released by the State Grid Corporation of China (SGCC), from 1 January 2014 to 30 October 2016
    """
    dataset_full = pd.read_csv('Dataset\Energy Load Timeseries\SGCC\data set.csv')
    X_full = dataset_full.iloc[:,:-2]
    X_full.columns = pd.to_datetime(X_full.columns)
    y_full = dataset_full.iloc[:,-1]
    return X_full, y_full

def load_SMD(group, group_index):
    path = "Dataset\MTS datasets\ServerMachineDataset"
    machine = "machine-{}-{}.csv".format(group, group_index)
    
    X_train_full = pd.read_csv(os.path.join(path,'train',machine), header=None)
    X_test_full = pd.read_csv(os.path.join(path,'test',machine), header=None)
    y_test_full = pd.read_csv(os.path.join(path,'test_label',machine), header=None)
    return X_train_full, X_test_full, y_test_full

def cubic_spline_interpolate(series:pd.DataFrame):
    # We dont interpolate poor quality timeseries. Interpolating poor quality data result in synthetic generation of poor data, which causes overfitting
    if (series.isna().sum()/len(series) > 0.50) or series.iloc[:300].isna().all() or series.iloc[-300:].isna().all():
        return series
    
    mask = ~series.isna()
    value_known = series[mask].values
    index_known = series.index[mask]
    
    cs = CubicSpline(index_known, value_known, extrapolate=False)
    y_interpolate = np.clip(cs(series.index), 0, a_max=None)
    
    return pd.Series(y_interpolate, index=series.index)

def load_KDD():
    """
    Load KDD99 dataset
    
    Returns: X_train, y_train, X_test, y_test
    """
    path="Dataset\MTS datasets\KDDCup"
    train_path = os.path.join(path, 'train.csv')
    test_path = os.path.join(path, 'test.csv')
    
    train_set = pd.read_csv(train_path, header=None)
    test_set = pd.read_csv(test_path, header=None)
    X_train, y_train = train_set.iloc[:,:-1], train_set.iloc[:,-1]
    X_test, y_test = test_set.iloc[:,:-1], test_set.iloc[:,-1]
    
    return X_train, y_train, X_test, y_test

def downsample(dataset, labels, normal_label, anomaly_ratio, random_state=42):
    normals = dataset.loc[labels==normal_label]
    abnormals = dataset.loc[labels!=normal_label]
    anom_count = int(np.floor(len(normals)/(1-anomaly_ratio))-len(normals))
    abnormals_sample = abnormals.sample(n=anom_count, random_state=random_state)
    downsampled = pd.concat([normals, abnormals_sample]).sample(frac=1, random_state=random_state)
    labels_downsampled = labels.loc[downsampled.index]
    
    return downsampled, labels_downsampled
     
def selective_downsample(dataset, labels, normal_label, anomaly_label, anomaly_ratio, random_state=42):
    normals = dataset.loc[labels==normal_label]
    abnormals = dataset.loc[labels==anomaly_label]       
    anom_count = int(np.floor(len(normals)/(1-anomaly_ratio))-len(normals))
    if anom_count > len(abnormals):
        downsampled = pd.concat([normals, abnormals]).sample(frac=1, random_state=random_state)
        labels_downsampled = labels.loc[downsampled.index]
    else:
        abnormals_sample = abnormals.sample(n=anom_count, random_state=random_state)
        downsampled = pd.concat([normals, abnormals_sample]).sample(frac=1, random_state=random_state)
        labels_downsampled = labels.loc[downsampled.index]
    return downsampled, labels_downsampled

def resample(dataset, labels, normal_label, anomaly_ratio, random_state=42):
    normals = dataset.loc[labels==normal_label]
    abnormals = dataset.loc[labels!=normal_label] 
    n = len(dataset)
    n_normal = (1-anomaly_ratio)*n
    n_abnormal = n - n_normal
    
    normal_samples = normals.sample(n=n_normal, random_state=42, replace=True)
    abnormal_sample = abnormals.sample(n=n_abnormal, random_state=42)
    resampled = pd.concat([normal_samples, abnormal_sample]).sample(frac=1, random_state=random_state)
    resampled_labels = labels.loc[resampled.index]
    return resampled, resampled_labels
    
def visualize(mse, percentiles=[]):
    plt.figure(figsize=(20, 5))
    if mse.numpy().flatten().shape[0] > 50000:
        plt.plot(mse[::300], marker='o', label='Anomaly Score', markersize=1)
    else: 
        plt.plot(mse, marker='o', label='Anomaly Score', markersize=1)
    for p in percentiles:
        plt.axhline(y=np.percentile(mse, p), color='r', linestyle='--', label='Threshold')
    plt.title("Anomaly Score per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.show()

def display_metrics(y_preds, y_tests):
        precision, recall = Precision(), Recall()
        precision.update_state(y_true=y_tests, y_pred=y_preds)
        recall.update_state(y_true=y_tests, y_pred=y_preds)
        p = precision.result().numpy()
        r = recall.result().numpy()
        f1 = 2*(p * r) / (p + r) if p+r > 0 else 0
        print(f'F1: {f1:.4f}, Precision: {p:.4f}, Recall: {r:.4f}')

def visualize_metrics(y_scores, y_true):
        precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)  # PR‐AUC
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        plt.figure()
        plt.plot(recalls, precisions, label=f"PR Curve (AP={avg_precision:.3f})")
        sc=plt.scatter(recalls, precisions, c=f1_scores, cmap='viridis', label='F1 Score')
        plt.colorbar(sc, label="F1 Score")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend()
        plt.show()
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0,1], [0,1], 'k--', label="Random guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

def visualize_scores_through_epochs(scores):
    f1_scores = [s[0] for s in scores]
    roc_scores = [s[1] for s in scores]
    pr_scores = [s[2] for s in scores]
    epochs = range(1, len(scores) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, f1_scores, marker='o', label='F1 Score')
    plt.plot(epochs, roc_scores, marker='s', label='ROC-AUC')
    plt.plot(epochs, pr_scores, marker='d', label='PR-AUC')

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Scores over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def get_time_series_between(start: datetime, end: datetime, time_column, data: pd.DataFrame) -> pd.DataFrame:
    mask = (data[time_column] >= start) & (data[time_column] <= end)
    series = data[mask].copy()
    return series
