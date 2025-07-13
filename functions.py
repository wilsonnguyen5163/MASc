import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from imblearn.over_sampling import SMOTE
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import ast


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
    """
    Load Server Machine Dataset
    
    Returns: X_train, y_train, X_test, y_test
    """
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

def load_PSM():
    """
    Load Pooled Server Metrics MTS dataset from eBay  
    
    Training set does not contain label, assumed to be exclusively normals
    
    Returns: X_train, _,  X_test, y_test
    """
    path="Dataset\MTS datasets\PooledServerMetrics"
    train_path = os.path.join(path, 'train.csv')
    test_path = os.path.join(path, 'test.csv')
    test_label_path = os.path.join(path, 'test_label.csv')
    
    train_set = pd.read_csv(train_path).iloc[:, 1:]
    test_set = pd.read_csv(test_path).iloc[:, 1:]
    y_test = pd.read_csv(test_label_path).iloc[:,1:]
    return train_set, None, test_set, y_test

def load_SMAP_MSL(channel, id):
    """
    Load SMAP / MSL dataset from NASA
    Requires channel-id for the respective telemetry data
    
    Train set is assumed exclusively normal
    
    Returns: X_train, X_test, y_test
    """
    path="Dataset\MTS datasets\MSL_SMAP"
    telemetry_sensor = f'{channel}-{id}.npy'
    train_path = os.path.join(path, 'train', telemetry_sensor)
    test_path = os.path.join(path, 'test', telemetry_sensor)
    
    train_set = np.load(train_path)
    test_set = np.load(test_path)
    descriptions = pd.read_csv(os.path.join(path, 'labeled_anomalies.csv'))
    anomaly_intervals = ast.literal_eval(descriptions[descriptions['chan_id']==f'{channel}-{id}']['anomaly_sequences'].to_numpy()[0])
    labels = np.zeros_like(test_set[:,0],dtype=int)
    for (l, h) in anomaly_intervals:
        labels[l:h+1] = 1

    return pd.DataFrame(train_set), None,  pd.DataFrame(test_set), pd.DataFrame(labels)

def load_DMDS_mixed_anomalies(train_sel, test_sel):
    """
    selections:  
        1: Oct 30, 2001
        2: Nov 9, 2001
        3: Nov 17, 2001
        4: Nov 20, 2001
        
    Dataset is not normalized
    Returns: X_train, y_train, X_test, y_test
    """
    selection_dict ={
        1: '30102001.txt',
        2: '09112001.txt',
        3: '17112001.txt',
        4: '20112001.txt',
    }
    label_dict = {
        1 : [[58800, 59800], [57340, 57890]],
        2 : [[57275, 57550], [58830, 58930], [58520, 58625], [60650, 60700], [60870, 60960]],
        3 : [[54600, 54700], [56670,56770], [53780,53794], [54193,54215], [55482,55517], [55977,56015], [57030,57072], [57475,57530], [57675,57800], [58150,58325]],
        4 : [[37780, 38400], [44400, -1]]
    }
    path = "Dataset\MTS datasets\DMDS"
    train_path = os.path.join(path, selection_dict[train_sel])
    test_path = os.path.join(path, selection_dict[test_sel])
    
    train_set = pd.read_csv(train_path, delim_whitespace=True, header=None)
    test_set = pd.read_csv(test_path, delim_whitespace=True, header=None)
    y_train = np.zeros(len(train_set), dtype=int)
    y_test = np.zeros(len(test_set), dtype=int)
    
    for (left, right) in label_dict[test_sel]:
        y_test[left:right+1] = 1
    
    for (left, right) in label_dict[train_sel]:
        y_train[left:right+1] = 1
    
    return train_set.drop(columns=[train_set.columns[0]]), y_train, test_set.drop(columns=[test_set.columns[0]]), y_test


def load_creditcard():
    """
    Load Credit Card Fraud Detection dataset    
    
    __return__:  
        X_train, y_train, X_test, y_test
    """
    
    path = "Dataset\MTS datasets\CreditCard\creditcard.csv"
    ds_full = pd.read_csv(path)
    y_full = ds_full["Class"]
    ds_full = ds_full.drop(columns=["Time", "Class"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(ds_full, y_full, test_size=0.3, random_state=42, stratify=y_full, shuffle=True)
    return X_train, y_train, X_test, y_test
    
def load_genesis():
    """
    Load Credit Card Fraud Detection dataset   
    
    Returns:  
        X_train, y_train, X_test, y_test
    """
    path = "Dataset\MTS datasets\Genesis\Genesis_AnomalyLabels.csv"
    ds_full = pd.read_csv(path)
    y_full = ds_full["Label"]
    ds_full = ds_full.drop(['Timestamp', 'Label'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(ds_full, y_full, test_size=0.3, random_state=42, stratify=y_full)
    return X_train, y_train, X_test, y_test

def load_water_pump_sensor(retain_timestamp=False):
    """
    Load water pump sensor dataset  
    
    All rows contains some sort of missing values
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    path = 'Dataset\MTS datasets\Water Pump Sensor Data\sensor.csv'
    ds_full = pd.read_csv(path)
    y_full = ds_full['machine_status'].map(lambda x : 0 if x=="NORMAL" else 1)
    if retain_timestamp:
        ds_full = ds_full.drop([ds_full.columns[0]], axis=1)
        ds_full['timestamp'] = pd.to_datetime(ds_full['timestamp'])
    else:  
        ds_full = ds_full.drop([ds_full.columns[0], 'timestamp'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(ds_full, y_full, test_size=0.3, random_state=42, stratify=y_full)
    return X_train, y_train, X_test, y_test

def inject_anomalies_from_testset(trainset:pd.DataFrame, testset:pd.DataFrame, test_labels, anomaly_ratio, random_state=42):
    """
    Assume trainset is free of anomaly, generates synthetic anomalies similar to those in test set and inject into train set  
    
    Returns: injected_X_set, injected_y_set
    """
    abnormals = testset.loc[test_labels==1]
    X_combined = pd.concat([trainset, abnormals], ignore_index=True)
    y_combined = np.concatenate([np.zeros(len(trainset), dtype=int), np.ones(len(abnormals), dtype=int)])
    smote = SMOTE(sampling_strategy={1: (len(abnormals) + int(np.floor(anomaly_ratio*len(trainset))))}, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_combined, y_combined)
    X_res = pd.DataFrame(X_res, columns=trainset.columns)
    y_res = np.asarray(y_res, dtype=int)
    
    total_abnormals = int((y_res == 1).sum())
    synthetic_abnormal_count = total_abnormals - len(abnormals)
    abnormal_indices = np.where(y_res == 1)[0]
    synthetic_abnormal_indices = abnormal_indices[-synthetic_abnormal_count:]
    synthetic_abnormals = X_res.iloc[synthetic_abnormal_indices].reset_index(drop=True)
    
    n_to_inject = len(synthetic_abnormals)
    total_length = len(trainset) + n_to_inject
    positions = np.arange(total_length)
    insert_position = np.random.RandomState(random_state).choice(positions, size=len(synthetic_abnormals), replace=False)
    insert_position.sort()
    
    normals = trainset.reset_index(drop=True)
    final_rows = []
    final_labels = []
    n_idx = 0  # pointer in normals
    a_idx = 0  # pointer in anomalies
    for pos in range(total_length):
        if a_idx < n_to_inject and pos == insert_position[a_idx]:
            final_rows.append(synthetic_abnormals.iloc[a_idx])
            final_labels.append(1)
            a_idx += 1
        else:
            final_rows.append(normals.iloc[n_idx])
            final_labels.append(0)
            n_idx += 1

    injected_train = pd.DataFrame(final_rows).reset_index(drop=True)
    injected_labels = np.array(final_labels, dtype=int)
    return injected_train, injected_labels 

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

def get_time_series_between(start: datetime, end: datetime, time_column, data: pd.DataFrame) -> pd.DataFrame:
    mask = (data[time_column] >= start) & (data[time_column] <= end)
    series = data[mask].copy()
    return series
