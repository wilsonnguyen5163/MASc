import sklearn
import numpy as np
import pandas as pd
import functions
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import math
import matplotlib.gridspec as gridspec
import xgboost as xgb
import traceback
from tslearn.metrics import dtw
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor, DMatrix
from collections import Counter
from matplotlib.patches import Patch


def on_key(event):
    if event.key == "enter":
        plt.close(event.canvas.figure)


class TDG:
    threshold = 0.7
    @staticmethod
    def candidate_gen(time: datetime.datetime, data: pd.DataFrame):
        """
        Generate all 24hr timeseries of the same day of the week for all weeks preceeding this time

        Args:
            time: datetime for a timestep
            data: full electrical load dataset
        """
        candidates = pd.DataFrame()
        try:
            i = 0
            while True:
                end_interval = time - datetime.timedelta(weeks=i)
                start_interval = end_interval - datetime.timedelta(hours=24)
                slice = functions.get_time_series_between(
                    start_interval, end_interval, time_column='Timestamp', data=data)
                if slice.empty:
                    break
                candidates = pd.concat([slice, candidates])
                i += 1
        except:
            # Error before obtaining any time series, most likely due to function import errors
            if candidates.empty:
                return None
        return candidates

    @staticmethod
    def training_pool_selection(time: datetime, loads_t: pd.DataFrame, candidates: pd.DataFrame):
        """
        Sort candidates 24hr timeseries based on DTW score to 24hr preceeding timestep t

        Args:
            time (datetime): time for current timestep t
            loads_t (pd.DataFrame): 24hr timeseries preceeding 'time'
            candidates (pd.DataFrame): list of candidates 24hr timeseries

        Returns:
            [[datetime1, dtw_score1], ... ], sorted in ascending dtw_score
        """
        distances = []
        while True:
            # Preventing trivial dwt computation (dtw(current, current): is 0 )
            time -= datetime.timedelta(weeks=1)
            # Obtain the preceeding 24h of the current week's focused timeframe
            curr_24h = candidates[(candidates['Timestamp'] >= time - datetime.timedelta(
                hours=24)) & (candidates['Timestamp'] <= time)]
            if curr_24h.empty:
                break
            if curr_24h.shape[0] != loads_t.shape[0]:
                continue
            formatted2 = curr_24h.drop(columns=['Timestamp'])
            # Compute DTW distance between 2 24hr timeseries
            distances.append(
                [time, dtw(loads_t.values.flatten(), formatted2.values.flatten())])
        distances = np.array(distances)

        sorted_distance = distances[distances[:, 1].argsort()]

        # Return 24hrs timeseries sorted from most-similar to least
        return sorted_distance

    @staticmethod
    def feature_reinforce(col: pd.DataFrame):
        """
        Feature reinforce each timeseries, replacing with average loads among the 24hr if the load at a resolution is not max / min
        Args:
            col (pd.Dataframe): pd.Dataframe
        """
        description = col.describe()
        v_max = description['max']
        v_min = description['min']
        v_avg = description['mean']
        return col.apply(lambda x: x if x == v_min or x == v_max else v_avg)

    @staticmethod
    def noisy_clustering(feature_reinforced_data: pd.DataFrame, k=2):
        kmeans = KMeans(n_clusters=k, random_state=123)
        assignments = kmeans.fit_predict(feature_reinforced_data)
        return assignments

    @staticmethod
    def sequence_extraction(training_pool, data: pd.DataFrame):
        """
        Iterate through the training_pool, formatted like: (Timestamp('2024-06-08 18:45:00'), 17.0509380648), ...
        Obtain the previous 24hr ending with the 'timestamp' time, flatten through feature_reinforce

        Args:
            training_pool: [
                (Timestamp('2024-06-08 18:45:00'), 17.0509380648), ...]
            data (pd.DataFrame): full dataset to obtain the load values of a 24hr timeseries

        Returns:
            pd.Dataframe : n+1 feature-reinforced 24hr timeseries
        """
        sequences = pd.DataFrame()
        for timestamp, _ in training_pool:
            series = data[(data['Timestamp'] >= timestamp -
                           datetime.timedelta(hours=24)) & (data['Timestamp'] <= timestamp)]
            series_feature_reinforced = series.drop(
                columns=['Timestamp']).apply(TDG.feature_reinforce)
            sequence = pd.DataFrame(
                [series_feature_reinforced.T.values.flatten()], index=[timestamp])
            sequences = pd.concat([sequences, sequence])
        sequences.index.name = "Timestamp"
        return sequences

    @staticmethod
    def training_data_generate(time: datetime, n_intervals, data: pd.DataFrame, loads_t):
        """
        Main TDG module
        Performs candidate generation, training pool selection, and sequence extract to form the training data for a timestamp t

        Args:
            time: actual time for current timestep t
            n_intervals: number of 24hr timeseries desired for training
            data: full electrical load dataset
            threshold: threshold to evaluate silhouette coefficient of kmeans clustering
            loads_t: 24hr timeseries preceeding 'time' of observed loads
        """
        # Error generating
        if (candidates := TDG.candidate_gen(time-datetime.timedelta(weeks=1), data)) is None:
            return None
        sorted_candidates = TDG.training_pool_selection(
            time, loads_t, candidates)

        # Obtain n+1 most similar candidates 24hr timeseries
        training_pool = sorted_candidates[:n_intervals+1]

        # Applying feature reinforced to n+1 most similar candidates 24hr timeseries
        reinforced_candidates = TDG.sequence_extraction(
            training_pool, candidates)
        reinforced_assignments = TDG.noisy_clustering(reinforced_candidates)

        # Current index at training pool to add more later if kmeans clustering fails
        curr_ind = n_intervals+1

        # Process repeats until n timeseries is obtained (note, starts with n+1)
        while len(training_pool) != n_intervals:
            score = silhouette_score(
                reinforced_candidates, reinforced_assignments)
            if (score < TDG.threshold):
                training_pool = np.delete(
                    training_pool, np.random.randint(0, n_intervals+1), axis=0)
            else:
                counter = Counter(reinforced_assignments)
                cluster = counter.most_common(1)[0][0]
                # if most common cluster have >= n items, drop the timeseries on the other cluster (that other cluster is considered noisy)
                if counter.most_common(1)[0][1] >= n_intervals:
                    training_pool = np.delete(training_pool, np.where(
                        reinforced_assignments != cluster), axis=0)

                # Otherwise, drop the timeseries that belongs to least common cluster and replace it with next most-common timeseries from DTW step
                else:
                    training_pool = np.delete(training_pool, np.where(
                        reinforced_assignments != cluster), axis=0)

                    try:
                        # Compute how many replacements needed
                        diff = n_intervals+1-len(training_pool)

                        # Pull replacements from the next most-commons
                        training_pool.append(
                            sorted_candidates[curr_ind:curr_ind+diff])
                        curr_ind += diff
                    except:
                        return None

                    # Redo the extractions
                    reinforced_candidates = TDG.sequence_extraction(
                        training_pool, candidates)
                    reinforced_assignments = TDG.noisy_clustering(
                        reinforced_candidates)

        # Goal is to have n noisefree 24hr timeseries for training. Must revert back to dataframe format for training
        training_data = pd.DataFrame()
        for timestamp, _ in training_pool:
            series = data[(data['Timestamp'] >= timestamp -
                           datetime.timedelta(hours=24)) & (data['Timestamp'] <= timestamp)]
            series = series.drop(columns=['Timestamp'])
            series = pd.DataFrame([series.values.flatten()], index=[timestamp])
            training_data = pd.concat([training_data, series])
        training_data.index.name = 'Timestamp'
        return training_data


class OSAP:
    learning_rate = 0.1
    num_rounds = 100
    hessian_scaling = 0.2
    params = {
        "max_depth": 6,
        "eta": 0.3,
        "objective": "reg:squarederror",  # overridden by custom objective
    }
    
    def base_quantile_regression_model(X_train_regressor, X_train_response, quantile):
        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile, n_estimators=OSAP.num_rounds, learning_rate=0.1)
        model.fit(X_train_regressor, X_train_response)
        return model
    
    @staticmethod
    def custom_quantile_regression_model(X_train_regressor, X_train_response, quantile):
        d_train = DMatrix(X_train_regressor, X_train_response)
        model = xgb.train(OSAP.params, d_train, num_boost_round=OSAP.num_rounds, obj=OSAP.quantile_loss(quantile))
        return model

    @staticmethod
    def quantile_loss(quantile):
        """
        Quantile loss function for XGBoost.
        Minimizing the quantile loss function w.r.t to the predicted y_predictions
        """
        def objective(preds, dtrain):
            labels = dtrain.get_label()
            errors = preds-labels
            grad = np.where(errors>0, errors * (1-quantile), quantile*errors)
            hess = np.full_like(preds, OSAP.hessian_scaling)  # hessian is constant for pinball loss
            return grad, hess
        return objective

    @staticmethod
    def data_formatter(X_train: pd.DataFrame):
        """
        Splits training data into l-1 regressors to predict l-th column
        Treats X_train as a table of features, each row is a tuple, regressing on last column
        Args:
            X_train (pd.DataFrame): DataFrame of n 24hr timeseries

        Returns:
            X (pd.DataFrame): features
            y (pd.DataFrame): target response for regression
        """
        X_regressors = X_train.iloc[:, :-1]
        y_response = X_train.iloc[:, -1]
        return X_regressors, y_response

    @staticmethod
    def one_step_ahead_prediction_interval(quantile, loads_t, X_train):
        """
        Main OSAP module
        Generate initial prediction interval for timestep t+1, given observed load at time t and dataset generated by TDG for time t
        Args:
            loads_t : Raw Power Consumption (kW) 24hr timeseries before time t
            X_train: 24hr timeseries generated by TDG, most similar to the 24hr leading up to time t
            quantile: tunable parameter

        Returns:
            y_low: interval lower bound
            y_high: interval higher bound
        """
        
        if loads_t.shape[1] != X_train.shape[1]:
            if loads_t.shape[0] != X_train.shape[1]:
                return None
            loads_t = loads_t.reset_index()
            loads_t = loads_t.drop(columns=['index'], axis=1).T
        high = np.max([quantile, 1-quantile])
        low = np.min([quantile, 1-quantile])
        X_train_regressors, X_train_response = OSAP.data_formatter(X_train)
        high_model = OSAP.custom_quantile_regression_model(
            X_train_regressors, X_train_response, high)
        low_model = OSAP.custom_quantile_regression_model(
            X_train_regressors, X_train_response, low)
        shifted = loads_t.iloc[:, 1:]
        shifted.columns = list(range(X_train_regressors.shape[1]))
        if (type(high_model) == xgb.core.Booster):
            shifted = DMatrix(shifted)
        prediction_low = low_model.predict(shifted)
        prediction_high = high_model.predict(shifted)
        return prediction_low[0], prediction_high[0]


class LBO:
    N = 5
    beta=1
    
    # Since the algorithm is scaling all initial prediction interval to be within D_load_min, D_load_max,
    # especially during correct predictions for earlier quantiles
    # the action destroys the initial interval prediction original size, which subsequent cases 2 and 3 needed for translation
    last_mid_intervals = 0
    
    @staticmethod
    def refine_interval(r_initial: list[float, float], L_t, R_t, X_load_max, X_load_min):
        """
        Refines the initial prediction for timestep t+1 based on recently observed load value at time t
        Args:
            r_initial: The initial prediction interval for t+1 timestep. Will be optimized based on current timestep
            L_t: observed load value at timestep t
            R_t: prediction interval for timestep t
            X_load_max: maximum load value within training data obtained by TDG
            X_load_min: minimum load value within training data obtained by TDG

        Returns:
            opt_interval: refined t+1 prediction interval
            case number: use to keep track of previous n-prediction cases
        """
        R_l, R_h = R_t[0], R_t[1]

        if (0 < R_l < X_load_min):
            if (math.isclose(R_l, X_load_min, rel_tol=0.15, abs_tol=0)):
                R_l = X_load_min
            elif (R_l == 0.5):
                R_l = X_load_min
            else:
                LBO.last_mid_intervals = (r_initial[1] - r_initial[0])/2
                r_initial = [np.max([r_initial[0], X_load_min]),
                            np.min([r_initial[1], X_load_max])]
                return r_initial, 4

        if (R_h > X_load_max):
            if (math.isclose(R_h, X_load_max, rel_tol=0.15, abs_tol=1)):
                R_h = X_load_max
            else:
                LBO.last_mid_intervals = (r_initial[1] - r_initial[0])/2
                r_initial = [np.max([r_initial[0], X_load_min]),
                         np.min([r_initial[1], X_load_max])]
                return r_initial, 4
        
        # Case 4 - Anomaly
        if not (X_load_min <= L_t <= X_load_max):
            LBO.last_mid_intervals = (r_initial[1] - r_initial[0])/2
            r_initial = [np.max([r_initial[0], X_load_min]),
                         np.min([r_initial[1], X_load_max])]
            return r_initial, 4
        
        # Case 1: Accurate prediction at timestep t, and prediction interval is between [min load, max load]
        if (R_l <= L_t <= R_h) and (X_load_min <= R_l and R_h <= X_load_max):
            LBO.last_mid_intervals = (r_initial[1] - r_initial[0])/2
            r_initial = [np.max([r_initial[0], X_load_min]),
                         np.min([r_initial[1], X_load_max])]
            return r_initial, 1
        # Case 2: Inaccurate prediction at timestep t, prediction interval is between [min load, max load], and there is enough room to shift the interval to bound L(t)
        elif (not (R_l <= L_t <= R_h)) and (X_load_min <= R_l and R_h <= X_load_max):
            error_p = L_t - (R_h- LBO.last_mid_intervals)
            if (np.min([L_t - X_load_min, X_load_max - L_t]) >= LBO.last_mid_intervals):
                LBO.last_mid_intervals = (r_initial[1] - r_initial[0])/2
                r_initial = [np.max([r_initial[0]+error_p, X_load_min]),
                         np.min([r_initial[1]+error_p, X_load_max])]
                return r_initial, 2
            # Case 3: Inaccurate prediction at timestep t, prediction interval is between [min load, max load], and there is NOT enough room to shift the interval to bound L(t)
            else:
                r_initial = [np.max([r_initial[0]+error_p, X_load_min]),
                         np.min([r_initial[1]+error_p, X_load_max])]
                LBO.last_mid_intervals = (r_initial[1] - r_initial[0])/2
                return r_initial, 3
        

        # Should never get here
        return None, None

    @staticmethod
    def refine_z(curr_z: int, prev_cases: list[int], curr_timestep_case: int):
        """
        Refining z for quantile computation of the timestep t+2
        Refines based on most recent n predictions

        Args:
            curr_z (int): current z value
            prev_cases (list[int]): previous n prediction cases
            curr_timestep_case (int): prediction case for timestep t
            beta (int): tunable parameter

        Returns:
            opt_z: modified
        """
        if len(prev_cases) < LBO.N:
            if curr_timestep_case == 1:
                return curr_z - prev_cases.count(1)
            elif curr_timestep_case == 2:
                return curr_z + prev_cases.count(2)
            elif curr_timestep_case == 3:
                return curr_z - LBO.beta*prev_cases.count(3)
        else:
            if curr_timestep_case == 1:
                return curr_z - prev_cases[-LBO.N:].count(1)
            elif curr_timestep_case == 2:
                return curr_z + prev_cases[-LBO.N:].count(2)
            elif curr_timestep_case == 3:
                return curr_z - LBO.beta*prev_cases[-LBO.N:].count(3)
        return None

    @staticmethod
    def lookback_optimizer(X_train_t: pd.DataFrame, L_t: float, r_initial: list[float], curr_z: int, prev_cases: list[int], prev_intervals):
        """
        Main LBO Module
        Analyze & tune parameters for next timestep & tracking recent prediction results

        Args:
            X_train_t (pd.DataFrame): training data generated from
            prev_loads_t (pd.Dataframe): recorded load value of the previous 24hr to timestep t
            L_t (float): observed load value at timestep t
            r_initial (list[float]): initial prediction interval for timestep t+1
            curr_z (int): current z value at timestep t+1 that needs to be updated for t+2 prediction
            prev_cases (list[int]): list of previous n prediction cases
            prev_intervals: previous intervals, note that the intervals are 1 step ahead of cases
            beta (_type_): hyperparameter, for scaling the change of z

        Returns:
            opt_R : Optimized prediction interval for timestep t+1
            new_z : new z value for quantile regression at time t+2
            anomaly_flag : use for anomaly analysis
        """
        R_t = prev_intervals[-1]
        opt_R, case_t = LBO.refine_interval(
            r_initial, L_t, R_t, X_train_t.values.max(), X_train_t.values.min())
        new_z = LBO.refine_z(curr_z, prev_cases, case_t)
        return opt_R, new_z, case_t

    def anomaly_analysis(loads_t: pd.DataFrame, X_train_t):
        """
        Compute the first-order difference between consecutive load values, then compare to the difference between loads at timestep t and t-1
        If the difference between timestep t and t-1 falls within the load differences seen in training data, then the anomaly is probably false flag
        Otherwise, it is anomalous as the load change is too drastic

        Args:
            loads_t :  pd.Dataframe of load values of 24hr preceeding time t (including load observed at time t)
            X_train_t: training data generated by TDG, most similar 24hr timeseries to the current 24hr time block

        Returns:
            boolean (0,1) : 0 = Normal, 1 = Anomaly
        """
        values = X_train_t.to_numpy()
        first_order_diff = values[:, :-1] - values[:, 1:]
        max_diff = np.max(first_order_diff)
        min_diff = np.min(first_order_diff)
        
        load_diff = loads_t.values[-1] - loads_t.values[-2]
        if (min_diff <= load_diff[0] <= max_diff):
            return 0
        return 1


class UnifiedFramework:
    # ************* Hyperparameters *******************
    M = 1000  # quantile k calculation
    N = 10  # number of previous predictions
    n = 7  # number of timeseries
    beta = 10
    threshold = 0.7 # TDG threshold
       
    learning_rate = 0.1 # XGBoost quantile regression learning rate
    num_rounds = 100    # XGBoost quantile regression 
    hessian_scaling = 0.2 # custom Hessian scaling
    
    # Parameter for XGBoost quantile regressions
    params = {
        "max_depth": 6,
        "eta": 0.3,
        "objective": "reg:squarederror",  # WILL overridden by custom objective
    }
    # ************************************************
    
    # Number of observed loads per 24hr, (resolution of 96 means 15-min interval between observed loads)
    resolution = 96

    def __init__(self, database: pd.DataFrame, unit):
        self.pred_intervals = []
        self.pred_cases = []
        self.database = database
        self.recorded_loads = None
        self.initialized = False
        self.curr_t = None
        self.load_unit = unit
        
        self.z_t = UnifiedFramework.M
        LBO.N = UnifiedFramework.N
        LBO.beta = UnifiedFramework.beta
        OSAP.hessian_scaling = UnifiedFramework.hessian_scaling
        OSAP.learning_rate = UnifiedFramework.learning_rate
        OSAP.num_rounds = UnifiedFramework.num_rounds
        TDG.threshold = UnifiedFramework.threshold
        

    def visualize(self, loads_t: pd.DataFrame, title: str, timestep, pred_intervals: list[list[float]]):
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        plt.plot(range(loads_t.shape[1]), loads_t.to_numpy()[0], color='blue')
        plt.plot(range(loads_t.shape[1]), loads_t.to_numpy()[0], linestyle='None', label='Observed Load',
                 color='blue', marker='o', markersize=2, markerfacecolor='white', markeredgecolor='red')
        if pred_intervals != None:
            # Wants to visualize the prediction boundary of all previous steps
            for i, interval in enumerate(reversed(pred_intervals)):
                (lower, upper) = interval
                if i == loads_t.shape[1]+1 or i == len(pred_intervals)-1 or i == len(pred_intervals)-2:
                    break
                if i == 0:  # Initial prediction for next load
                    plt.fill_between([loads_t.shape[1]-0.5-i, loads_t.shape[1]+0.5-i],
                                     lower, upper, color='teal', alpha=0.3)
                else:
                    color = 'green' if (
                        lower <= loads_t.iloc[0, -i] <= upper) else 'red'
                    plt.fill_between([loads_t.shape[1]-0.5-i, loads_t.shape[1]+0.5-i],
                                     lower, upper, color=color, alpha=0.3)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.xlim(0, len(loads_t.to_numpy()[0])+30)
        plt.xlabel("Time Step")
        plt.ylabel("Observed Load "+self.load_unit)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.legend(handles=[
            plt.Line2D([0], [0], color='white', marker='o', markerfacecolor='white',
                       markeredgecolor='red', label='Load', linestyle=None, markersize=3),
            Patch(facecolor='teal', label="t+1 prediction", alpha=0.2),
            Patch(facecolor='green', label="good interval", alpha=0.2),
            Patch(facecolor='red', label="poor prediction", alpha=0.2),

        ], loc='lower right')
        plt.show()

    def run(self, new_data):
        for i in range(new_data.shape[0]):
            try:
                time_t = new_data.iloc[i, 0]
                load_t = new_data.iloc[i, 1]
                if not self.initialized:
                    # Read a previous 24h, one timestep before the training time
                    self.initialized = True
                    self.recorded_loads = functions.get_time_series_between(
                        time_t-datetime.timedelta(hours=24, minutes=15), time_t-datetime.timedelta(minutes=15), 'Timestamp', self.database).iloc[:, 1].reset_index().drop(columns=['index'])
                    D_t = TDG.training_data_generate(time_t-datetime.timedelta(
                        minutes=15), UnifiedFramework.n, self.database, self.get_most_current_24hr())
                    
                    # intialize the first prediction to be maximal, to assume perfect fit
                    self.pred_intervals.append(
                        [self.database["Observed Load"].values.min(), self.database['Observed Load'].values.max()])
                    

                    # Intialize first initial prediction for time t (starting time)
                    R_o_low, R_o_high = OSAP.one_step_ahead_prediction_interval(
                        1, self.recorded_loads, D_t)
                    
                    opt_intervals, _, _ = LBO.lookback_optimizer(D_t, self.recorded_loads.values[-1][0], [
                                                                R_o_low, R_o_high], self.get_quantile(), self.pred_cases, self.pred_intervals)
                    self.update_cases_intervals(1, opt_intervals)

                # Add load_t to database of observed_loads
                self.database.loc[len(self.database)] = {
                    "Timestamp": time_t, "Observed Load": load_t}
                self.recorded_loads.loc[len(self.recorded_loads)] = load_t
                loads_t = self.get_most_current_24hr()
                D_t = TDG.training_data_generate(
                    time_t, UnifiedFramework.n, self.database, loads_t)
                
                R_o_low, R_o_high = OSAP.one_step_ahead_prediction_interval(
                    self.get_quantile(), loads_t, D_t)
                
                    
                opt_intervals, self.z_t, case_t = LBO.lookback_optimizer(D_t, load_t, [
                                                                        R_o_low, R_o_high], self.z_t, self.pred_cases, self.pred_intervals)

                # Anomaly
                if (case_t == 4):
                    self.z_t = UnifiedFramework.M
                    
                    if LBO.anomaly_analysis(loads_t, D_t):
                        print('*******************Anomaly detected************************')
                
                print("R(t): {},    L(t): {:.3f},    case_t: {}, Dmin:{:.3f}, Dmax:{:.3f}".format(
                    self.pred_intervals[-1], load_t, case_t, D_t.values.min(), D_t.values.max()))

                print("Raw interval: {},     next interval prediction {},    quantile = {},     z(t) = {}\n".format([R_o_low, R_o_high],
                    opt_intervals, self.get_quantile(), self.z_t))
                if (case_t == 5):
                    raise RuntimeError(
                        "Case 5 occured during non-initializing interval tuning")
                
                self.update_cases_intervals(case_t, opt_intervals)
                self.visualize(loads_t.T, "Forecasting results",
                            None, self.pred_intervals)
            except AttributeError as e:
                traceback.print_exc()
                print(self.database.values, self.database.shape)
                print(loads_t.values, loads_t.shape)
                continue

    def get_quantile(self) -> float:
        return (1/(1+math.pow(math.e, (-self.z_t/100))))

    def run_real_time(self):
        pass

    def get_most_current_24hr(self) -> pd.DataFrame:
        return self.recorded_loads.iloc[-UnifiedFramework.resolution-1:].reset_index().drop(columns=['index'])

    def update_cases_intervals(self, new_case, new_interval):
        """
        Updating the list of n previous steps
        """
        self.pred_cases.append(new_case)
        self.pred_intervals.append(new_interval)

class test_TDG:
    def __init__(self, database):
        self.database = database

    def visualize(self, loads_t:pd.Series, loads_t_reinforced:pd.Series, candidates:pd.DataFrame, feature_reinforced:pd.DataFrame)->None:
        fig = plt.figure(figsize=(16,10))
        gs = gridspec.GridSpec(3, 5, figure=fig)
        ax1 = fig.add_subplot(gs[0,0:2])
        ax2 = fig.add_subplot(gs[0,2:5])
        
        axes_rows2 = [fig.add_subplot(gs[1,i]) for i in range(len(candidates))]
        axes_rows3 = [fig.add_subplot(gs[2,i]) for i in range(len(candidates))]
        
        
        ax1.plot(range(loads_t.shape[1]), loads_t.to_numpy()[0], color='blue')
        ax2.plot(range(loads_t_reinforced.shape[1]), loads_t_reinforced.to_numpy()[0], color='blue')
        
        for i, ax in enumerate(axes_rows2):
            ax.plot(range(candidates.shape[1]), candidates.iloc[i].T.to_numpy(), color='blue')
        
        for i, ax in enumerate(axes_rows3):
            ax.plot(range(feature_reinforced.shape[1]), feature_reinforced.iloc[i].T.to_numpy(), color='blue')
            
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()
        plt.show()
    
    def visualize_clustering(self):
        pass
    
    def run(self):
        time = pd.to_datetime("2018-10-05 01:00:00")
        candidates = TDG.candidate_gen(time-datetime.timedelta(weeks=1), self.database)
        loads_t  = functions.get_time_series_between(time-datetime.timedelta(hours=24), time, 'Timestamp', self.database).iloc[:, 1].reset_index().drop(columns=['index'])
        sorted_candidates = TDG.training_pool_selection(time, loads_t, candidates)
        training_pool = sorted_candidates[:5]
        series = pd.DataFrame()
        for timestamp, _ in training_pool:
            seq = self.database[(self.database['Timestamp'] >= timestamp -
                           datetime.timedelta(hours=24)) & (self.database['Timestamp'] <= timestamp)].drop(columns=['Timestamp'])
            seq = pd.DataFrame([seq.T.values.flatten()], index=[timestamp])
            series = pd.concat([series, seq])
        series.index.name = 'Timestamp'
        
        reinforced_candidates = TDG.sequence_extraction(training_pool, candidates)
        loads_t_reinforced = loads_t.apply(TDG.feature_reinforce)
        self.visualize(loads_t.T, loads_t_reinforced.T, series, reinforced_candidates)
        
        
        
if __name__ == "__main__":
    X_full_train, X_full_test, load_unit = functions.load_apartment_electrical_dataset()

    framework = UnifiedFramework(X_full_train, load_unit)
    framework.run(X_full_test)
    '''
    tester = test_TDG(X_full_train)
    tester.run()
    framework = UnifiedFramework(X_2024)
    framework.run(X_2025)
    '''
    
    
