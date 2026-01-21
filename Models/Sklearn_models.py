from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, IncrementalPCA
import tensorflow as tf
import numpy as np


class IsolationForest_MTS():
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', random_state=None):
        self.random_state = random_state
        self.contamination = contamination
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.fitted = False
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state
        )

    def _flatten_window(self, window):
        # windows: (window_size, n_features) : (T, F)
        w, f = window.shape
        return window.reshape(-1, w * f)

    def _flatten_windows(self, windows):
        # windows: (n_windows, window_size, n_features) : (B, T, F)
        n_w, w, f = windows.shape
        return windows.reshape(n_w, w * f)

    def fit(self, windows):
        if isinstance(windows, tf.data.Dataset):
            X_flat = np.concatenate(
                [self._flatten_window(x.numpy()) for x in windows], axis=0)
        else:
            X_flat = self._flatten_windows(windows)
        self.model.fit(X_flat)
        self.fitted = True
        # store threshold and other attrs if needed
        self.threshold_ = getattr(self.model, "threshold_", None)

    def score(self, windows):
        if isinstance(windows, tf.data.Dataset):
            X_flat = np.concatenate(
                [self._flatten_window(x.numpy()) for x in windows], axis=0)
        else:
            X_flat = self._flatten_window(windows)
        # sklearn: lower decision_function -> more anomalous
        # decision_function returns higher = more normal; we invert to make higher = more anomalous
        df = self.model.decision_function(X_flat)        # higher = more normal
        # make anomaly_score such that larger -> more anomalous
        anomaly_score = -df
        # optionally obtain binary labels (-1 anomaly, 1 inlier)
        labels = self.model.predict(X_flat)  # -1 anomaly, 1 inlier
        return {
            'anomaly_score': anomaly_score,
            'decision_function': df,
            'labels': labels
        }


class PCA_AE_for_MTS:
    def __init__(self, n_components, window_size, use_incremental=False):
        self.n_components = n_components
        self.window_size = window_size
        self.use_incremental = use_incremental
        self.pca = None
        if use_incremental:
            self.pca = IncrementalPCA(n_components=n_components)
        else:
            self.pca = PCA(n_components=n_components)
        self.fitted = False

    def _flatten_window(self, window):
        # windows: (window_size, n_features) : (T, F)
        w, f = window.shape
        return window.reshape(-1, w * f)

    def _flatten_windows(self, windows):
        # windows: (n_windows, window_size, n_features) : (B, T, F)
        n_w, w, f = windows.shape
        return windows.reshape(n_w, w * f)

    def fit(self, windows):
        """
        X: (T, n_features) training time series (preferably mostly normal)
        """
        if isinstance(windows, tf.data.Dataset):
            X_flat = np.concatenate(
                [self._flatten_window(x.numpy()) for x in windows], axis=0)
        else:
            X_flat = self._flatten_window(windows)
        # fit PCA
        self.pca.fit(X_flat)
        # store component variances for score distance (sd)
        self.explained_variance_ = getattr(
            self.pca, "explained_variance_", None)
        self.fitted = True

    def reconstruct_flat(self, X_flat_scaled):
        # return reconstructed (scaled) flat windows
        scores = self.pca.transform(X_flat_scaled)  # (n_windows, n_components)
        # back in scaled flattened space
        recon = self.pca.inverse_transform(scores)
        return recon, scores

    def score(self, windows):
        """
        X: (T, n_features) - input time series (train or test)
        returns:
          window_scores: dict with keys 'od' (orthogonal distance MSE per window),
                                   'sd' (score distance T2 per window),
                                   'combined' (some combination)
          windows_start_indices: list of start indices for mapping
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before compute_window_scores().")
        if isinstance(windows, tf.data.Dataset):
            X_flat = np.concatenate(
                [self._flatten_window(x.numpy()) for x in windows], axis=0)
        else:
            X_flat = self._flatten_windows(windows)
        recons, scores = self.reconstruct_flat(X_flat)
        # orthogonal distance (residual) per window: use squared norm per row
        residuals = X_flat - recons
        od = np.mean(residuals**2, axis=1)  # MSE per window
        # score distance (Hotelling's T2): sum((score_i^2) / lambda_i)
        # Use explained_variance_ (component variances). If not available (IncrementalPCA),
        # approximate with var(scores, axis=0).
        lambdas = self.explained_variance_
        if lambdas is None:
            lambdas = np.var(scores, axis=0, ddof=1)
        # avoid zero division
        lambdas = np.maximum(lambdas, 1e-12)
        sd = np.sum((scores**2) / lambdas.reshape(1, -1), axis=1)
        # simple combined score (normalized)
        od_norm = (od - np.median(od)) / \
            (np.median(np.abs(od - np.median(od))) + 1e-12)
        sd_norm = (sd - np.median(sd)) / \
            (np.median(np.abs(sd - np.median(sd))) + 1e-12)
        combined = od_norm + sd_norm
        return {
            'anomaly_score': od,
            'sd': sd,
            'combined': combined
        }
