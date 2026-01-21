from typing import Any
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from tensorflow import keras
import numpy as np
import pandas as pd
import math
import functions
from functions import AnomalyWindowSampler

ALPHA_CLS = 0.1
ANOMALY_GEN_BETA = 0.2
FINITE_WINDOW_N = 200000  # Default value for unknown cardinality datasets

WeightNorm = tfa.layers.WeightNormalization


class Chomp1d(keras.layers.Layer):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def call(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :-self.chomp_size, :]


class TCNBlock(keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 bias=True,
                 dropout=0.2,
                 residual=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.chomp_size = int(padding)
        self.residual = residual
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Helper to build a conv1d possibly wrapped with weight norm
        def make_conv(filters):
            conv_kernel_init = tf.keras.initializers.RandomNormal(
                mean=0, stddev=0.01)
            conv = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                dilation_rate=dilation,
                padding='valid',                # we apply explicit padding via ZeroPadding1D
                use_bias=bias,
                kernel_initializer=conv_kernel_init
            )
            return WeightNorm(conv, data_init=False)

        self.pad1 = tf.keras.layers.ZeroPadding1D(padding=self.chomp_size)
        self.conv1 = make_conv(n_outputs)
        self.chomp1 = Chomp1d(self.chomp_size)
        self.relu1 = tf.keras.layers.ReLU()
        self.drop1 = tf.keras.layers.Dropout(rate=dropout)

        self.pad2 = tf.keras.layers.ZeroPadding1D(padding=self.chomp_size)
        self.conv2 = make_conv(n_outputs)
        self.chomp2 = Chomp1d(self.chomp_size)
        self.relu2 = tf.keras.layers.ReLU()
        self.drop2 = tf.keras.layers.Dropout(rate=dropout)

        # 1x1 conv to match channels if needed (downsample)
        if n_inputs != n_outputs:
            conv_kernel_init = tf.keras.initializers.RandomNormal(
                mean=0, stddev=0.01)
            self.downsample = tf.keras.layers.Conv1D(
                filters=n_outputs,
                kernel_size=1,
                padding='same',
                use_bias=bias,
                kernel_initializer=conv_kernel_init,
            )
        else:
            self.downsample = None

    def call(self, x, training=False):
        """
        Produces temporal context for the batch of timewindows. 

        Args:
            x (Tensor): (B, T, F)
            training (bool, optional): training status flag. Defaults to False.

        Returns:
            context tensor: (B, T, n_output)
        """
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out, training=training)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out, training=training)

        # residual connection
        if self.residual:
            if self.downsample is not None:
                res = self.downsample(x)
            else:
                res = x
            return out + res
        else:
            return out


class PyTorchLinearInit(keras.initializers.Initializer):
    """Initializes Dense kernels like PyTorch's nn.Linear:
       Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))."""

    def __call__(self, shape, dtype=None):
        # shape is (fan_in, fan_out) for Dense kernel
        fan_in = shape[0] if len(shape) >= 1 else 1
        limit = 1.0 / math.sqrt(max(1, int(fan_in)))
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype or tf.float32)

    def get_config(self):
        return {}


class MLPClassification(keras.layers.Layer):
    def __init__(self, hidden_dims=16, out_dim=2, use_bias=True, kernel_std=0.01, leaky_alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        pt_init = PyTorchLinearInit()

        self.fc1 = keras.layers.Dense(
            hidden_dims, use_bias=use_bias, kernel_initializer=pt_init)
        self.activation = keras.layers.LeakyReLU(alpha=leaky_alpha)

        self.fc2 = keras.layers.Dense(
            out_dim, use_bias=use_bias, kernel_initializer=pt_init)

    def call(self, x, training=False):
        x = self.fc1(x, training=training)
        x = self.activation(x)
        x = self.fc2(x, training=training)
        return x


class MLPProjector(keras.layers.Layer):
    def __init__(self, representation_dim=16, out_dim=16, use_bias=True, kernel_std=0.01, leaky_alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        pt_init = PyTorchLinearInit()

        self.fc1 = keras.layers.Dense(
            representation_dim, use_bias=use_bias, kernel_initializer=pt_init)
        self.fc1_bypass = keras.layers.Dense(
            representation_dim, use_bias=use_bias, kernel_initializer=pt_init)
        self.activation = keras.layers.LeakyReLU(alpha=leaky_alpha)
        self.fc2 = keras.layers.Dense(
            out_dim, use_bias=use_bias, kernel_initializer=pt_init)

    def call(self, x, training=False):
        """
        Project the context into feature space of out_dim. Note: project then pool approached is used

        Args:
            x (Tensor): (B, T, F)
            training (bool, optional): training status flag. Defaults to False.

        Returns:
            z (Tensor): Main projection
            z_prime (Tensor) : 
        """
        hidden_rep = self.fc1(x, training=training)
        hidden_rep = self.activation(hidden_rep)
        z = self.fc2(hidden_rep, training=training)

        hidden_bypass = self.fc1_bypass(x, training=training)
        hidden_bypass = self.activation(hidden_bypass)
        z_prime = self.fc2(hidden_bypass, training=training)

        return z, z_prime


class COUTA(keras.Model):
    def __init__(self, input_dim, hidden_dims=32, emb_dim=10, kernel_size=2, tcn_bias=True, dropout_rate=0.2, 
                 kernel_regularizer=None, alpha_cls=0.1,  clipnorm=1.0, early_stop_patience=10, batch_size=64):
        super().__init__()
        self.clip_norm = clipnorm
        self.patience = early_stop_patience
        self.alpha_cls = alpha_cls
        self.tcn_layers = []
        self.batch_size = batch_size
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        for i in range(len(hidden_dims)):
            dilation = 2**i
            padding = (kernel_size-1)*dilation
            in_c = input_dim if i == 0 else hidden_dims[i-1]
            out_c = hidden_dims[i]
            self.tcn_layers += [TCNBlock(in_c, out_c, kernel_size=kernel_size, stride=1, dilation=dilation,
                                         padding=padding, dropout=dropout_rate)]

        self.tcn_net = keras.Sequential(self.tcn_layers)
        self.cls_head = MLPClassification(
            hidden_dims[-1], out_dim=1, use_bias=True)
        self.projection_head = MLPProjector(hidden_dims[-1], emb_dim)

    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self.optimizer = optimizer

    @tf.function
    def loss_NAC(self, scores, labels):
        loss = tf.reduce_mean(tf.square(tf.reshape(
            scores, [-1]) - tf.reshape(labels, [-1])))
        return loss

    @tf.function
    def loss_UMC(self, dist_z, dist_zprime):
        variance = tf.square(dist_z-dist_zprime)
        tensor = 0.5 * tf.exp(tf.multiply(-1.0, variance)) * \
            (dist_z + dist_zprime) + 0.5 * variance
        loss = tf.reduce_mean(tensor)
        return loss

    def fit_manual(self, train_ds: tf.data.Dataset, epochs=100, verbose=0, clipping=False):
        """
        Fit COUTA model on train_ds

        Args:
            train_ds (tf.data.Dataset): Non-batched dataset of sliding windows
            batch_size (int, optional): Defaults to 32.
            epochs (int, optional): Defaults to 100.
            verbose (int, optional): Display training progress. Defaults to 0.
        """
        self.set_hypersphere_centroid(train_ds, eps=0.1)

        train_losses = keras.metrics.Mean(name='train_loss')
        val_losses = keras.metrics.Mean(name='val_loss')
        card = tf.data.experimental.cardinality(train_ds).numpy()
        if card < 0 or card == tf.data.experimental.UNKNOWN_CARDINALITY:
            card = 0
            for _ in train_ds:
                card += 1
        train_size = int(0.75 * card)
        train_split = train_ds.take(train_size)
        val_ds = train_ds.skip(train_size).batch(
            self.batch_size, drop_remainder=True)

        best_val_loss = float('inf')
        wait = 0
        ckpt = tf.train.Checkpoint(
            model=self,
            optimizer=self.optimizer
        )
        manager = tf.train.CheckpointManager(
            ckpt, './checkpoint_states/COUTA', max_to_keep=1)

        @tf.function
        def train_step(x_batch, neg_batch):
            with tf.GradientTape() as tape:
                ctx_normal = self.tcn_net(x_batch, training=True)
                ctx_abnormal = self.tcn_net(neg_batch, training=True)

                feature_norm = ctx_normal[:, -1, :]
                feature_abnormal = ctx_abnormal[:, -1, :]

                norm_score = self.cls_head(feature_norm, training=True)
                anom_score = self.cls_head(feature_abnormal, training=True)
                norm_labels = -tf.ones_like(norm_score, dtype=tf.float32)
                anom_labels = tf.ones_like(anom_score, dtype=tf.float32)
                scores = tf.concat([norm_score, anom_score], axis=0)
                labels = tf.concat([norm_labels, anom_labels], axis=0)
                loss_nac = self.loss_NAC(scores, labels)

                emb_norm, emb_prime = self.projection_head(
                    feature_norm, training=True)
                center = tf.reshape(self.center.read_value(), [1, -1])
                dist_z = tf.reduce_sum(tf.square(emb_norm - center), axis=-1)
                dist_zprime = tf.reduce_sum(
                    tf.square(emb_prime - center), axis=-1)
                loss_umc = self.loss_UMC(dist_z, dist_zprime)

                total_loss = loss_umc + self.alpha_cls * loss_nac

            gradient = tape.gradient(total_loss, self.trainable_variables)

            if clipping:
                gradient, _ = tf.clip_by_global_norm(gradient, self.clip_norm)
            self.optimizer.apply_gradients(
                zip(gradient, self.trainable_variables))
            train_losses.update_state(total_loss)
            return total_loss

        @tf.function
        def val_step(x_batch):
            ctx_normal = self.tcn_net(x_batch, training=False)
            feature_norm = ctx_normal[:, -1, :]

            emb_norm, emb_prime = self.projection_head(
                feature_norm, training=False)
            center = tf.reshape(self.center.read_value(), [1, -1])
            dist_z = tf.reduce_sum(tf.square(emb_norm - center), axis=-1)
            dist_zprime = tf.reduce_sum(tf.square(emb_prime - center), axis=-1)
            loss_umc = self.loss_UMC(dist_z, dist_zprime)

            val_losses.update_state(loss_umc)
            return loss_umc

        for epoch in range(epochs):
            train_losses.reset_states()
            val_losses.reset_states()
            train_ds = train_split.shuffle(10000).batch(
                self.batch_size, drop_remainder=True)
            neg_size = int(self.batch_size * 0.2)
            rng = np.random.RandomState(seed=42+epoch)
            if tf.data.experimental.cardinality(train_ds).numpy() < 0:
                size = FINITE_WINDOW_N
            else:
                size = tf.data.experimental.cardinality(train_ds).numpy()
            epoch_seed = rng.randint(0, 1e+6, size=size)
            # iterate over training batches
            for i, x_batch in enumerate(train_ds):
                try:
                    batch_np = x_batch.numpy()
                    rng = np.random.RandomState(seed=epoch_seed[i])
                    neg_idx = rng.randint(0, batch_np.shape[0], size=neg_size)
                    neg_windows = batch_np[neg_idx]
                    neg_batch, _ = AnomalyWindowSampler.sample_(
                        neg_windows, point_perturb=True, contextual_perturb=True, collective_perturb=True, seed=epoch_seed[i])
                except AttributeError:
                    continue
                # ensure shapes/dtypes are tensors
                neg_batch = tf.convert_to_tensor(
                    neg_batch, dtype=x_batch.dtype)
                x_batch = tf.cast(x_batch, dtype=tf.float32)
                _ = train_step(x_batch, neg_batch)

            # validation pass: iterate paired normal/neg batches from val_ds
            for i, x_batch in enumerate(val_ds):
                _ = val_step(x_batch)

            cur_val = val_losses.result().numpy()
            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} - train_loss: {train_losses.result().numpy():.6f} - val_loss: {cur_val:.6f}")

            # checkpointing + early stopping
            if best_val_loss - cur_val > 1e-4:
                best_val_loss = cur_val
                wait = 0
                manager.save()
            else:
                wait += 1
                if wait >= self.patience:
                    if verbose:
                        print("Early stopping.")
                    ckpt.restore(manager.latest_checkpoint)

                    break

    def set_hypersphere_centroid(self, train_ds, eps=0.1):
        ds = train_ds.shuffle(10000).batch(1024)
        representations = []
        for x_batch in ds:
            ctx = self.tcn_net(x_batch, training=False)
            ctx_feature = ctx[:, -1, :]
            reps, _ = self.projection_head(ctx_feature)
            representations.append(reps)
        stacked = tf.concat(representations, axis=0)
        center = tf.reduce_mean(stacked, axis=0)
        eps = tf.constant(eps, dtype=center.dtype)
        
        center = tf.where(tf.logical_and(tf.abs(center) < eps , center < 0), 
                          -eps, 
                          tf.where(tf.logical_and(tf.abs(center) < eps, center > 0), eps, center))
        self.center = tf.Variable(center, trainable=False)

    def score(self, data_ds):
        ds = data_ds.batch(1024)
        scores = []
        for x_batch in ds:
            ctx = self.tcn_net(x_batch, training=False)
            ctx_feature = ctx[:, -1, :]
            emb_z, emb_zprime = self.projection_head(ctx_feature)
            center = tf.reshape(self.center.read_value(), [1, -1])
            dist_z = tf.reduce_sum(tf.square(emb_z - center), axis=-1)
            dist_zprime = tf.reduce_sum(
                tf.square(emb_zprime - center), axis=-1)
            score = dist_z+dist_zprime
            scores.extend(score.numpy())
        score = np.array(scores)
        return score
