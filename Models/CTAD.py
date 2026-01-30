import numpy as np
import tensorflow as tf
from tensorflow import keras
import functions


class CTAD(keras.models.Model):
    def __init__(self, lstm_units=64, projection_hu=128, latent_dim=32, kernel_regularizer=None, dropout_rate=0.0,
                 batch_size=512, clip_val=1, patience=5, score='l2', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 512
        self.model = keras.Sequential([
            keras.layers.LSTM(lstm_units, return_sequences=False,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4)),
            keras.layers.Dense(projection_hu),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(projection_hu),
        ])
        self.patience = patience
        self.pos_aug1 = functions.AnomalyWindowSampler().augment_noise
        self.pos_aug2 = functions.AnomalyWindowSampler().augment_dropout_point
        self.neg_aug1 = functions.AnomalyWindowSampler().augment_additive
        self.neg_aug2 = functions.AnomalyWindowSampler().augment_amplify

        self.dist = score

        self.nv_pos_aug1 = 1
        self.nv_pos_aug2 = 1
        self.nv_neg_aug1 = 1
        self.nv_neg_aug2 = 1
        self.p_pos1 = 0.5
        self.p_pos2 = 0.5
        self.p_neg1 = 0.5
        self.p_neg2 = 0.5
        self.noise_sigma = 0.05
        self.bias = 0.2
        self.amp = 1.5

    def compile(self, optimizer=None, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self.optimizer = optimizer

    def call(self, x, training=False):
        return self.model(x, training=training)

    def create_augments(self, batch_windows, nv=10, p=0.5, random_state=123):
        if isinstance(batch_windows, tf.Tensor):
            batch_windows = batch_windows.numpy()
        N = batch_windows.shape[0]

        # produce two positive views (these should preserve "normal" semantics)
        pos1 = self.pos_aug1(batch_windows, nv=self.nv_pos_aug1, p=self.p_pos1,
                             sigma=self.noise_sigma, random_state=random_state)
        pos2 = self.pos_aug2(batch_windows, nv=self.nv_pos_aug2, p=self.p_pos2,
                             random_state=random_state + 1)

        # produce two negative (anomalous) views (these should make "bad" examples)
        neg1 = self.neg_aug1(batch_windows, nv=self.nv_neg_aug1, p=self.p_pos1,
                             bias=self.bias, random_state=random_state + 2)
        neg2 = self.neg_aug2(batch_windows, nv=self.nv_neg_aug2, p=self.p_pos2,
                             amp_val=self.amp, random_state=random_state + 3)

        full_batch = np.concatenate(
            [pos1, pos2, neg1, neg2], axis=0)   # shape (4N, W, F)
        return full_batch

    def set_aug_params(self, nv_pos1=1, nv_pos2=1, nv_neg1=1, nv_neg2=1, p_pos1=0.5, p_pos2=0.5, p_neg1=0.5, p_neg2=0.5,
                       noise_sigma=0.05, bias=0.2, amp=1.5):
        self.nv_pos_aug1 = nv_pos1
        self.nv_pos_aug2 = nv_pos2
        self.nv_neg_aug1 = nv_neg1
        self.nv_neg_aug2 = nv_neg2
        self.p_pos1 = p_pos1
        self.p_pos2 = p_pos2
        self.p_neg1 = p_neg1
        self.p_neg2 = p_neg2
        self.noise_sigma = noise_sigma
        self.bias = bias
        self.amp = amp

    @tf.function
    def cosine_sim(self, Z):
        Z = tf.cast(Z, tf.float32)
        Z = tf.math.l2_normalize(Z, axis=1, epsilon=1e-8)
        sim = tf.matmul(Z, Z, transpose_b=True)
        return sim

    @tf.function
    def compute_loss(self, Z, N, tau=0.1):
        M = tf.shape(Z)[0]
        sim = self.cosine_sim(Z)
        logits = sim / tau
        large_neg = -1e9
        logits_no_self = tf.linalg.set_diag(logits, tf.fill([M], large_neg))
        # log denominator for each row i: logsumexp over j != i (i^+ union i^-)
        log_denom = tf.reduce_logsumexp(logits_no_self, axis=1)  # shape (4N,)

        # compute log-probabilities (log-softmax excluding self)
        log_probs = logits_no_self - \
            tf.expand_dims(log_denom, 1)  # shape (4N, 4N)

        twoN = tf.cast(2 * N, tf.int32)   # if N is Python int this is fine
        idx = tf.range(M, dtype=tf.int32)
        row = tf.expand_dims(idx, 1)      # shape (M,1)
        col = tf.expand_dims(idx, 0)      # shape (1,M)

        # boolean mask of entries where both row < 2N and col < 2N
        both_in_first_block = tf.logical_and(
            tf.less(row, twoN), tf.less(col, twoN))  # (M,M)
        diag_eq = tf.equal(row, col)  # diagonal mask
        pos_mask_bool = tf.logical_and(
            both_in_first_block, tf.logical_not(diag_eq))
        mask_pos = tf.cast(pos_mask_bool, log_probs.dtype)  # float mask (M,M)

        # For rows > 2N-1 mask_pos rows are zero; we only average rows 0..2N-1
        # Sum log_probs over positive columns per anchor row
        sum_log_prob_pos = tf.reduce_sum(
            log_probs * mask_pos, axis=1)  # shape (4N,)
        # Count positives per anchor (should be 2N-1 for anchors, 0 for other rows)
        pos_count = tf.reduce_sum(mask_pos, axis=1)  # shape (4N,)

        # Avoid division by zero for non-anchor rows; we only keep rows 0..2N-1
        mean_log_prob_per_row = sum_log_prob_pos / \
            (pos_count + 1e-8)  # shape (4N,)

        # take the anchors rows: 0..2N-1
        anchor_mean_log_prob = mean_log_prob_per_row[:twoN]  # shape (2N,)

        # loss = - mean over anchors
        loss = - tf.reduce_mean(anchor_mean_log_prob)
        return loss

    def fit_manual(self, train_ds: tf.data.Dataset, epochs=100, verbose=0, clipping=False):
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
            ckpt, './checkpoint_states/CTAD', max_to_keep=1)

        @tf.function
        def train_step(full_batch, N, clipping=False):
            with tf.GradientTape() as tape:
                Z = self(full_batch, training=True)
                loss = self.compute_loss(Z, N, tau=0.1)
            gradient = tape.gradient(loss, self.trainable_variables)
            if clipping:
                gradient, _ = tf.clip_by_global_norm(gradient, self.clip_norm)
            self.optimizer.apply_gradients(
                zip(gradient, self.trainable_variables))
            train_losses.update_state(loss)
            return loss

        @tf.function
        def val_step(full_batch, N):
            Z = self(full_batch, training=False)
            loss = self.compute_loss(Z, N, tau=0.1)
            val_losses.update_state(loss)
            return loss

        for e in range(epochs):
            train_losses.reset_states()
            val_losses.reset_states()
            train_ds = train_split.shuffle(10000).batch(
                self.batch_size, drop_remainder=True)
            for idx, batch in enumerate(train_ds):
                batch_np = batch.numpy()
                N = batch_np.shape[0]   # original N
                full_batch_np = self.create_augments(batch_np)
                assert full_batch_np.shape[0] == 4 * N

                full_batch_tf = tf.convert_to_tensor(
                    full_batch_np, dtype=tf.float32)
                train_loss = train_step(full_batch_tf, N, clipping=clipping)

            for idx, batch in enumerate(val_ds):
                batch_np = batch.numpy()
                N = batch_np.shape[0]   # original N
                full_batch_np = self.create_augments(batch_np)
                assert full_batch_np.shape[0] == 4 * N

                full_batch_tf = tf.convert_to_tensor(
                    full_batch_np, dtype=tf.float32)
                val_loss = val_step(full_batch_tf, N)

            cur_val = val_losses.result().numpy()
            if verbose:
                print(
                    f"Epoch {e+1}/{epochs} - train_loss: {train_losses.result().numpy():.6f} - val_loss: {cur_val:.6f}")

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
        self.compute_center(train_ds)

    def compute_center(self, train_ds, eps=0.1):
        embs = []
        for batch in train_ds:
            embeddings = self(batch, training=False)
            embs.append(embeddings)
        stacked = tf.concat(embs, axis=0)
        center = tf.reduce_mean(stacked, axis=0)
        ''' center = tf.where(tf.logical_and(tf.abs(center) < eps , center < 0), 
                          -eps, 
                          tf.where(tf.logical_and(tf.abs(center) < eps, center > 0), eps, center)) '''
        self.center = tf.Variable(center, trainable=False)

    def score(self, data_ds):
        if self.dist == 'l2':
            dists = []
            ds = data_ds.batch(1024)
            centroid = self.center.read_value()
            for batch in ds:
                embeddings = self(batch, training=False)
                l2_dists = tf.reduce_sum(
                    tf.square(embeddings - centroid), axis=1)
                dists.extend(l2_dists.numpy())
            scores = np.array(dists)
            return scores
        if self.dist == 'cosine':
            dists = []
            ds = data_ds.batch(1024)
            centroid = tf.math.l2_normalize(
                self.center.read_value(), axis=0, epsilon=1e-12)
            for batch in ds:
                embeddings = tf.math.l2_normalize(
                    self(batch, training=False), axis=1, epsilon=1e-12)
                cosine_sim = tf.reduce_sum(embeddings * centroid, axis=1)
                cosine_diff = 1 - cosine_sim
                dists.extend(cosine_diff.numpy())
            scores = np.array(dists)
            return scores
