import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.preprocessing import LabelEncoder
import warnings
from datasets import load_dataset
from huggingface_hub import hf_hub_download


class Dataset_ETP(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETP', target='level', scale=True, timeenc=0, freq='d', norm=True):
        self.args = args
        self.flag = flag
        self.root_path = root_path
        self.max_seq_len = args.seq_len
        self.norm = norm
        self.__read_data__()

    def _conv_downsample_static_to_1d(self, static_channels):
        """
        Convert static channels from [C, T] to [C] with 1D convolution + global average.
        This keeps a convolutional downsampling narrative for static features.
        """
        arr = np.asarray(static_channels, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f'Expected static channels with shape [C, T], got {arr.shape}')

        kernel_size = int(getattr(self.args, 'uea_static_conv_kernel', 5))
        if kernel_size <= 0:
            kernel_size = 1
        kernel_size = min(kernel_size, arr.shape[1])

        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        filtered = np.stack([
            np.convolve(ch, kernel, mode='same') for ch in arr
        ], axis=0)
        return filtered.mean(axis=1).astype(np.float32)

    def _safe_split_indices(self, n_samples, labels):
        test_size = getattr(self.args, 'test_size', 0.3)
        if test_size <= 0 or test_size >= 1:
            test_size = 0.3

        shuffle_flag = bool(getattr(self.args, 'random_split', True))
        random_state = getattr(self.args, 'seed', 0)
        indices = np.arange(n_samples)

        try:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                shuffle=shuffle_flag,
                stratify=labels,
                random_state=random_state,
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                shuffle=shuffle_flag,
                stratify=None,
                random_state=random_state,
            )
        return train_idx, test_idx

    def _pad_or_truncate_sequence(self, seq):
        seq = np.asarray(seq, dtype=np.float32)
        if len(seq) >= self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        else:
            pad = np.zeros(self.max_seq_len - len(seq), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        return seq

    def _build_uea_time_marks(self, seq_len):
        # UEA .ts does not carry calendar timestamps in this setting.
        time_marks = np.zeros((seq_len, 4), dtype=np.float32)
        time_marks[:, 3] = np.arange(seq_len, dtype=np.float32)
        return time_marks

    def _read_data_uea(self):
        ts_file = getattr(self.args, 'uea_file', getattr(self.args, 'series_file', 'sample_data.ts'))
        ts_path = os.path.join(self.root_path, ts_file)
        if not os.path.exists(ts_path):
            raise FileNotFoundError(f'UEA ts file not found: {ts_path}')

        X_df, y_raw = load_from_tsfile_to_dataframe(ts_path)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(y_raw).astype(np.int64)

        series_data = []
        static_data = []
        time_marks_data = []

        for i in range(len(X_df)):
            first_col = X_df.columns[0]
            seq = np.asarray(X_df.iloc[i][first_col], dtype=np.float32)
            seq = self._pad_or_truncate_sequence(seq)

            static_cols = [c for c in X_df.columns if c != first_col]
            static_channels = []
            for c in static_cols:
                v = np.asarray(X_df.iloc[i][c], dtype=np.float32)
                v = self._pad_or_truncate_sequence(v)
                static_channels.append(v)

            if len(static_channels) == 0:
                static_vec = np.zeros((1,), dtype=np.float32)
            else:
                static_mat = np.stack(static_channels, axis=0)
                static_vec = self._conv_downsample_static_to_1d(static_mat)

            series_data.append(seq)
            static_data.append(static_vec)
            time_marks_data.append(self._build_uea_time_marks(self.max_seq_len))

        self.series_data = np.stack(series_data).astype(np.float32)
        self.static_data = np.stack(static_data).astype(np.float32)
        self.time_marks_data = np.stack(time_marks_data).astype(np.float32)
        self.labels = labels

        n = len(self.series_data)
        train_idx, test_idx = self._safe_split_indices(n, self.labels)

        if self.norm:
            series_scaler = StandardScaler()
            self.series_data[train_idx] = series_scaler.fit_transform(
                self.series_data[train_idx].reshape(-1, 1)
            ).reshape(self.series_data[train_idx].shape)
            self.series_data[test_idx] = series_scaler.transform(
                self.series_data[test_idx].reshape(-1, 1)
            ).reshape(self.series_data[test_idx].shape)

            static_scaler = StandardScaler()
            self.static_data[train_idx] = static_scaler.fit_transform(self.static_data[train_idx])
            self.static_data[test_idx] = static_scaler.transform(self.static_data[test_idx])

        flag_low = str(self.flag).lower() if self.flag is not None else 'train'
        sel_idx = train_idx if flag_low.startswith('train') else test_idx
        sel_idx = np.sort(sel_idx)

        self.series_data = self.series_data[sel_idx]
        self.time_marks_data = self.time_marks_data[sel_idx]
        self.static_data = self.static_data[sel_idx]
        self.labels = self.labels[sel_idx]

    def _read_data_etp(self):
        static_file = getattr(self.args, 'static_file', 'ETP_static_no.csv')
        series_file = getattr(self.args, 'series_file', 'ETP_series.csv')

        static_df = pd.read_csv(os.path.join(self.root_path, static_file))
        series_df = pd.read_csv(os.path.join(self.root_path, series_file))

        self.static_cols = [c for c in static_df.columns if c not in ['UID', 'level']]
        self.static_data = static_df[self.static_cols].values.astype(np.float32)
        self.labels = static_df['level'].values.astype(np.int64)

        series_grouped = series_df.groupby('UID')
        self.series_data = []
        self.time_marks_data = []

        for uid, g in series_grouped:
            g = g.sort_values('DATA_DATE')
            seq = g['sum_power'].values.astype(np.float32)

            time_marks = []
            for date_str in g['DATA_DATE']:
                date_obj = pd.to_datetime(date_str)
                month = date_obj.month
                day = date_obj.day
                weekday = date_obj.weekday()
                hour = 12
                time_marks.append([month, day, weekday, hour])

            time_marks = np.array(time_marks)

            # Keep a fixed sequence length.
            if len(seq) >= self.max_seq_len:
                seq = seq[-self.max_seq_len:]
                time_marks = time_marks[-self.max_seq_len:]
            else:
                pad = np.zeros(self.max_seq_len - len(seq))
                seq = np.concatenate([pad, seq])

                # Pad time marks with the first mark or a fallback value.
                pad_time = np.tile(time_marks[0] if len(time_marks) > 0 else [1, 1, 0, 0],
                                (self.max_seq_len - len(time_marks), 1))
                time_marks = np.concatenate([pad_time, time_marks], axis=0)

            self.series_data.append(seq)
            self.time_marks_data.append(time_marks)

        self.series_data = np.stack(self.series_data)
        self.time_marks_data = np.stack(self.time_marks_data)

        # Split before normalization to avoid leakage.
        n = len(self.series_data)
        train_idx, test_idx = self._safe_split_indices(n, self.labels)

        train_series = self.series_data[train_idx]
        test_series = self.series_data[test_idx]

        # Fit on train split, transform both splits.
        if self.norm:
            series_scaler = StandardScaler()
            train_series_norm = series_scaler.fit_transform(
                train_series.reshape(-1, 1)
            ).reshape(train_series.shape)
            test_series_norm = series_scaler.transform(
                test_series.reshape(-1, 1)
            ).reshape(test_series.shape)
            self.series_data[train_idx] = train_series_norm
            self.series_data[test_idx] = test_series_norm

            static_scaler = StandardScaler()
            self.static_data[train_idx] = static_scaler.fit_transform(
                self.static_data[train_idx]
            )
            self.static_data[test_idx] = static_scaler.transform(
                self.static_data[test_idx]
            )

        flag_low = str(self.flag).lower() if self.flag is not None else 'train'
        sel_idx = train_idx if flag_low.startswith('train') else test_idx
        sel_idx = np.sort(sel_idx)

        # slice arrays so this Dataset instance only holds the requested split
        self.series_data = self.series_data[sel_idx]
        self.time_marks_data = self.time_marks_data[sel_idx]
        if hasattr(self, 'static_data'):
            self.static_data = self.static_data[sel_idx]
        if hasattr(self, 'labels'):
            self.labels = self.labels[sel_idx]

    def __read_data__(self):
        data_name = str(getattr(self.args, 'data', 'ETP')).upper()
        if data_name == 'UEA':
            self._read_data_uea()
        else:
            self._read_data_etp()

    def __len__(self):
        return len(self.series_data)

    def __getitem__(self, idx):
        series = self.series_data[idx]             
        static = self.static_data[idx]             
        label = self.labels[idx]
        time_marks = self.time_marks_data[idx]

        # Convert to [T, 1].
        series = series.reshape(-1, 1)

        return {
            'series': torch.tensor(series, dtype=torch.float),
            'static': torch.tensor(static, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long),
            'time_marks': torch.tensor(time_marks, dtype=torch.float)
        }