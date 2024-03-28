"""Dataset loaders for the three cohorts evaluated in the paper (Sec. 4.1):
HCP-A (task recognition + MoCA-score pretraining), ADNI and OASIS (disease
diagnosis).
"""

import os

import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.utils.data import Dataset

SC_MAT_KEY = 'aal116_sift_radius2_count_connectivity'


def _row_normalize(sc):
    row_sums = sc.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return sc / row_sums[:, np.newaxis]


def _total_normalize(sc):
    """Paper Sec. 4.1: divide fiber counts by the subject's total count.

    Unlike row normalization, this preserves symmetry of an undirected SC.
    """
    total = sc.sum()
    return sc if total == 0 else sc / total


def _load_sc_mat(path, key=SC_MAT_KEY, normalization='row'):
    sc = scipy.io.loadmat(path)[key]
    if normalization == 'row':
        sc = _row_normalize(sc)
    elif normalization == 'total':
        sc = _total_normalize(sc)
    elif normalization != 'none':
        raise ValueError(f'Unsupported SC normalization: {normalization!r}')
    return torch.from_numpy(sc).float()


def _load_sc_csv(path):
    sc = pd.read_csv(path, header=None)
    row_sums = sc.sum(axis=1).to_numpy()
    row_sums[row_sums == 0] = 1
    return torch.from_numpy(sc.to_numpy() / row_sums[:, np.newaxis]).float()


def _hcpa_task_label(filename):
    if 'FACENAME' in filename:
        return np.int64(0)
    if 'VISMOTOR' in filename:
        return np.int64(1)
    if 'CARIT' in filename:
        return np.int64(2)
    if 'REST' in filename:
        return np.int64(3)
    raise ValueError(f'Cannot infer an HCP-A task label from {filename!r}')


class FCSCDataset(Dataset):
    """HCP-A task-recognition FC-only dataset (4 classes: FACENAME/VISMOTOR/CARIT/REST).

    Used with the SPDNet baseline (no structural connectivity involved).
    """

    def __init__(self, data_dir, delimiter=',', slice=None):
        super(FCSCDataset, self).__init__()
        self.data_dir = data_dir
        self.delimiter = delimiter
        self.data_path = []
        self.labels = []
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]
            self.labels = self.labels[slice]

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if not filename.endswith('.csv'):
                continue
            self.data_path.append(os.path.join(self.data_dir, filename))
            self.labels.append(_hcpa_task_label(filename))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data = pd.read_csv(
            self.data_path[idx], delimiter=self.delimiter, header=0,
            usecols=lambda column: column != 'Unnamed: 0',
        ).fillna(0).values
        data = torch.from_numpy(data).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label


class FC_SCDataset(Dataset):
    """HCP-A task-recognition FC+SC dataset (4 classes: FACENAME/VISMOTOR/CARIT/REST).

    Used to train DeepHoloBrain, which needs the structural-connectivity
    graph to derive the harmonic wavelets. SC is looked up per-subject from
    a folder named after the first 14 characters of the FC filename.
    """

    def __init__(self, data_dir, sc_dir, delimiter=',', slice=None, sc_normalization='row'):
        super(FC_SCDataset, self).__init__()
        self.data_dir = data_dir
        self.sc_dir = sc_dir
        self.delimiter = delimiter
        self.sc_normalization = sc_normalization
        self.data_path = []
        self.labels = []
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]
            self.labels = self.labels[slice]

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if not filename.endswith('.csv'):
                continue
            self.data_path.append(os.path.join(self.data_dir, filename))
            self.labels.append(_hcpa_task_label(filename))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        path = self.data_path[idx]
        data = pd.read_csv(
            path, delimiter=self.delimiter, header=0,
            usecols=lambda column: column != 'Unnamed: 0',
        ).fillna(0).values
        data = torch.from_numpy(data).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        sc_folder = os.path.join(self.sc_dir, os.path.basename(path)[:14])
        sc_data = None
        if os.path.isdir(sc_folder):
            for sc_filename in os.listdir(sc_folder):
                if sc_filename.endswith('.mat'):
                    sc_data = _load_sc_mat(
                        os.path.join(sc_folder, sc_filename),
                        normalization=self.sc_normalization,
                    )
                    break
        if sc_data is None:
            raise FileNotFoundError(
                f"No structural-connectivity .mat file found under '{sc_folder}' for FC file '{path}'"
            )
        return data, label, sc_data


class ADataset(Dataset):
    """ADNI diagnosis dataset (CN vs AD, from `Label1` in the label CSV), FC + SC (.mat)."""

    def __init__(self, data_dir, label_path, sc_dir, delimiter=',', slice=None):
        super(ADataset, self).__init__()
        self.data_dir = data_dir
        self.sc_dir = sc_dir
        self.delimiter = delimiter
        self.data_path = []
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]

        self.labels = pd.read_csv(label_path, header=0)

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.txt'):
                self.data_path.append(os.path.join(self.data_dir, filename))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        path = self.data_path[idx]
        data = torch.from_numpy(
            pd.read_csv(path, delimiter=' ', header=None).fillna(0).values
        ).float()

        subject_id = os.path.basename(path)[4:12]
        matching_row = self.labels[self.labels['subject_id'].str[:8] == subject_id]
        label = matching_row.iloc[0]['Label1'] if not matching_row.empty else 0
        label = torch.tensor(label, dtype=torch.long)

        sc_prefix = os.path.basename(path)[:12]
        sc_data = None
        if os.path.isdir(self.sc_dir):
            for sc_filename in os.listdir(self.sc_dir):
                if sc_filename.endswith('.mat') and sc_filename[:12] == sc_prefix:
                    sc_data = _load_sc_mat(os.path.join(self.sc_dir, sc_filename))
                    break
        if sc_data is None:
            raise FileNotFoundError(
                f"No structural-connectivity .mat file with prefix '{sc_prefix}' found under '{self.sc_dir}'"
            )
        return data, label, sc_data


class OASISDataset(Dataset):
    """OASIS diagnosis dataset (preclinical-AD vs AD, from `Label1`), FC + SC (.csv)."""

    def __init__(self, data_dir, label_path, sc_dir, delimiter=',', slice=None):
        super(OASISDataset, self).__init__()
        self.data_dir = data_dir
        self.sc_dir = sc_dir
        self.delimiter = delimiter
        self.data_path = []
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]

        self.labels = pd.read_csv(label_path, header=0)

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.txt'):
                self.data_path.append(os.path.join(self.data_dir, filename))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        path = self.data_path[idx]
        data = torch.from_numpy(
            pd.read_csv(path, delimiter=' ', header=None).fillna(0).values
        ).float()

        subject_id = os.path.basename(path)[:8]
        matching_row = self.labels[self.labels['SUBJECT_ID'].str[:8] == subject_id]
        label = matching_row.iloc[0]['Label1'] if not matching_row.empty else 0
        label = torch.tensor(label, dtype=torch.long)

        sc_data = None
        if os.path.isdir(self.sc_dir):
            for sc_filename in os.listdir(self.sc_dir):
                if sc_filename.endswith('.csv') and sc_filename[:8] == subject_id:
                    sc_data = _load_sc_csv(os.path.join(self.sc_dir, sc_filename))
                    break
        if sc_data is None:
            raise FileNotFoundError(
                f"No structural-connectivity .csv file with prefix '{subject_id}' found under '{self.sc_dir}'"
            )
        return data, label, sc_data


class HCPADataset_pre(Dataset):
    """HCP-A MoCA-score regression dataset, used to pretrain DeepHoloBrain/SPDNet
    before fine-tuning on ADNI (Sec. 4.3, "Generality as a Pre-trained Model")."""

    def __init__(self, data_dir, label_path, sc_dir, delimiter=',', slice=None):
        super(HCPADataset_pre, self).__init__()
        self.data_dir = data_dir
        self.sc_dir = sc_dir
        self.delimiter = delimiter
        self.data_path = []
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]

        self.labels = pd.read_csv(label_path, header=0)

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.csv'):
                self.data_path.append(os.path.join(self.data_dir, filename))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        path = self.data_path[idx]
        data = torch.from_numpy(
            pd.read_csv(
                path, delimiter=self.delimiter, header=0,
                usecols=lambda column: column != 'Unnamed: 0',
            ).fillna(0).values
        ).float()

        subject_id = os.path.basename(path)[4:14]
        matching_row = self.labels[self.labels['src_subject_id'].str[:10] == subject_id]
        label = matching_row.iloc[0]['moca_total'] if not matching_row.empty else 28
        label = torch.tensor(label).float()

        sc_folder = os.path.join(self.sc_dir, os.path.basename(path)[:14])
        sc_data = None
        if os.path.isdir(sc_folder):
            for sc_filename in os.listdir(sc_folder):
                if sc_filename.endswith('.mat'):
                    sc_data = _load_sc_mat(os.path.join(sc_folder, sc_filename))
                    break
        if sc_data is None:
            raise FileNotFoundError(
                f"No structural-connectivity .mat file found under '{sc_folder}' for FC file '{path}'"
            )
        return data, label, sc_data
