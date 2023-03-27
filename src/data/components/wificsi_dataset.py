from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset


class WiFiCSIDataset(Dataset):
    def __init__(self, data_dir: str = "data/"):
        self.filenames = list(Path(data_dir).glob("*.npz"))
        self.class_to_idx = {
            "BE": 0,
            "FA": 1,
            "PI": 2,
            "RU": 3,
            "SD": 4,
            "SU": 5,
            "WA": 6,
        }
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

    def get_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Convert sample idx to file name
        """
        name = self.filenames[idx]
        return np.load(name.as_posix())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        entry = self.get_sample(idx)
        return entry["data"][:12000].T, self.class_to_idx[entry["label"].item()]
