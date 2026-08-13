import os
import re
from typing import List, Callable, Optional
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode



class ConvertHUToAtt:
    def __init__(self, min_hu: float = -1000,
                       max_hu: float = 1500,
                       mu_water: float = 0.0193):
        
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.mu_water = mu_water

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clip(x, self.min_hu , self.max_hu)    
        mu = self.mu_water * (x / 1000 + 1)
        return mu

class CTNormalizer:
    def __init__(self, min_hu: float = -1000,
                       max_hu: float = 1500,
                       mu_water: float = 0.0193):
        
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.mu_water = mu_water

        self.mu_min = mu_water * (min_hu / 1000 + 1)
        self.mu_max = mu_water * (max_hu / 1000 + 1)

    def normalize(self, x: torch.Tensor, clip: bool = False) -> torch.Tensor:
        if clip:
            x = torch.clip(x, self.mu_min, self.mu_max)
        x = (x - self.mu_min) / (self.mu_max - self.mu_min)
        return x
    
    def unnormalize(self, x: torch.Tensor, clip: bool = False) -> torch.Tensor:
        if clip:
            x = torch.clip(x, 0, 1)
        x = (self.mu_max - self.mu_min) * x + self.mu_min
        return x
 

class ConvertToZeroOne:
    def __init__(self, min_hu: float = -1000, max_hu: float = 1500):
        self.min_hu = min_hu
        self.max_hu = max_hu

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clip(x, self.min_hu, self.max_hu)
        x = (x - self.min_hu) / (self.max_hu - self.min_hu)
        return x

class CTDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 patient_list: List[int],
                 transform: Callable = None):

        self.root_dir = root_dir
        self.patient_list = patient_list
        self.transform = transform
        self.file_paths = self._gather_files(root_dir, patient_list)

    def _gather_files(self, root_dir, patient_list):
        file_paths = []
        dirs = [os.path.join(root_dir, str(name)) for name in patient_list]

        for dir in dirs:
            for root, _, files in os.walk(dir):
                for file in files:
                    if file.endswith(".npy"):
                        file_paths.append(os.path.join(dir, file))
        return sorted(file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        vol = self._load_file(file_path)
        if self.transform:
            vol = self.transform(vol)
        return vol

    def _load_file(self, file_path: str) -> torch.Tensor:
        data = np.load(file_path, mmap_mode='r').astype(np.float32)
        data = torch.from_numpy(data).unsqueeze(0)
        return data


class CTDatasetML(Dataset):
    def __init__(
        self,
        root_dir: str,
        patient_list: List[int],
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.patient_list = patient_list
        self.transform = transform

        # One sample = one patient folder
        self.samples = self._gather_patients(root_dir, patient_list)

    def _gather_patients(self, root_dir, patient_list):
        samples = []

        for patient_id in patient_list:
            patient_dir = os.path.join(root_dir, str(patient_id))
            if os.path.isdir(patient_dir):
                samples.append((patient_dir, patient_id))

        return sorted(samples, key=lambda x: x[1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_dir, patient_id = self.samples[idx]
        vol = self._load_volume(patient_dir)

        if self.transform:
            vol = self.transform(vol)

        return vol, patient_id

    def _load_volume(self, patient_dir: str) -> torch.Tensor:
        slice_files = [
            f for f in os.listdir(patient_dir)
            if f.endswith(".npy")
        ]

        if not slice_files:
            raise RuntimeError(f"No .npy slices found in {patient_dir}")

        slice_files = sorted(slice_files, key=self._extract_slice_index)

        slices = []
        for fname in slice_files:
            fpath = os.path.join(patient_dir, fname)
            arr = np.load(fpath).astype(np.float32)   # shape: (512, 512)

            if arr.ndim != 2:
                raise ValueError(
                    f"Expected slice shape (H, W), got {arr.shape} in {fpath}"
                )

            slices.append(arr)

        # (D, H, W)
        volume = np.stack(slices, axis=0)

        # (1, D, H, W)
        volume = torch.from_numpy(volume).unsqueeze(0)

        return volume

    @staticmethod
    def _extract_slice_index(filename: str) -> int:
        match = re.search(r"slice_(\d+)\.npy$", filename)
        if match is None:
            raise ValueError(f"Filename does not match expected pattern: {filename}")
        return int(match.group(1))


train_transforms = transforms.Compose([
    ConvertToZeroOne(),
    # ConvertHUToAtt(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomAffine(
        degrees=20,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        interpolation=InterpolationMode.BILINEAR,
    ),
    transforms.ColorJitter(brightness=0.01, contrast=0.01)
])


test_transforms = transforms.Compose([
    ConvertToZeroOne()
])


train_hu_transforms = transforms.Compose([
    ConvertHUToAtt(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomAffine(
        degrees=20,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        interpolation=InterpolationMode.BILINEAR,
    ),
    transforms.ColorJitter(brightness=0.01, contrast=0.01),
])


test_hu_transforms = transforms.Compose([
    ConvertHUToAtt()
])


def get_ct_dataloaders(root_dir: str,
                       patient_list: List[int],
                       batch_size: int = 1,
                       shuffle: bool = True,
                       train_mode: bool = False):

    transform = train_transforms if train_mode else test_transforms
    dataset = CTDataset(root_dir, patient_list, transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=True)
    return dataloader


def get_att_ct_dataloaders(root_dir: str,
                            patient_list: List[int],
                            batch_size: int = 1,
                            shuffle: bool = True,
                            train_mode: bool = False):

    transform = train_hu_transforms if train_mode else test_hu_transforms
    dataset = CTDataset(root_dir, patient_list, transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=True)
    return dataloader


def get_ct_dataloaders_create_multilevel(root_dir: str,
                                         patient_list: List[int],
                                         batch_size: int = 1,
                                         shuffle: bool = True,
                                         train_mode: bool = False):

    dataset = CTDatasetML(root_dir, patient_list, transform=None)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=True)
    return dataloader