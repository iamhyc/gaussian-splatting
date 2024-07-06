#!/usr/bin/env python3
import numpy as np
import torch

class LazyLoader:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.instance = None
        pass
    
    def __getattribute__(self, name: str):
        if name in ['cls', 'args', 'kwargs', 'instance']:
            return super().__getattribute__(name)
        else:
            if self.instance is None:
                self.instance = self.cls(*self.args, **self.kwargs)
            return getattr(self.instance, name)

    def __getitem__(self, key):
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return self.instance[key]

    def __del__(self):
        if self.instance is not None:
            del self.instance
        pass
    
    pass

class PointCloudMark:
    def __init__(self, pcd_size: int, image_size: int, device='cuda', _marks=None):
        _length = int(np.ceil( image_size/8 ))
        self.device = device
        if _marks is not None:
            self._marks = _marks
        else:
            self._marks = torch.zeros((pcd_size,_length), dtype=torch.uint8, device=device)
        pass

    @staticmethod
    def from_marks(marks: np.ndarray, device='cuda'):
        pcd_size, image_size = marks.shape
        image_size *= 8
        pcm = PointCloudMark(pcd_size, image_size, device=device,
                             _marks=torch.tensor(marks, dtype=torch.uint8, device=device))
        return pcm

    def __getitem__(self, key):
        return self._marks[key]

    def set(self, pt_map, bit: int):
        index, offset = (bit//8), (bit%8)
        val = 1 << offset
        ##
        pt_map = torch.tensor(pt_map, dtype=torch.long, device=self.device)
        self._marks[ pt_map, index ] |= val
        pass

    def select(self, bit: int, reverse: bool=False) -> torch.Tensor:
        index, offset = (bit//8), (bit%8)
        val = 1 << offset
        if reverse:
            return (self._marks[:, index] & val == 0)
        else:
            return (self._marks[:, index] & val != 0)

    def prune(self, prune_mask: torch.Tensor) -> torch.Tensor:
        pruned = self._marks[prune_mask]
        self._marks = self._marks[~prune_mask]
        return pruned

    def concat(self, marks: torch.Tensor):
        self._marks = torch.cat([self._marks, marks], dim=0)

    def save(self, file_path: str):
        with open(file_path, 'wb+') as fp:
            np.save(fp, self._marks.cpu().numpy())

    pass
