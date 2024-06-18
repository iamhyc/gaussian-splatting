#!/usr/bin/env python3
import numpy as np

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
            if not self.instance:
                self.instance = self.cls(*self.args, **self.kwargs)
            return getattr(self.instance, name)

    def __del__(self):
        if self.instance:
            del self.instance
        pass
    
    pass

class PointCloudMark:
    def __init__(self, pcd_size: int, image_size: int):
        _length = int(np.ceil( image_size/8 ))
        self.marks = np.zeros((pcd_size,_length), dtype=np.uint8)
        pass

    @staticmethod
    def from_marks(marks: np.ndarray):
        pcd_size, image_size = marks.shape
        image_size *= 8
        marks_obj = PointCloudMark(pcd_size, image_size)
        marks_obj.marks  = marks
        return marks_obj

    def set(self, pt_map: np.ndarray, bit: int):
        index, offset = (bit//8), (bit%8)
        val = 1 << offset
        self.marks[ (pt_map,), index ] |= val
        pass

    def select(self, bit: int, reverse: bool=False):
        index, offset = (bit//8), (bit%8)
        val = 1 << offset
        if reverse:
            return (self.marks[:, index] & val == 0)
        else:
            return (self.marks[:, index] & val != 0)

    def select_multi(self, bits: list):
        return np.logical_or.reduce([self.select(bit) for bit in bits])

    def prune(self, prune_mask: np.ndarray):
        pruned = self.marks[prune_mask]
        self.marks = self.marks[~prune_mask]
        return pruned

    def concat(self, marks: np.ndarray):
        self.marks = np.concatenate([self.marks, marks], axis=0)

    pass
