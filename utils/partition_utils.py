#!/usr/bin/env python3
import numpy as np

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

    def select(self, pt_map: np.ndarray, bit: int):
        index, offset = (bit//8), (bit%8)
        val = 1 << offset
        return np.where(self.marks[ (pt_map,), index ] & val != 0)
    pass
