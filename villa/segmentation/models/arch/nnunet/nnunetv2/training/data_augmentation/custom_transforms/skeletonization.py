from typing import Tuple

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, dilation

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class SkeletonTransform(BasicTransform):
    def __init__(self, do_tube: bool = True, ignore_label: int = None):
        """
        Calculates the skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        self.do_tube = do_tube
        self.ignore_label = ignore_label
    
    def apply(self, data_dict, **params):
        seg_all = data_dict['segmentation'].numpy()
        if self.ignore_label is not None:
            seg_all = np.where(seg_all == self.ignore_label, 0, seg_all)
        # Add tubed skeleton GT
        bin_seg = (seg_all > 0)
        seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)
        
        # Skeletonize
        if not np.sum(bin_seg[0]) == 0:
            skel = skeletonize(bin_seg[0])
            skel = (skel > 0).astype(np.int16)
            if self.do_tube:
                skel = dilation(dilation(skel))
            skel *= seg_all[0].astype(np.int16)
            seg_all_skel[0] = skel

        data_dict["skel"] = torch.from_numpy(seg_all_skel)
        return data_dict
        
class MedialSurfaceTransform(BasicTransform):
    def __init__(self, do_tube: bool = True, ignore_label: int = None):
        """
        Calculates the medial surface skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        self.do_tube = do_tube
        self.ignore_label = ignore_label
    
    def apply(self, data_dict, **params):
        seg_all = data_dict['segmentation'].numpy()
        if self.ignore_label is not None:
            seg_all = np.where(seg_all == self.ignore_label, 0, seg_all)
        # Add tubed skeleton GT
        bin_seg = (seg_all > 0)
        seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)
        
        # Skeletonize
        if not np.sum(bin_seg[0]) == 0:
            # skel = skeletonize(bin_seg[0], surface=True)
            skel = np.zeros_like(bin_seg[0])
            Z, Y, X = skel.shape
            
            # 逐層提取骨架 (2.5D)
            for z in range(Z):
                skel[z] |= skeletonize(bin_seg[0][z])

            if self.do_tube:
                # 1. 產生反轉遮罩：讓 EDT 計算到最近「骨架像素(0)」的距離
                skel_inv = (skel == 0)

                # 2. 執行歐幾里得距離轉換 (EDT)
                # dist_map 裡面的每個數值，代表該像素距離最近的骨架有多遠
                dist_map = distance_transform_edt(skel_inv)

                # 3. 透過半徑設定閾值，產生直徑為 3 的圓柱體
                # dist_map <= 1.5 會把距離骨架 0, 1.0, 1.414(對角線) 的像素都包進來
                skel = (dist_map <= 3).astype(np.int16)
            else:
                skel = (skel > 0).astype(np.int16)
            skel *= seg_all[0].astype(np.int16)
            seg_all_skel[0] = skel

        data_dict["skel"] = torch.from_numpy(seg_all_skel)
        return data_dict
