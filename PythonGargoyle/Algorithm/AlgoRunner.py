import traceback
from torch import Tensor
import numpy as np
from typing import List
from Algorithm.CoreFuncs import unbit_pack_data_torch, LoadGainOffset_torch, AppGainOffset_torch, find_bad_pxl_torch, bad_pxl_corr_torch, remove_batch_bg, LightCyberAlg
from matplotlib import pyplot as plt
from colorama import Fore

class AlgoRunner:
    INSTANCE = None

    def __init__(self, H: int, W: int, offset_path: str, gain_path: str, corner_window_harris: int = 3, nms_window_harris: int = 3,
                 cutoff_freq: float = 0.4, filter_order: int = 32, minimum_dist_between_points_to_be_considered_different: int = 25,
                 max_corners: int = 6):
        print(Fore.CYAN + '--------------------------------')
        print('Preprocessing')
        self.Gain, self.Offset = LoadGainOffset_torch(offset_path, gain_path, W, H, device='cuda:0')                
        self.bad_pixels_neighborhoods, self.bad_pixels_flat = find_bad_pxl_torch(self.Gain, W, H, device='cuda:0')
        
        # Setting other members
        self.plt_flg = False
        self.use_conv_or_mean = False#True
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        self.minimum_dist_between_points_to_be_considered_different = minimum_dist_between_points_to_be_considered_different
        self.max_corners = max_corners
        self.corner_window_harris = corner_window_harris
        self.sigma_harris = corner_window_harris/2
        self.corners_thresh = 100
        self.final_threshold = 1
        self.use_corner_std_or_strengths_flg = True
        self.nms_window_harris = nms_window_harris
        
        print('All parameters')
        print('H = ' + str(H))
        print('W = ' + str(W))
        print('offset_path = ' + offset_path)
        print('gain_path = ' + gain_path)
        print('use_conv_or_mean = ' + str(self.use_conv_or_mean))
        print('cutoff_freq = ' + str(self.cutoff_freq))
        print('filter_order = ' + str(self.filter_order))
        print('minimum_dist_between_points_to_be_considered_different = ' + str(self.minimum_dist_between_points_to_be_considered_different))
        print('max_corners = ' + str(self.max_corners))
        print('sigma_harris = ' + str(self.sigma_harris))
        print('corners_thresh = ' + str(self.corners_thresh))
        print('final_threshold = ' + str(self.final_threshold))
        print('use_corner_std_or_strengths_flg = ' + str(self.use_corner_std_or_strengths_flg))
        print('nms_window_harris = ' + str(self.nms_window_harris))
        
        AlgoRunner.INSTANCE = self

        print('Finished preprocessing')
        print('--------------------------------' + Fore.RESET)
        
    def accelrated_vid(self, data: Tensor) -> List[np.ndarray]:
        all_corners = []
        Nfr, H, W = data.shape
        
        dat_nuc_tr = AppGainOffset_torch(data, self.Gain, self.Offset, device='cuda:0')
        corrected_frames = bad_pxl_corr_torch(dat_nuc_tr, self.bad_pixels_neighborhoods, self.bad_pixels_flat)
        frames = remove_batch_bg(corrected_frames, self.cutoff_freq, self.filter_order, self.use_conv_or_mean)
        
        if self.plt_flg:
            plt.figure(123)
            plt.imshow(data[2].cpu().squeeze().numpy())
            plt.title('corrected frames #3')
            plt.figure(1234)
            plt.imshow(data[1].cpu().squeeze().numpy())
            plt.title('corrected frames #2')
            plt.figure(2345)
            plt.plot(np.nansum(np.nansum(frames.cpu().squeeze().numpy(),axis=2),axis=1))
            plt.title('corrected frames average pixel level')
            plt.figure(867)
            plt.imshow(frames.mean(dim=0).cpu().squeeze().numpy())
            plt.title('corrected frames mean')
                
        all_corners = LightCyberAlg(frames, 
                                sigma_harris = self.sigma_harris, nms_window_harris  = self.nms_window_harris, 
                                max_corners = self.max_corners, corners_thresh = self.corners_thresh, 
                                minimum_dist_between_points_to_be_considered_different = self.minimum_dist_between_points_to_be_considered_different, 
                                final_threshold = self.final_threshold, use_corner_std_or_strengths_flg = self.use_corner_std_or_strengths_flg, plt_flg = self.plt_flg)
        
        # torch.cuda.synchronize()
        # e = time.time()
        # print(f"RUNTIME: {e-st}")
        return all_corners
    
    def accelrated_video_from_bitpacked_memview(self, frames_memview: memoryview, memview_length, H: int, W: int) -> List[List[List[int]]]:
        print('In accelrated_video_from_bitpacked_memview')
        try:
            print('Num bytes: %d' % memview_length)
            frames_bp = np.ndarray(buffer=frames_memview, shape=(memview_length,), dtype=np.uint8)
            print('ndarray created')
            frames = unbit_pack_data_torch(frames_bp, H, W)
            print('ndarray unpacked')
            # frames = torch.as_tensor(frames_cupy.astype(cupy.int32), device="cuda:0")
            ret = self.accelrated_vid(frames)
            print('finished video processing')
            ret = [ [list(np_arr.astype(np.int32)) for np_arr in frame_res] for frame_res in ret ]
            print('Result length: %d' % (len(ret)))
            print('Result: %s' % (ret))
            
            return ret
        except :
            print(traceback.format_exc())
            print("Error from within python process")
            raise


