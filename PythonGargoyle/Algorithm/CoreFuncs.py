import time
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math

def load_data(fp: str, H: int, W: int) -> np.ndarray:
    with open(fp, 'rb') as fid:
        header = 8
        packed = 1.5
        ext_bytes = 64
        np.fromfile(fid, dtype=np.uint8, count=header)
        np.fromfile(fid, dtype=np.uint8, count=int(W * H * packed))
        np.fromfile(fid, dtype=np.uint8, count=max(0, ext_bytes - header))
        dat_all = np.fromfile(fid, dtype=np.uint8)
    return dat_all

def unbit_pack_data_torch(data_np: np.ndarray, H: int, W: int) -> Tensor:
    packed = 1.5
    
    # Convert to PyTorch tensor
    dat_all_tr = torch.tensor(data_np, dtype=torch.uint8, device='cuda:0')
    fr_len = int(W * H * packed + 64)
    n_fr = len(dat_all_tr) // fr_len
    
    qab1, qab2, qab3 = dat_all_tr[:fr_len*n_fr].view(n_fr,fr_len)[:,8:-56].view(n_fr,-1,3).permute(2,1,0).view(3,H,W//2,n_fr).permute(0,2,1,3).type(torch.int16).unbind(dim=0)
    
    # Process qab2 and combine qab values
    qab2_least4 = qab2 % 16
    qab2_most4 = qab2 // 16
    odd_col_pix = qab1 + 256 * qab2_least4
    even_col_pix = qab3 * 16 + qab2_most4
    
    comb = torch.zeros((2, W // 2, H, n_fr), dtype=torch.int16, device='cuda:0')
    comb[0, :, :, :] = odd_col_pix
    comb[1, :, :, :] = even_col_pix
    
    # Reshape to match the final expected shape
    comb = comb.permute(*reversed(range(len(comb.shape)))).reshape(*reversed((W , H, n_fr))).permute(*reversed(range(len((W , H, n_fr))))).permute(2, 1, 0)
    return comb

def LoadGainOffset_torch(offset_path, gain_path, W, H, device):
    # Read Gain and Offset using NumPy, then convert to PyTorch tensors
    if offset_path.lower().endswith(".raw"):
        offset_np = np.fromfile(offset_path, dtype=np.uint16).reshape(W, H).T.astype(np.int16)
        gain_np = np.fromfile(gain_path, dtype=np.float64).reshape(W, H).T.astype(np.float32)
        gain_np[0,:8]=0
        Offset = torch.from_numpy(offset_np).to(device)
        Gain = torch.from_numpy(gain_np).to(device)    
    elif offset_path.lower().endswith(".txt"):
        offset_np = np.loadtxt(offset_path, dtype=np.uint16).reshape(H, W).astype(np.int16)
        gain_np = np.loadtxt(gain_path, dtype=np.float64).reshape(H, W).astype(np.float32)
        gain_np[0,:8]=0
        Offset = torch.from_numpy(offset_np).to(device)
        Gain = torch.from_numpy(gain_np).to(device)    
    # Gain[0,:8]=0
    # Gain = Gain*0+1
    # Offset = Offset*0
    return Gain, Offset

def AppGainOffset_torch(data_raw, Gain, Offset, device):
    
    # Ensure dimensions match
    if data_raw.shape[1:] != Offset.shape:
        raise ValueError(f"Dimension mismatch: data_raw shape {data_raw.shape}, Offset shape {Offset.shape}")
    
    return data_raw * Gain - Offset

def find_bad_pxl_torch(Gain_torch, W, H, device):
    # Assuming Gain is a 2D tensor with shape [H, W]
    H, W = Gain_torch.shape
    sentinel_value = W * H  # Define the sentinel value
    
    # Find indices of bad pixels
    y_bad, x_bad = torch.where(Gain_torch == 0)
        
    # Offsets for a 3x3 neighborhood
    offsets_y = torch.tensor([-1, -1, -1, 0, 0, 1, 1, 1], device=device)
    offsets_x = torch.tensor([-1, 0, 1, -1, 1, -1, 0, 1], device=device)

    # Add offsets to bad pixel coordinates
    neighborhood_y = (y_bad.unsqueeze(1) + offsets_y)
    neighborhood_x = (x_bad.unsqueeze(1) + offsets_x)

    # Replace out-of-bounds indices with values that map to the sentinel value when flattened
    out_of_bounds_mask = ((neighborhood_y < 0) | (neighborhood_y >= H)) | ((neighborhood_x < 0) | (neighborhood_x >= W))
    neighborhood_y[out_of_bounds_mask] = 0
    neighborhood_x[out_of_bounds_mask] = sentinel_value
    
    # Flatten the coordinates to get indices in the flattened array
    bad_pixels_neighborhoods = neighborhood_y * W + neighborhood_x

    # Flatten the coordinates of bad pixels to get their indices in the flattened Gain tensor
    bad_pixels_flat = y_bad * W + x_bad

    # Create a mask where each bad pixel's neighborhood is compared against all bad pixels
    bad_pixel_mask = bad_pixels_neighborhoods.unsqueeze(2) == bad_pixels_flat.unsqueeze(0).unsqueeze(1)
    
    # Any neighborhood that contains a bad pixel is replaced with sentinel_value
    bad_pixels_neighborhoods[bad_pixel_mask.any(dim=2)] = sentinel_value

    return bad_pixels_neighborhoods, bad_pixels_flat

def bad_pxl_corr_torch(data_raw_torch, bad_pixels_neighborhoods, bad_pixels_flat):
    batch_size, H, W = data_raw_torch.shape
    sentinel_value = W * H

    # Flatten data_raw_torch to 2D
    data_raw_flat = data_raw_torch.reshape(batch_size, -1)
    
    ind_ = (bad_pixels_neighborhoods == sentinel_value)
    bad_pixels_neighborhoods[ind_] = 0
    
    # Expand bad_pixels_neighborhoods and bad_pixels_flat to match batch size
    neighborhoods_expanded = bad_pixels_neighborhoods.unsqueeze(0).expand(batch_size, -1, -1)
    bad_pixels_flat_expanded = bad_pixels_flat.unsqueeze(0).expand(batch_size, -1)

    # Gather the values from the bad_pixels_neighborhoods
    neighborhood_values = torch.gather(data_raw_flat, 1, neighborhoods_expanded.view(batch_size, -1))
    neighborhood_values = neighborhood_values.view(batch_size, -1, 8)

    # Replace the sentinel values with NaN
    neighborhood_values[ind_.unsqueeze(0).expand(batch_size, -1, -1)] = float('nan')
        
    # Compute the mean of the bad_pixels_neighborhoods, ignoring NaNs
    neighborhood_means = torch.nanmean(neighborhood_values, dim=2)

    # Replace bad pixel values with the computed means in the original data
    data_raw_flat.scatter_(1, bad_pixels_flat_expanded, neighborhood_means)

    # Reshape data_raw_torch back to original shape
    data_raw_torch = data_raw_flat.view(batch_size, H, W)

    return data_raw_torch

def _get_fs(fs, nyq):
    """
    Utility for replacing the argument 'nyq' (with default 1) with 'fs'.
    """
    if nyq is None and fs is None:
        fs = 2
    elif nyq is not None:
        if fs is not None:
            raise ValueError("Values cannot be given for both 'nyq' and 'fs'.")
        fs = 2*nyq
    return fs

def grant(M: int):
    alpha = 0.54
    a = [alpha, 1. - alpha]
    fac = np.linspace(-np.pi, np.pi, M)
    w = np.zeros(M)
    for k in range(len(a)):
        w += a[k] * np.cos(k * fac)
    return w

def firwin2(numtaps, freq, gain, nfreqs=None, nyq=None,
            antisymmetric=False, fs=None):

    nyq = 0.5 * _get_fs(fs, nyq)

    if len(freq) != len(gain):
        raise ValueError('freq and gain must be of same length.')

    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError(('ntaps must be less than nfreqs, but firwin2 was '
                          'called with ntaps=%d and nfreqs=%s') %
                         (numtaps, nfreqs))

    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError('freq must start with 0 and end with fs/2.')
    d = np.diff(freq)
    if (d < 0).any():
        raise ValueError('The values in freq must be nondecreasing.')
    d2 = d[:-1] + d[1:]
    if (d2 == 0).any():
        raise ValueError('A value in freq must not occur more than twice.')
    if freq[1] == 0:
        raise ValueError('Value 0 must not be repeated in freq')
    if freq[-2] == nyq:
        raise ValueError('Value fs/2 must not be repeated in freq')

    if antisymmetric:
        if numtaps % 2 == 0:
            ftype = 4
        else:
            ftype = 3
    else:
        if numtaps % 2 == 0:
            ftype = 2
        else:
            ftype = 1

    if ftype == 2 and gain[-1] != 0.0:
        raise ValueError("A Type II filter must have zero gain at the "
                         "Nyquist frequency.")
    elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
        raise ValueError("A Type III filter must have zero gain at zero "
                         "and Nyquist frequencies.")
    elif ftype == 4 and gain[0] != 0.0:
        raise ValueError("A Type IV filter must have zero gain at zero "
                         "frequency.")

    if nfreqs is None:
        nfreqs = 1 + 2 ** int(math.ceil(math.log(numtaps, 2)))

    if (d == 0).any():
        # Tweak any repeated values in freq so that interp works.
        freq = np.array(freq, copy=True)
        eps = np.finfo(float).eps * nyq
        for k in range(len(freq) - 1):
            if freq[k] == freq[k + 1]:
                freq[k] = freq[k] - eps
                freq[k + 1] = freq[k + 1] + eps
        # Check if freq is strictly increasing after tweak
        d = np.diff(freq)
        if (d <= 0).any():
            raise ValueError("freq cannot contain numbers that are too close "
                             "(within eps * (fs/2): "
                             "{}) to a repeated value".format(eps))

    # Linearly interpolate the desired response on a uniform mesh `x`.
    x = np.linspace(0.0, nyq, nfreqs)
    fx = np.interp(x, freq, gain)

    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    shift = np.exp(-(numtaps - 1) / 2. * 1.j * np.pi * x / nyq)
    if ftype > 2:
        shift *= 1j

    fx2 = fx * shift

    # Use irfft to compute the inverse FFT.
    out_full = np.fft.irfft(fx2)

    wind = grant(numtaps)

    # Keep only the first `numtaps` coefficients in `out`, and multiply by
    # the window.
    out = out_full[:numtaps] * wind

    if ftype == 3:
        out[out.size // 2] = 0.0

    return out

class LogPoint:
    def __init__(self, name: str):
        torch.cuda.synchronize()
        self.name = name
        self.allocated_memory_start = torch.cuda.memory_allocated()//10**9
        self.st_time = time.time()

    def end(self):
        torch.cuda.synchronize()
        self.e_time = time.time()
        self.allocated_memory_end = torch.cuda.memory_allocated()//10**9

    def end_and_print(self):
        self.end()
        print(f"{self.name}: runtime: {self.runtime} , memory usage: {self.allocated_memory_end-self.allocated_memory_start}")

    @property
    def runtime(self):
        try:
            return self.e_time-self.st_time
        except AttributeError:
            raise RuntimeError("Log Point has not ended yet")

def LightCyberAlg(input_tensor, sigma_harris = 1.5, nms_window_harris = 5, max_corners=10, 
                        corners_thresh = 300, minimum_dist_between_points_to_be_considered_different=5, final_threshold = .5, 
                        use_corner_std_or_strengths_flg = True, plt_flg = False, use_G1=True, device = 'cuda:0'):
    """
    Light-Cyber-Alg: Harris corner detector with standard deviation window for a batch of grayscale images.
    
    Parameters
    ----------
    input_tensor : torch.Tensor
        Batch of grayscale image tensors with shape (batch_size, height, width).
    
    sigma_harris : float
        Standard deviation of the smoothing Gaussian filter.
    
    nms_window_harris : int
        Size of the window for standard deviation calculation around each corner.
    
    max_corners : int, optional
        Maximum number of corners to return for each frame.
    
    corners_thresh : float, optional
        The low bound sensitivity for corner detection.
    
    minimum_dist_between_points_to_be_considered_different : int, optional
        Radius of region considered in non-maximal suppression.
    
    final_threshold : float, optional
        Threshold for selecting strong corners.
    
    use_G1 : bool, optional
        Flag to use separable Gaussian filters (True) or a combined 2D filter (False).
    
    without_loops_flg : bool, optional
        Flag to use vectorized implementation without loops (True) or with loops (False).

    Returns
    -------
    result_list_xy_loc_val : numpy
        A list of the xy loc of the corners.
    """
    
    # Derivative masks for image gradients
    dx = torch.tensor([[-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    dy = dx.transpose(2, 3)

    # Convolution for gradients in x and y directions
    Ix = F.conv2d(input_tensor.unsqueeze(1), dx, padding='same')
    Iy = F.conv2d(input_tensor.unsqueeze(1), dy, padding='same')

    # Gaussian filter preparation
    f_wid = round(3 * sigma_harris)
    gauss_np = torch.exp(-torch.arange(-f_wid, f_wid + 1)**2/2/sigma_harris**2)/(2*torch.pi*sigma_harris**2)**.5
    G1_1D = gauss_np.clone().detach().to(dtype=torch.float32, device=device)
    G1 = G1_1D.reshape(1, 1, -1, 1)
    G1_transpose = G1_1D.reshape(1, 1, 1, -1)
    G2 = torch.mul(G1, G1_transpose)

    # Apply Gaussian filter (separable or combined)
    if use_G1:
        Ix2 = F.conv2d(F.conv2d(Ix.pow(2), G1, padding='same'), G1.transpose(2, 3), padding='same').squeeze(1)
        Iy2 = F.conv2d(F.conv2d(Iy.pow(2), G1, padding='same'), G1.transpose(2, 3), padding='same').squeeze(1)
        Ixy = F.conv2d(F.conv2d(Ix * Iy, G1, padding='same'), G1.transpose(2, 3), padding='same').squeeze(1)
    else:
        Ix2 = F.conv2d(Ix.pow(2), G2, padding='same').squeeze(1)
        Iy2 = F.conv2d(Iy.pow(2), G2, padding='same').squeeze(1)
        Ixy = F.conv2d(Ix * Iy, G2, padding='same').squeeze(1)

    # Compute Harris response
    harris = (Ix2 * Iy2 - Ixy.pow(2)) / (Ix2 + Iy2 + 1e-12)

    # Non-maximal suppression and corner strength thresholding
    max_pool = F.max_pool2d(harris.unsqueeze(1), kernel_size=int(2 * minimum_dist_between_points_to_be_considered_different + 1), stride=1, padding=minimum_dist_between_points_to_be_considered_different)
    corners = (harris * ((harris == max_pool.squeeze(1)) & (harris > corners_thresh))).nan_to_num()
    
    # Parameters for batch and image dimensions
    batch_size, height, width = input_tensor.shape

    # Find top max_corners corners in each frame
    corners_vals, idx = torch.topk(corners.view(corners.size(0), -1), k=max_corners)
    corners_y, corners_x = idx.div(corners.size(2), rounding_mode='floor'), idx % corners.size(2)

    # Vectorized computation of standard deviation in corner windows
    disp_x, disp_y = torch.meshgrid(torch.arange(-nms_window_harris // 2, nms_window_harris // 2 + 1).to(device), 
                                    torch.arange(-nms_window_harris // 2, nms_window_harris // 2 + 1).to(device), indexing='ij')
    disp_x = disp_x.reshape(1, 1, -1, 1, 1)
    disp_y = disp_y.reshape(1, 1, -1, 1, 1)
    
    window_x = (corners_x.view(batch_size, max_corners, 1, 1, 1) + disp_x).clamp(min=0, max=width - 1)
    window_y = (corners_y.view(batch_size, max_corners, 1, 1, 1) + disp_y).clamp(min=0, max=height - 1)
    
    std_vals = input_tensor[torch.arange(batch_size).view(batch_size, 1, 1, 1, 1).expand_as(window_x), window_y, window_x].std(dim=[2, 3, 4])
    corner_std_vals_vid = torch.zeros_like(corners)  # Initialize corner std values
    corner_std_vals_vid[
        torch.arange(batch_size).view(-1, 1, 1),
        corners_y[:, :max_corners].unsqueeze(2),
        corners_x[:, :max_corners].unsqueeze(2)
    ] = std_vals[:, :max_corners].unsqueeze(2)
    
    corner_strengths = corners.mean(dim=0).nan_to_num()
    corner_std_vals = corner_std_vals_vid.mean(dim=0).nan_to_num()
        
    # Apply threshold to corner_std_vals and find coordinates
    if use_corner_std_or_strengths_flg:
        out_vals = corner_std_vals
    else:
        out_vals = corner_strengths
    
    smooth_flg = 2
    if smooth_flg==1:
        # Gaussian filter preparation - for collecting all the energy in the speackle region
        sigma_smooth = minimum_dist_between_points_to_be_considered_different/8
        f_wid = np.ceil(3 * sigma_smooth)
        gauss_smooth_corner_map_np = torch.exp(-torch.arange(-f_wid, f_wid + 1)**2/2/sigma_smooth**2)
        Gscm_np = gauss_smooth_corner_map_np/gauss_smooth_corner_map_np.sum()
        Gscm = Gscm_np.clone().detach().to(dtype=torch.float32, device=device)
        con_values_g = F.conv2d(F.conv2d(out_vals.unsqueeze(0).unsqueeze(0), Gscm.reshape(1, 1, -1, 1), padding='same'), Gscm.reshape(1, 1, 1, -1), padding='same')
    else:
        s = minimum_dist_between_points_to_be_considered_different
        avg_corner_map_np = np.where(np.sqrt((np.arange(s)[:,None] - s/2 + .5)**2 + (np.arange(s) - s/2 + .5)**2) <= s/2, 1, 0)*np.exp(-((np.arange(s)[:,None] - s/2 + .5)**2 + (np.arange(s) - s/2 + .5)**2)/s**2/4)
        avg_corner_map = torch.from_numpy(avg_corner_map_np).clone().detach().to(dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        con_values_g = F.conv2d(out_vals.unsqueeze(0).unsqueeze(0), avg_corner_map, padding='same')
    
    max_pool_std = F.max_pool2d(con_values_g, kernel_size=int(2 * minimum_dist_between_points_to_be_considered_different + 1), stride=1, padding=minimum_dist_between_points_to_be_considered_different).squeeze(0).squeeze(0)
    threshold_mask = ((con_values_g > final_threshold) & (con_values_g == max_pool_std) & (corner_std_vals != 0))
    
    xy_threshold = threshold_mask.squeeze(0).squeeze(0).nonzero()
    
    xy_loc_val = torch.cat((xy_threshold,con_values_g[0,0,xy_threshold[:, 0], xy_threshold[:, 1]].unsqueeze(1)), dim=1)
    xy_loc_val_np = xy_loc_val.cpu().numpy()  # Convert frames_xy_loc_val to numpy array
    xy_loc_val_np = xy_loc_val_np[xy_loc_val_np[:, 2].argsort()[::-1]] # sort the corners strengths
    
    # Create a list with "batch size" entries, each containing frames_xy_loc_val_np
    result_list_xy_loc_val = [xy_loc_val_np.copy() for _ in range(input_tensor.shape[0])]
    
    if plt_flg:
        plt.figure(23)
        plt.imshow(out_vals.cpu().squeeze().numpy())
        plt.title('Result corner image')
    
    return result_list_xy_loc_val

def main_tst():
    H = 268
    W = 640
    fp = "/home/avraham/ggg/GargoyleEfcom/data/Vid.raw"
    f = load_data(fp, H, W)
    fut = unbit_pack_data_torch(f, 268, 640)

if __name__ == '__main__':
    main_tst()
