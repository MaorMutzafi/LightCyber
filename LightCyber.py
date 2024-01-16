import torch
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import time 

def LightCyberAlg(input_tensor, sigma, win_sz_std, max_corners_harris=10, 
                        low_thresh_harris=300, radius=5, threshold=4, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                        use_G1=True, without_loops_flg=False):
    """
    Light-Cyber-Alg: Harris corner detector with standard deviation window for a batch of grayscale images.
    
    Parameters
    ----------
    input_tensor : torch.Tensor
        Batch of grayscale image tensors with shape (batch_size, height, width).
    
    sigma : float
        Standard deviation of the smoothing Gaussian filter.
    
    win_sz_std : int
        Size of the window for standard deviation calculation around each corner.
    
    max_corners_harris : int, optional
        Maximum number of corners to return for each frame.
    
    low_thresh_harris : float, optional
        The low bound sensitivity for corner detection.
    
    radius : int, optional
        Radius of region considered in non-maximal suppression.
    
    threshold : float, optional
        Threshold for selecting strong corners.
    
    use_G1 : bool, optional
        Flag to use separable Gaussian filters (True) or a combined 2D filter (False).
    
    without_loops_flg : bool, optional
        Flag to use vectorized implementation without loops (True) or with loops (False).

    Returns
    -------
    corner_strengths : torch.Tensor
        Mean corner strengths over the entire batch, same size as input.
    
    std_values : torch.Tensor
        Mean standard deviation values over the entire batch, same size as input.
    """
    # Ensure input is in float32 and subtract the mean frame
    input_tensor = input_tensor.type(torch.float32)
    input_tensor = input_tensor - torch.mean(input_tensor, dim=0)

    # Derivative masks for image gradients
    dx = torch.tensor([[-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    dy = dx.transpose(2, 3)

    # Convolution for gradients in x and y directions
    Ix = F.conv2d(input_tensor.unsqueeze(1), dx, padding='same')
    Iy = F.conv2d(input_tensor.unsqueeze(1), dy, padding='same')

    # Gaussian filter preparation
    f_wid = round(3 * sigma)
    G1_1D = torch.tensor(norm.pdf(torch.arange(-f_wid, f_wid + 1), loc=0, scale=sigma), dtype=torch.float32).to(device)
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
    max_pool = F.max_pool2d(harris.unsqueeze(1), kernel_size=int(2 * radius + 1), stride=1, padding=radius)
    corners = harris * ((harris == max_pool.squeeze(1)) & (harris > low_thresh_harris))

    # Parameters for batch and image dimensions
    batch_size, height, width = input_tensor.shape

    # Find top max_corners_harris corners in each frame
    corners_vals, idx = torch.topk(corners.view(corners.size(0), -1), k=max_corners_harris)
    corners_y, corners_x = idx.div(corners.size(2), rounding_mode='floor'), idx % corners.size(2)

    if without_loops_flg:
        # Vectorized computation of standard deviation in corner windows
        disp_x, disp_y = torch.meshgrid(torch.arange(-win_sz_std // 2, win_sz_std // 2 + 1).to(device), 
                                        torch.arange(-win_sz_std // 2, win_sz_std // 2 + 1).to(device), indexing='ij')
        disp_x = disp_x.reshape(1, 1, -1, 1, 1)
        disp_y = disp_y.reshape(1, 1, -1, 1, 1)
        
        window_x = (corners_x.view(batch_size, max_corners_harris, 1, 1, 1) + disp_x).clamp(min=0, max=width - 1)
        window_y = (corners_y.view(batch_size, max_corners_harris, 1, 1, 1) + disp_y).clamp(min=0, max=height - 1)
        
        std_vals = input_tensor[torch.arange(batch_size).view(batch_size, 1, 1, 1, 1).expand_as(window_x), window_y, window_x].std(dim=[2, 3, 4])
        corner_std_vals = torch.zeros_like(corners)  # Initialize corner std values
        corner_std_vals[
            torch.arange(batch_size).view(-1, 1, 1),
            corners_y[:, :max_corners_harris].unsqueeze(2),
            corners_x[:, :max_corners_harris].unsqueeze(2)
        ] = std_vals[:, :max_corners_harris].unsqueeze(2)
        
        corner_strengths = corners.mean(dim=0)
        corner_std_vals = corner_std_vals.mean(dim=0)
    else:
        # Loop-based computation of corner strengths and standard deviation values
        corner_strengths = torch.zeros_like(input_tensor[0])
        corner_std_vals = torch.zeros_like(input_tensor[0])
        
        for b in range(batch_size):
            for i in range(max_corners_harris):
                corner_strengths[corners_y[b, i], corners_x[b, i]] += corners_vals[b, i]
                window = input_tensor[b, 
                                      max(corners_y[b, i] - win_sz_std // 2, 0):min(corners_y[b, i] + win_sz_std // 2 + 1, height),
                                      max(corners_x[b, i] - win_sz_std // 2, 0):min(corners_x[b, i] + win_sz_std // 2 + 1, width)]
                corner_std_vals[corners_y[b, i], corners_x[b, i]] += window.std()

        # Average over the entire batch
        corner_strengths /= batch_size
        corner_std_vals /= batch_size
    
    # Apply threshold to corner_std_vals and find coordinates
    use_corner_std_flg = True
    if use_corner_std_flg:
        std_values_g = F.conv2d(F.conv2d(corner_std_vals.unsqueeze(0).unsqueeze(0), G1, padding='same'), G1_transpose, padding='same')
        max_pool_std = F.max_pool2d(std_values_g, kernel_size=int(2 * radius + 1), stride=1, padding=radius).squeeze(0).squeeze(0)
        threshold_mask = ((corner_std_vals > threshold) & (std_values_g == max_pool_std) & (corner_std_vals != 0))
    else:
        con_values_g = F.conv2d(F.conv2d(corner_strengths.unsqueeze(0).unsqueeze(0), G1, padding='same'), G1_transpose, padding='same')
        max_pool_std = F.max_pool2d(con_values_g, kernel_size=int(2 * radius + 1), stride=1, padding=radius).squeeze(0).squeeze(0)
        threshold_mask = ((corner_std_vals > threshold) & (con_values_g == max_pool_std) & (corner_std_vals != 0))
    xy_threshold = threshold_mask.squeeze(0).squeeze(0).nonzero()
    
    return corner_strengths, corner_std_vals, xy_threshold

def load_numpy_dat(file_path, W, H):
    fid = open(file_path, 'rb')
    
    packed = 1.5
    np.fromfile(fid, dtype=np.uint8, count=64+int(W * H * packed))
    
    dat_all = np.fromfile(fid, dtype=np.uint8)
    return dat_all

def depack_raw_data_torch(dat_all, device='cuda', n_fr = None):
    packed = 1.5
    
    # Convert to PyTorch tensor
    dat_all_tr = torch.tensor(dat_all, dtype=torch.uint8, device=device)
    fr_len = int(W * H * packed + 64)
    n_fr = len(dat_all_tr) // fr_len if n_fr is None else n_fr
    
    qab1, qab2, qab3 = dat_all_tr[:fr_len*n_fr].view(n_fr,fr_len)[:,8:-56].view(n_fr,-1,3).permute(2,1,0).view(3,H,W//2,n_fr).permute(0,2,1,3).type(torch.int16).unbind(dim=0)
    
    # Process qab2 and combine qab values
    qab2_least4 = qab2 % 16
    qab2_most4 = qab2 // 16
    odd_col_pix = qab1 + 256 * qab2_least4
    even_col_pix = qab3 * 16 + qab2_most4
    
    comb = torch.zeros((2, W // 2, H, n_fr), dtype=torch.int16, device=device)
    comb[0, :, :, :] = odd_col_pix
    comb[1, :, :, :] = even_col_pix
    
    # Reshape to match the final expected shape
    comb = comb.permute(*reversed(range(len(comb.shape)))).reshape(*reversed((W , H, n_fr))).permute(*reversed(range(len((W , H, n_fr))))).permute(2, 1, 0)
    return comb

def LoadGainOffset_torch(GainOffsetDirPath, W, H, device):
    # Read Gain and Offset using NumPy, then convert to PyTorch tensors
    Offset = torch.from_numpy(np.fromfile(GainOffsetDirPath + 'Offset.raw', dtype=np.uint16).reshape(W, H).T.astype(np.int16)).to(device)
    Gain = torch.from_numpy(np.fromfile(GainOffsetDirPath + 'Gain.raw', dtype=np.float64).reshape(W, H).T.astype(np.float32)).to(device)
    
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
    neighborhoods = neighborhood_y * W + neighborhood_x

    # Flatten the coordinates of bad pixels to get their indices in the flattened Gain tensor
    bad_pixels_flat = y_bad * W + x_bad

    # Create a mask where each bad pixel's neighborhood is compared against all bad pixels
    bad_pixel_mask = neighborhoods.unsqueeze(2) == bad_pixels_flat.unsqueeze(0).unsqueeze(1)
    
    # Any neighborhood that contains a bad pixel is replaced with sentinel_value
    neighborhoods[bad_pixel_mask.any(dim=2)] = sentinel_value

    return neighborhoods, bad_pixels_flat

def bad_pxl_corr_torch(data_raw_torch, neighborhoods, bad_pixels_flat):
    batch_size, H, W = data_raw_torch.shape
    sentinel_value = W * H

    # Flatten data_raw_torch to 2D
    data_raw_flat = data_raw_torch.reshape(batch_size, -1)
    
    ind_ = (neighborhoods == sentinel_value)
    neighborhoods[ind_] = 0
    
    # Expand neighborhoods and bad_pixels_flat to match batch size
    neighborhoods_expanded = neighborhoods.unsqueeze(0).expand(batch_size, -1, -1)
    bad_pixels_flat_expanded = bad_pixels_flat.unsqueeze(0).expand(batch_size, -1)

    # Gather the values from the neighborhoods
    neighborhood_values = torch.gather(data_raw_flat, 1, neighborhoods_expanded.view(batch_size, -1))
    neighborhood_values = neighborhood_values.view(batch_size, -1, 8)

    # Replace the sentinel values with NaN
    neighborhood_values[ind_.unsqueeze(0).expand(batch_size, -1, -1)] = float('nan')

    # Compute the mean of the neighborhoods, ignoring NaNs
    neighborhood_means = torch.nanmean(neighborhood_values, dim=2)

    # Replace bad pixel values with the computed means in the original data
    data_raw_flat.scatter_(1, bad_pixels_flat_expanded, neighborhood_means)

    # Reshape data_raw_torch back to original shape
    data_raw_torch = data_raw_flat.view(batch_size, H, W)

    return data_raw_torch

def visualize_results(corner_strengths, std_values, xy_threshold, input_image):
    corner_strengths = corner_strengths.cpu()
    std_values = std_values.cpu()
    xy_threshold = xy_threshold.cpu()
    input_image = input_image.cpu().squeeze().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)  # Set sharex and sharey to True

    # Visualize original image
    axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Visualize corner strengths
    axs[1].imshow(corner_strengths.cpu(), cmap='hot')
    axs[1].set_title('Corner Strengths')
    axs[1].axis('off')

    # Visualize standard deviation values
    axs[2].imshow(std_values.cpu(), cmap='hot')
    axs[2].scatter(xy_threshold[:, 1].cpu(), xy_threshold[:, 0].cpu(), c='blue', s=10)  # Points where std_values > threshold
    axs[2].set_title('Standard Deviation Values (with Threshold Points)')
    axs[2].axis('off')

    plt.show()

def main():
    # camera params
    W = 640
    H = 268
    # alg params
    sigma = 1.5
    win_sz_std = 5
    max_corners_harris = 10
    low_thresh_harris = 300
    radius = 3
    threshold = 2
    use_cuda = True # use cuda if avaible

    file_path = 'C:/Users/User/Documents/Mafaat_new_Topics/LightCyber/Exp/Kfir_11June2023/Data/OS_12062023024132_70m_4mW/Vid.raw'
    GainOffsetDirPath = 'C:/Users/User/Documents/Mafaat_new_Topics/LightCyber/Exp/Kfir_11June2023/Data/GainOffset/'

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("cuda is not available, using CPU")
        device = torch.device("cpu")

    # load the data using numpy
    dat_all_np = load_numpy_dat(file_path, W, H)

    # pre-process
    Gain, Offset = LoadGainOffset_torch(GainOffsetDirPath, W, H, device=device.type)
    neighborhoods, bad_pixels_flat = find_bad_pxl_torch(Gain, W, H, device)


    start_time = time.time()

    # Organizing the data
    dat_all_tr = depack_raw_data_torch(dat_all_np, device=device.type, n_fr = 500)
    dat_nuc_tr = AppGainOffset_torch(dat_all_tr, Gain, Offset, device)
    Frs_corrected = bad_pxl_corr_torch(dat_nuc_tr, neighborhoods, bad_pixels_flat)

    # run the alg.
    corner_strengths, std_values, xy_threshold = LightCyberAlg(Frs_corrected, sigma = sigma, win_sz_std = win_sz_std, max_corners_harris = max_corners_harris, 
        low_thresh_harris = low_thresh_harris, radius = radius, threshold = threshold, device = device, 
        use_G1=True, without_loops_flg=True)
    # corner_strengths and std_values are your outputs.
    # xy_threshold is the coordinates where std_values > threshold after max pooling.
    print(time.time() - start_time)
    # Show the results
    visualize_results(corner_strengths, std_values, xy_threshold, Frs_corrected[0])


if __name__ == "__main__":
    main()