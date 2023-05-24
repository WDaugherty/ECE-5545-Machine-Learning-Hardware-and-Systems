import torch
import torch.nn.functional as F


def conv2d(x, k, b, method='naive'):
    """
    Convolution of single instance and single input and output channel
    :param x:  (H, W) PyTorch Tensor
    :param k:  (Hk, Wk) PyTorch Tensor
    :param b:  (1,) PyTorch tensor or scalar
    :param method: Which method do we use to implement it. Valid choices include
                   'naive', 'torch', 'pytorch', 'im2col', 'winograd', and 'fft'
    :return:
        Output tensor should have shape (H_out, W_out)
    """
    method = method.lower()
    if method == 'naive':
        return naive(x, k, b)
    elif method in ['torch', 'pytorch']:
        return pytorch(x, k, b)
    elif method == 'im2col':
        return im2col(x, k, b)
    elif method == 'winograd':
        return winograd(x, k, b)
    elif method == 'fft':
        return fft(x, k, b)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(x, k, b):
    """ Sliding window solution. """
    output_shape_0 = x.shape[0] - k.shape[0] + 1
    output_shape_1 = x.shape[1] - k.shape[1] + 1
    result = torch.zeros(output_shape_0, output_shape_1)
    for row in range(output_shape_0):
        for col in range(output_shape_1):
            window = x[row: row + k.shape[0], col: col + k.shape[1]]
            result[row, col] = torch.sum(torch.multiply(window, k))
    return result + b


def pytorch(x, k, b):
    """ PyTorch solution. """
    return F.conv2d(
        x.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        k.unsqueeze(0).unsqueeze(0),  # (1, 1, Hk, Wk)
        b   # (1, )
    ).squeeze(0).squeeze(0)  # (H_out, W_out)


def im2col(x, k, b):
    """ TODO: implement `im2col`"""
    # Source for im2col: https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
    #Co-coded with ChatGPT and copilot 
    # Calculate output dimensions
    output_shape_0 = x.shape[0] - k.shape[0] + 1
    output_shape_1 = x.shape[1] - k.shape[1] + 1

    # Create im2col matrix
    im2col_matrix = []
    for row in range(output_shape_0):
        for col in range(output_shape_1):
            window = x[row: row + k.shape[0], col: col + k.shape[1]]
            im2col_matrix.append(window.flatten())
    im2col_matrix = torch.stack(im2col_matrix).transpose(0, 1)

    # Perform matrix multiplication and add the bias
    result = torch.matmul(k.flatten(), im2col_matrix) + b
    return result.view(output_shape_0, output_shape_1)

def winograd(x, k, b):
    """
    Perform Winograd convolution for 3x3 kernels.

    Args:
        x (torch.Tensor): Input tensor of shape (height, width).
        k (torch.Tensor): Kernel tensor of shape (3, 3).
        b (torch.Tensor): Bias tensor.

    Returns:
        torch.Tensor: Output tensor after performing Winograd convolution.
    """
    if k.shape != (3, 3):
        raise ValueError("Winograd convolution is only supported for 3x3 kernels")

    # Pad the input tensor
    height, width = x.shape
    kernel_height, kernel_width = k.shape
    padded_x = F.pad(x, (0, 1, 0, 1))

    # Define Winograd matrices
    G_mat = torch.tensor([
        [1, 0, 0],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0, 0, 1]], 
        dtype=x.dtype)

    B_mat = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, -1, 1],
        [-1, 1, 1, 0],
        [0, 0, 0, -1]], 
        dtype=x.dtype)

    A_mat = torch.tensor([
        [1, 0],
        [1, 1],
        [1, -1],
        [0, -1]], 
        dtype=x.dtype)

    # Perform Winograd convolution
    output = torch.zeros(height - kernel_height + 2, width - kernel_width + 2)
    for row in range(0, height - kernel_height + 2, 2):
        for col in range(0, width - kernel_width + 2, 2):
            output[row:row+2, col:col+2] += A_mat.T @ ((G_mat @ k @ G_mat.T) * (B_mat.T @ padded_x[row:row+4, col:col+4] @ B_mat)) @ A_mat

    # Add bias and return the result
    return output[:-1, :-1] + b

def fft(x, k, b):
    """
    Perform Fast Fourier Transform (FFT) based convolution.

    Args:
        x (torch.Tensor): Input tensor x of shape (height, width).
        k (torch.Tensor): Input tensor k representing the kernel of shape (kernel_height, kernel_width).
        b (float): Bias term.

    Returns:
        torch.Tensor: Output tensor after performing FFT-based convolution.
    """

    kernel_height, kernel_width = k.shape
    height, width = x.shape

    # Pad the kernel to the size of the input
    padded_kernel = torch.zeros((height, width), dtype=x.dtype)
    grid_y, grid_x = torch.meshgrid(torch.arange(kernel_height), torch.arange(kernel_width))
    new_grid_y = (grid_y.flip(0) - kernel_height // 2) % padded_kernel.size(0)
    new_grid_x = (grid_x.flip(1) - kernel_width // 2) % padded_kernel.size(1)
    padded_kernel[new_grid_y, new_grid_x] = k[grid_y, grid_x]

    # Perform 2D FFT on the input and padded kernel
    input_fft = torch.fft.fft2(x)
    kernel_fft = torch.fft.fft2(padded_kernel)

    # Perform element-wise multiplication and inverse FFT
    result = torch.real(torch.fft.ifft2(kernel_fft * input_fft))

    # Crop the result using slicing based on the kernel size and add the bias
    if kernel_height == 1:
        return result + b
    return result[(kernel_height - 1) // 2:-(kernel_height - 1) // 2, (kernel_width - 1) // 2:-(kernel_width - 1) // 2] + b
