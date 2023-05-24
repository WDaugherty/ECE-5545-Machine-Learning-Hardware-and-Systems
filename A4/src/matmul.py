import torch
import torch.nn.functional as F


def matmul(A, B, method='naive', **kwargs):
    """
    Multiply two matrices.
    :param A: (N, M) torch tensor.
    :param B: (M, K) torch tensor.
    :param method:
    :return:
        Output matrix with shape (N, K)
    """
    method = method.lower()
    if method in ['naive', 'pytorch', 'torch']:
        return naive(A, B)
    elif method == 'svd':
        return svd(A, B, **kwargs)
    elif method in ['log', 'logmatmul']:
        return logmatmul(A, B, **kwargs)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(A, B, **kwargs):
    return A @ B


def svd(A, B, rank_A=None, rank_B=None):
    """
    Apply low-rank approximation (SVD) to both matrix A and B with rank rank_A
    and rank_B respectively.
    :param A: (N, M) pytorch tensor
    :param B: (M, K) pytorch tensor
    :param rank_A: None or int. None means use original A matrix.
    :param rank_B: None or int. None means use original B matrix.
    :return: a (N, K) pytorch tensor
    """
    if rank_A is not None:
        U_A, S_A, V_A = torch.svd(A)
        A = U_A[:, :rank_A] @ (S_A[:rank_A].reshape(-1,1)  * V_A[:, :rank_A].T)

    if rank_B is not None:
        U_B, S_B, V_B = torch.svd(B)
        B = U_B[:, :rank_B] @ (S_B[:rank_B].reshape(-1,1)  * V_B[:, :rank_B].T)

    return A @ B


def logmatmul(A, B, **kwargs):
    """
    Perform matrix multiplication using log-domain arithmetic.

    Args:
        A (torch.Tensor): Input tensor A of shape (m, n).
        B (torch.Tensor): Input tensor B of shape (n, p).
        **kwargs: Unused keyword arguments.

    Returns:
        torch.Tensor: Output tensor after performing log-domain matrix multiplication.
    """

    m, n = A.shape
    n, p = B.shape

    A_abs = torch.log2(torch.abs(A))
    B_abs = torch.log2(torch.abs(B))
    A_sign = torch.sign(A)
    B_sign = torch.sign(B)

    A_abs = A_abs.view(m, n, 1)
    B_abs = B_abs.view(1, n, p)
    A_sign = A_sign.view(m, n, 1)
    B_sign = B_sign.view(1, n, p)

    log_sum = A_abs + B_abs
    sign_product = A_sign * B_sign

    products = sign_product * (2 ** log_sum)
    output = torch.sum(products, dim=1)

    return output


