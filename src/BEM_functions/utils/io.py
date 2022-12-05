import torch


def warp_tensor(x: torch.Tensor):
    n_ = x.shape[-1]
    if n_ == 1:
        x = x.squeeze(-1)
    elif n_ == 2:
        x = x[..., 0] + x[..., 1] * 1j
    else:
        raise NotImplemented(
            "A Tensor with shape = (*, {}) can not be warped, "
            "we only support 1 (Scalar) or 2 (Complex) for the last dim".format(n_)
        )
    
    return x

def unwarp_tensor(x: torch.Tensor):
    n_ = 1
    if x.dtype == torch.complex32 or x.dtype == torch.complex64 or x.dtype == torch.complex128:
        n_ = 2
    
    if n_ == 1:
        x = x.unsqueeze(-1)
    elif n_ == 2:
        x = torch.stack(
            (x.real, x.imag),
            -1
        )
    
    return x