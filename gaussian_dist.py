import torch
import numpy as np


def gaussian_likelihood(x, mu, log_std):
    """
    Args:
        x: Tensor with shape [batch, dim]
        mu: Tensor with shape [batch, dim]
        log_std: Tensor with shape [batch, dim] or [dim]

    Returns:
        Tensor with shape [batch]
    """

    dim = x.shape[-1]
    sigma = log_std.exp()
    log_p = -0.5*((((x-mu) / sigma).pow(2) + 2*log_std).sum(x.dim()-1) + dim*torch.log(torch.Tensor([2*torch.pi])))

    return log_p


if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """

    batch_size = 32
    dim = 10

    x = torch.rand(batch_size, dim)
    mu = torch.rand(batch_size, dim)
    log_std = torch.rand(dim)

    your_gaussian_likelihood = gaussian_likelihood(x, mu, log_std)
    print(your_gaussian_likelihood)
 