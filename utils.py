import torch
import skimage.metrics as skm
import lpips

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    (batch_size, height, width, channels).
    """
    psnr = skm.peak_signal_noise_ratio(
        img1, #.cpu().numpy().transpose(0, 2, 3, 1),
        img2, #.cpu().numpy().transpose(0, 2, 3, 1),
        data_range=img1.max().item(),
    )
    return psnr


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    ssim = skm.structural_similarity(
        img1, #.cpu().numpy().transpose(0, 2, 3, 1),
        img2, #.cpu().numpy().transpose(0, 2, 3, 1),
        multichannel=True,
        data_range=img1.max().item()
    )
    return ssim


def calculate_lpips(
    img1: torch.Tensor, img2: torch.Tensor, net="alex") -> float:
    loss_fn_alex = lpips.LPIPS(net=net, version="net-lin") # best forward scores
    lpips_loss = loss_fn_alex(img1, img2)#.mean()
    return lpips_loss#.item()
