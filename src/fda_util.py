import numpy as np
import argparse
from math import floor

def FDA_source_to_target_np(src_img, trg_img, L=0.1 ):
    """
    Args:
    src_img (np.array): numpy representation of 3xHxW image from source domain
    trg_img (np.array): numpy representation of 3xHxW image from target domain
    L: scalar in [0, 1]. For L = 0, src amplitudes stays unchanged.
          For L = 1, src amplitudes fully substituted with target amplitudes.
          See https://arxiv.org/pdf/2004.05498.pdf for details
    
    Return: 
    Source image with target low frequencies applied to source
    """

    src_img_np = np.transpose(src_img, (2, 0, 1)) #.cpu().numpy()
    trg_img_np = np.transpose(trg_img, (2, 0, 1)) #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return np.transpose(src_in_trg, (1, 2, 0)).astype(int)


def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape

    b = (  np.floor(np.amin((h,w))*L) / 2 ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    
    return a_src

def parse_arguments():
    parser = argparse.ArgumentParser(description="FDA", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--target_size", type=str, default = "1",
        help = "Number of shots to learn the mapping from source to target"
    )

    # PATHS
    parser.add_argument("--dataset_root", type=str, default="./datasets/svox/images", help="Root path of the dataset")
    parser.add_argument("--train_q", type=str, default="train/queries", help="Path train query")
    parser.add_argument("--val_q", type=str, default="val/queries", help="Path val query")
    parser.add_argument("--beta", type=float, default=0.001, help = "Beta hyperparameter")
    parser.add_argument("--tune_beta", default=False, action = "store_true", help="If True perform tuning on beta")
    parser.add_argument("--targets", nargs = "+", default=["night", "overcast", "rain", "snow", "sun"], help="Domains of adaptation")
    parser.add_argument("--shots", nargs = "+", default=['1', '5', '20', '50', 'ALL'], help="Number of shots for domain adaptation")

    return parser.parse_args()

def scale(np_img, cmin=0., cmax=255., low=0., high=255.):
    cscale = cmax - cmin
    scale = float(high - low) / cscale
    bytedata = (np_img * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0
    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)