import numpy as np
from math import floor

# TODO Remove after testing
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def src_to_target_np(src_img, trg_img, beta = 0.1):
    """
    Args:
    src_img (np.array): numpy representation of 3xHxW image from source domain
    trg_img (np.array): numpy representation of 3xHxW image from target domain
    beta: scalar in [0, 1]. For beta = 0, src amplitudes stays uncchanged.
          For beta = 1, src amplitudes fully substituted with target amplitudes.
          See https://arxiv.org/pdf/2004.05498.pdf for details
    
    Return: 
    Source image with target low frequencies applied to source
    """

    # 2D Fourier Transform
    src_ft = np.fft.fft2(src_img)
    trg_ft = np.fft.fft2(trg_img)

    # Separate amplitudes and phases
    src_amp = np.abs(src_ft)
    src_phase = np.angle(src_ft)
    trg_amp = np.abs(trg_ft)
    trg_amp = np.angle(trg_ft)

    replaced_src_amp = replace_low_freq_np(src_amp, trg_amp, beta)

    # Recompose source FT with replace low level frequencies
    src_ft = replaced_src_amp * np.exp(1j*2*np.pi*src_phase)
    

    # Inverse 2D FT
    # Loro lo fanno diverso... E in piu mi rimane una parte immaginaria residua
    src_to_trg_img = np.fft.ifft2(src_ft)
    print(src_to_trg_img)

    return src_to_trg_img


def replace_low_freq_np(src_amp, trg_amp, beta):
    _, h, w = src_amp.shape

    # Center of the image coordinates
    c_h = floor(h/2)
    c_w = floor(w/2)

    b = int(beta*floor(min(h, w)) / 2 )

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b 
    w2 = c_w + b + 1

    src_amp[:, h1:h2, w1:w2] = trg_amp[:, h1:h2, w1:w2]

    # Test plot: visualize the replaced region
    # rectangle = patches.Rectangle((w1, h2), 2*b+1, 2*b+1, facecolor = 'none', linewidth = 1, edgecolor = 'r')
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(src_amp[0, :, :])
    # ax.add_patch(rectangle)

    # plt.show()

    return src_amp


if __name__ == '__main__':

    # Run test
    src_img_path = '/home/valerio/francesco/adageo-WACV2021/src/datasets/svox/examples/RobotCar_snow.jpg'
    trg_img_path = '/home/valerio/francesco/adageo-WACV2021/src/datasets/svox/examples/RobotCar_rain.jpg'

    # Read images in numpy array
    src_img = np.array(Image.open(src_img_path), dtype=float)
    trg_img = np.array(Image.open(trg_img_path), dtype=float)

    # Pass images to the function
    src_to_trg_img = src_to_target_np(src_img, trg_img)

    # Plot source image before and after
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(src_img)
    # ax[1].imshow(src_to_trg_img)
    
    # plt.show()