import numpy as np
import skimage
import skimage.transform as trans

"""
Some lines borrowed from: https://www.kaggle.com/sashakorekov/end-to-end-resnet50-with-tta-lb-0-93
"""


def rotate_clk_img_and_msk(img, msk):
    angle = np.random.choice((4, 6, 8, 10, 12, 14, 16, 18, 20))
    img_o = trans.rotate(img, angle, resize=False, preserve_range=True, mode='symmetric')
    msk_o = trans.rotate(msk, angle, resize=False, preserve_range=True, mode='symmetric')
    return img_o, msk_o

def rotate_cclk_img_and_msk(img, msk):
    angle = np.random.choice((-20, -18, -16, -14, -12, -10, -8, -6, -4))
    img_o = trans.rotate(img, angle, resize=False, preserve_range=True, mode='symmetric')
    msk_o = trans.rotate(msk, angle, resize=False, preserve_range=True, mode='symmetric')
    return img_o, msk_o

def flipping_img_and_msk(img, msk):
    img_o = np.flip(img, axis=1)
    msk_o = np.flip(msk, axis=1)
    return img_o, msk_o

def zoom_img_and_msk(img, msk):
    zoom_factor = np.random.choice((0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5))
    h, w = img.shape[:2]
    channels_img = img.shape[2] if img.ndim == 3 else 1
    channels_msk = msk.shape[2] if msk.ndim == 3 else 1

    # Width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # Resize the image and mask to the new dimensions
    img = trans.resize(img, (zh, zw), preserve_range=True, mode='symmetric')
    msk = trans.resize(msk, (zh, zw), preserve_range=True, mode='symmetric')

    region = np.random.choice((0, 1, 2, 3, 4))

    if zoom_factor < 1.0:  # Zooming out
        pad_h = (h - zh) // 2
        pad_w = (w - zw) // 2

        # Create padded arrays
        padded_img = np.zeros((h, w, channels_img), dtype=img.dtype)
        padded_msk = np.zeros((h, w), dtype=msk.dtype)  # Single-channel mask

        # Place the resized image and mask into the center of the padded arrays
        padded_img[pad_h:pad_h+zh, pad_w:pad_w+zw, :] = img
        padded_msk[pad_h:pad_h+zh, pad_w:pad_w+zw] = msk

        # Optionally, you can randomize the position of the zoomed-out image within the padded area
        start_y = np.random.randint(0, max(1, h - zh + 1))
        start_x = np.random.randint(0, max(1, w - zw + 1))

        outimg = padded_img[start_y:start_y + h, start_x:start_x + w, :]
        outmsk = padded_msk[start_y:start_y + h, start_x:start_x + w]

    else:  # Zooming in
        if region == 0:
            outimg = img[0:h, 0:w, :]
            outmsk = msk[0:h, 0:w]  # Single-channel mask
        elif region == 1:
            outimg = img[0:h, zw-w:zw, :]
            outmsk = msk[0:h, zw-w:zw]
        elif region == 2:
            outimg = img[zh-h:zh, 0:w, :]
            outmsk = msk[zh-h:zh, 0:w]
        elif region == 3:
            outimg = img[zh-h:zh, zw-w:zw, :]
            outmsk = msk[zh-h:zh, zw-w:zw]
        elif region == 4:
            marh = h // 2
            marw = w // 2
            outimg = img[(zh//2-marh):(zh//2+marh), (zw//2-marw):(zw//2+marw), :]
            outmsk = msk[(zh//2-marh):(zh//2+marh), (zw//2-marw):(zw//2+marw)]

    # Ensure the output is the same size as the input
    img_o = trans.resize(outimg, (h, w), preserve_range=True, mode='symmetric')
    msk_o = trans.resize(outmsk, (h, w), preserve_range=True, mode='symmetric')
    return img_o, msk_o

