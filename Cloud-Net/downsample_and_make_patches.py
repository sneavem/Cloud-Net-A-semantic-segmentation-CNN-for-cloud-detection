import cv2
import numpy as np

def extract_patches(image_array, patch_size, pad_value=0):
    patches = []
    height, width = image_array.shape

    for y in range(0, height, patch_size[1]):
        for x in range(0, width, patch_size[0]):
            patch = image_array[y:y + patch_size[1], x:x + patch_size[0]]
            # Apply zero padding if patch size is smaller
            if patch.shape[0] < patch_size[1] or patch.shape[1] < patch_size[0]:
                padded_patch = np.full(patch_size, pad_value, dtype=image_array.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                patches.append(padded_patch)
            else:
                patches.append(patch)
    return patches

if __name__ == '__main__':
    patch_size_band1 = (752,752 )  # (height, width)
    patch_size_band2 = (384, 384)

    # Assuming band1_array and band2_array are the aligned images loaded as numpy arrays
    band1_array = cv2.imread('aligned_first_band_100m.tif', cv2.IMREAD_GRAYSCALE)
    band2_array = cv2.imread('aligned_second_band_200m.tif', cv2.IMREAD_GRAYSCALE)

    patches_band1, patches_band2 = extract_corresponding_patches(band1_array, band2_array, patch_size_band1, patch_size_band2)


