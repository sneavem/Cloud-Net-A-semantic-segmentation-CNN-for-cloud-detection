import os
import numpy as np
from PIL import Image
import re
import cv2
from scipy.ndimage import label, find_objects
# from osgeo import gdal

def fit_image(input_file, output_file, target_size):
    """
    Resize an image to a target size by either cropping or padding, separately for height and width, and save it to a new file.

    :param input_file: Path to the input TIFF file
    :param output_file: Path to the output TIFF file with resizing
    :param target_size: Tuple (height, width) of the target size
    """
    with Image.open(input_file) as img:
        img_array = np.array(img)
        print(f'image height:  {img_array.shape}')
        img_height, img_width = img_array.shape

        target_height, target_width = target_size

        # Handle height
        if img_height > target_height:
            crop_y = (img_height - target_height) // 2
            img_array = img_array[crop_y:crop_y + target_height, :]
        elif img_height < target_height:
            pad_y = (target_height - img_height) // 2
            padded_array = np.zeros((target_height, img_width), dtype=img_array.dtype)
            padded_array[pad_y:pad_y + img_height, :] = img_array
            img_array = padded_array

        # update img_height to the new size after padding or cropping
        img_height = img_array.shape[0]

        # Handle width
        if img_width > target_width:
            crop_x = (img_width - target_width) // 2
            img_array = img_array[:, crop_x:crop_x + target_width]
        elif img_width < target_width:
            pad_x = (target_width - img_width) // 2
            padded_array = np.zeros((img_height, target_width), dtype=img_array.dtype)
            padded_array[:, pad_x:pad_x + img_width] = img_array
            img_array = padded_array

        resized_img = Image.fromarray(img_array)
        resized_img.save(output_file, format='TIFF')

def rotate_and_crop(image_path, output_path_rotated):
    """
    Process an image to rotate and crop it based on the bounding box of non-zero points.
    
    Parameters:
    - image_path: str, path to the input image.
    - output_path_rotated: str, path to save the rotated and cropped image.
    
    Returns:
    - rotated_image: The processed image after rotation and cropping.
    """
    pil_image = Image.open(image_path)

    if pil_image.mode in ("RGB", "RGBA"):
        print("Color image detected, converting to grayscale.")
        gray = pil_image.convert("L")
        gray_np = np.array(gray, dtype=np.uint8)
    else:
        print("Grayscale image detected.")
        gray_np = np.array(pil_image, dtype=np.uint8)

    _, mThreshold = cv2.threshold(gray_np, 50, 255, cv2.THRESH_BINARY)
    Points = cv2.findNonZero(mThreshold)

    if Points is not None:
        rotated_rect = cv2.minAreaRect(Points)

        angle = rotated_rect[2]
        (h, w) = gray_np.shape[:2]
        center = rotated_rect[0]
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        mSource_Bgr = cv2.cvtColor(gray_np, cv2.COLOR_GRAY2BGR)
        rotated_image = cv2.warpAffine(gray_np, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        box = cv2.boxPoints(rotated_rect)
        box = np.intp(cv2.transform(np.array([box]), rotation_matrix))[0]

        x, y, w, h = cv2.boundingRect(box)
        cropped_image = rotated_image[y:y+h, x:x+w]

        cv2.imwrite(output_path_rotated, cropped_image)
        print(f"Rotated and cropped image saved to {output_path_rotated}")
        
        return cropped_image
    else:
        print("No non-zero points detected in the image.")
        return None

def extract_unique_sceneids(training_folder_root, training_folder):
    """Extract unique scene IDs for training."""
    path = os.path.join(training_folder_root, training_folder)
    files = [f for f in os.listdir(path) if f.endswith('.TIF')]
    
    sceneid_list = []
    for file_name in files:
        sceneid = re.search(r'LC\w+', file_name).group(0)
        sceneid_list.append(sceneid)
    
    return sorted(set(sceneid_list))


def extract_rowcol_each_patch(name):
    """Extract row and column numbers from the patch name."""
    pattern = r'_(\d+)_by_(\d+)_'
    
    match = re.search(pattern, name)
    
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return row, col
    else:
        raise ValueError("Filename does not match expected pattern")

def get_patches_for_sceneid(path, sceneid):
    """Find all patch masks corresponding to a unique sceneID."""
    files = [f for f in os.listdir(path) if sceneid in f and f.endswith('.TIF')]
    return sorted(files)

def stitch_scene(folder_root, folder, sceneid, pr_patch_size, save=True):
    """Stitch patches together to form a complete scene mask for each folder.
    Returns the width and height of the uncropped stitced image as a tuple (width, height)
    """
    print(f'Working on {folder} for sceneID: {sceneid}')
    folder_path = os.path.join(folder_root, folder)
    scid_related_patches = get_patches_for_sceneid(folder_path, sceneid)
    assert scid_related_patches, f"No patches found for {sceneid} in {folder}"

    # Determine the maximum row and column indices from the patch names
    max_row, max_col = 0, 0
    for patch_name in scid_related_patches:
        row, col = extract_rowcol_each_patch(patch_name)
        if row > max_row:
            max_row = row
        if col > max_col:
            max_col = col
    print(f'Max row: {max_row}, max_col: {max_col}')
    
    # Calculate the size of the complete scene mask
    complete_height = pr_patch_size * max_row
    complete_width = pr_patch_size * max_col
    
    # Initialize the complete mask with zeros (black)
    complete_pred_mask = np.zeros((complete_height, complete_width), dtype=np.uint8)
    
    for patch_name in scid_related_patches:
        predicted_patch_path = os.path.join(folder_root, folder, patch_name)
        predicted_patch = np.array(Image.open(predicted_patch_path))
        w, h = predicted_patch.shape[:2]
        
        assert w == pr_patch_size and h == pr_patch_size, 'patch size different than expected'
        
        # Extract row and column number from the patch name
        patch_row, patch_col = extract_rowcol_each_patch(patch_name)

        # Stitching up patch masks together
        complete_pred_mask[(patch_row-1)*pr_patch_size : patch_row*pr_patch_size,
                           (patch_col-1)*pr_patch_size : patch_col*pr_patch_size] = predicted_patch
    
    # Save the stitched image
    if save:
        complete_folder = f'stitched_{folder}'
        os.makedirs(os.path.join(folder_root, complete_folder), exist_ok=True)
        
        save_path = os.path.join(folder_root, complete_folder, f'{sceneid}.TIF')
        Image.fromarray(complete_pred_mask).save(save_path)
        print(f'Saved stitched image for sceneID {sceneid} in {complete_folder}')
        # rotate_and_crop(save_path, save_path)

    return complete_width, complete_height 


if __name__ == "__main__":
    training_folder_root = '38-Cloud_training'
    training_folders = ['train_blue', 'train_green', 'train_gt', 'train_nir', 'train_red']
    
    og_patch_size = 384
    
    # Stitch and derotate patches 
    # Don't need to crop. use CSV provided by Cloudnet repo with mostly useful patches 
    # Just switch train_nir patches for train_lwir patches. 

    
    
    all_uniq_sceneid = extract_unique_sceneids(training_folder_root, training_folders[0])
    for sceneid in all_uniq_sceneid:
        stitch_scene(training_folder_root, training_folders[0], all_uniq_sceneid, og_patch_size)

    # BlueFox has size 752x480.
    # GSD of landsat 8 visual bands is 30m. GSD of Bluefox on beavercube ~100m

    # Boson is LWIR  has size 640x512 or 320x256 with GSD of ~200m. Don't use NIR band 




    
    
    
    

    








