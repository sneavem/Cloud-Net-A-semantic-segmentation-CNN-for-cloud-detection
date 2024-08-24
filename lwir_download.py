import os
import subprocess
import argparse
import numpy as np
from PIL import Image
import re
import cv2
from scipy.ndimage import label, find_objects
from repatch_data import extract_unique_sceneids, stitch_scene, rotate_and_crop, fit_image

def extract_patches(tiff_file, patch_size, output_dir, scene_id, gt_dir):
    """
    Extracts patches from a TIFF file and saves them with a specified naming convention using Pillow.

    :param tiff_file: Path to the input TIFF file
    :param patch_size: Tuple (height, width) of the patches
    :param output_dir: Directory to save the patches
    :param scene_id: Scene ID used for naming the patches
    """
    with Image.open(tiff_file) as img:
        img_width, img_height = img.size
        patch_height = patch_width = patch_size

        num_patches_y = img_height // patch_height
        num_patches_x = img_width // patch_width

        # Adds an extra iteration as some gt images have one more column or row than expected 
        for i in range(num_patches_y + 1):  
            for j in range(num_patches_x + 1):  
                if i < num_patches_y and j < num_patches_x:
                    start_y = i * patch_height
                    end_y = start_y + patch_height
                    start_x = j * patch_width
                    end_x = start_x + patch_width

                    patch = img.crop((start_x, start_y, end_x, end_y))
                else:
                    # Create an extra patch of all zeros
                    patch = Image.new('L', (patch_width, patch_height), 0)

                pattern = re.compile(r'(\d+)_' + re.escape(f'{i+1}') + r'_by_' + re.escape(f'{j+1}') 
                                     + r'_' + re.escape(f'{scene_id}'))
                patch_num = 100  # Default value
                
                match_flag = False
                for filename in os.listdir(gt_dir):
                    match = pattern.search(filename)
                    if match:
                        patch_num = int(match.group(1))
                        match_flag = True
                
                # assert match_flag, f'failed on pattern: {pattern}'

                if match_flag:
                    patch_name = f'lwir_patch_{patch_num}_{i+1}_by_{j+1}_{scene_id}.TIF'
                    os.makedirs(output_dir, exist_ok=True)  
                    patch_path = os.path.join(output_dir, patch_name)

                    patch.save(patch_path, format='TIFF')

def zero_pad_image(input_file, output_file, target_size):
    """
    Zero pads an image to a target size (both height and width) and saves it to a new file.

    :param input_file: Path to the input TIFF file
    :param output_file: Path to the output TIFF file with padding
    :param target_size: Tuple (target_height, target_width) of the padded image size
    """
    with Image.open(input_file) as img:
        img_array = np.array(img)
        img_height, img_width = img_array.shape

        target_height, target_width = target_size

        padded_array = np.zeros((target_height, target_width), dtype=img_array.dtype)

        pad_y = (target_height - img_height) // 2
        pad_x = (target_width - img_width) // 2

        padded_array[pad_y:pad_y + img_height, pad_x:pad_x + img_width] = img_array

        padded_img = Image.fromarray(padded_array)
        padded_img.save(output_file, format='TIFF')



def download_landsat_band(scene_id, band_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    command = [
        'pylandsat', 'download',
        '--files', band_name,
        scene_id
    ]
    
    result = subprocess.run(command, cwd=output_dir, text=True, capture_output=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print(f"Successfully downloaded {band_name} for scene {scene_id}.")
    else:
        print(f"Failed to download {band_name} for scene {scene_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from TIFF files.")
    parser.add_argument('mode', choices=['train', 'test'], help="Mode to run the script: 'train' or 'test'")
    args = parser.parse_args()
    mode = args.mode

    training_folder_root = '../38-Cloud_training' if mode == 'train' else '../38-Cloud_test'
    training_folders = ['train_nir'] if mode == 'train' else ['test_nir']

    lwir_folder = 'train_lwir'  if mode == 'train' else 'test_lwir'

    band_name = 'B10.TIF'
    lwir_folder_full_scene = 'train_lwir_full_scene' if mode == 'train' else 'test_lwir_full_scene'
    gt_folder = 'train_gt' if mode == 'train' else 'test_blue'

    gt_dir = os.path.join(training_folder_root, gt_folder)
    og_patch_size = 384

    all_sceneid = extract_unique_sceneids(training_folder_root, training_folders[0])

    for scene_id in all_sceneid:
        download_landsat_band(scene_id, band_name, os.path.join(training_folder_root, lwir_folder_full_scene))
        new_file = f'{scene_id}_{band_name}'
        new_file_dir = os.path.join(training_folder_root, lwir_folder_full_scene, scene_id)    
        new_file_path = os.path.join(new_file_dir, new_file)    
        os.makedirs(new_file_dir, exist_ok = True)
        # rotate_and_crop(new_file_path, new_file_path)
        img_dim = stitch_scene(training_folder_root, gt_folder, scene_id, og_patch_size, False)
        print(img_dim)
        fit_image(new_file_path, new_file_path, img_dim)
        output_dir = os.path.join(training_folder_root, lwir_folder)
        extract_patches(new_file_path, og_patch_size, output_dir, scene_id, gt_dir)

    # BlueFox has size 752x480.
    # GSD of landsat 8 visual bands is 30m. GSD of Bluefox on beavercube ~100m

    # Boson is LWIR  has size 640x512 or 320x256 with GSD of ~200m. Don't use NIR band 




    
    
    
    

    








