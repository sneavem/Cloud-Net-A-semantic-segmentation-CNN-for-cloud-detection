import os
from typing import List
import cv2
import numpy as np
import sys
import math
import xir
import vart
from generators import mybatch_generator_prediction
import tifffile as tiff
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# from utils import get_input_image_names


GLOBAL_PATH = '/home/root/38-cloud'
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_training')
TEST_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_test')
# PRED_FOLDER = os.path.join(GLOBAL_PATH, 'Predictions')
PRED_FOLDER = TRAIN_FOLDER # TODO: CHange this to the same folder as the TEST_FOLDER

in_rows = 192
in_cols = 192
# in_rows = 192
# in_cols = 192
num_of_channels = 4
num_of_classes = 1
batch_sz = 3
max_bit = 65535  # maximum gray level in landsat 8 images
experiment_name = "Cloud-Net_trained_on_38-Cloud_training_patches"


# getting input images names
# test_patches_csv_name = 'test_patches_38-Cloud.csv'
# train_patches_csv_name = 'training_patches_38-Cloud.csv'
train_patches_csv_name = 'training_patches_38-cloud_nonempty.csv'

def get_input_image_names(list_names, directory_name, if_train=True):
    list_img = []
    list_msk = []
    list_test_ids = []

    for filenames in tqdm(list_names['name'], miniters=1000):
        nred = 'red_' + filenames
        nblue = 'blue_' + filenames
        ngreen = 'green_' + filenames
        nnir = 'nir_' + filenames

        if if_train:
            dir_type_name = "train"
            fl_img = []
            nmask = 'gt_' + filenames
            fl_msk = os.path.join(directory_name, 'train_gt', f'{nmask}.TIF')
            list_msk.append(fl_msk)
        else:
            dir_type_name = "test"
            fl_img = []
            fl_id = f'{filenames}.TIF'
            list_test_ids.append(fl_id)

        fl_img_red = os.path.join(directory_name, f'{dir_type_name}_red', f'{nred}.TIF')
        fl_img_green = os.path.join(directory_name, f'{dir_type_name}_green', f'{ngreen}.TIF')
        fl_img_blue = os.path.join(directory_name, f'{dir_type_name}_blue', f'{nblue}.TIF')
        fl_img_nir = os.path.join(directory_name, f'{dir_type_name}_nir', f'{nnir}.TIF')

        fl_img.extend([fl_img_red, fl_img_green, fl_img_blue, fl_img_nir])
        list_img.append(fl_img)

    if if_train:
        return list_img, list_msk
    else:
        return list_img, list_test_ids

def Sigmoid(xx):
    x = np.asarray( xx, dtype="float32")
    return 1 / (1 + np.exp(-x))

def fix2float(fix_point, value):
    return value.astype(np.float32) * np.exp2(fix_point, dtype=np.float32)

# The main_test script uses the following line to get the test images and their ids
#TODO: change back to test_patches_csv_name for test images
# TODO also change back to TEST_FOLDER 
df_test_img = pd.read_csv(os.path.join(TRAIN_FOLDER, train_patches_csv_name))
# Switched if_train to True
test_img, test_ids = get_input_image_names(df_test_img, TRAIN_FOLDER, if_train=True)

pred_dir = experiment_name + '_train_192_test_384'
imgs_mask_test = mybatch_generator_prediction(test_img, in_rows, in_cols, batch_sz, max_bit)
imgs_mask_test = list(next(imgs_mask_test))

"""
 obtain dpu subgrah
"""
def get_child_subgraph_dpu(graph):
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def main(argv):
    # Getting test image
    test_img = np.stack(imgs_mask_test, axis=0)
    print(f'test_img shape: {test_img.shape}')

    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)
    # I'm not sure if this assert condition is essential 
    # assert len(subgraphs) == 1  # only one DPU kernel
    runner = vart.RunnerExt.create_runner (subgraphs[0], "run")
    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()

    input_ndim = tuple(input_tensor_buffers[0].get_tensor().dims)
    print(f'output_shape: {output_tensor_buffers[0].get_tensor().dims}')
    print(f'input_ndims: {input_ndim}')
    batch = input_ndim[0]
    print(f'batch_sz: {batch}')
    width = input_ndim[1]
    print(f'width: {width}')
    height = input_ndim[2]
    print(f'height: {height}')
    fixpos = input_tensor_buffers[0].get_tensor().get_attr("fix_point")
    output_fixpos = output_tensor_buffers[0].get_tensor().get_attr("fix_point")
    print(f'fixpos: {fixpos}')
    print(f' output_fixpos: {output_fixpos}')

    # image = preprocess_one_image(argv[2], width, height, MEANS, SCALES, fixpos)
    # input_data = np.asarray(input_tensor_buffers[0])
    # print(f'input_data shape: {input_data.shape}')
    # print(f'test_img shape: {test_img.shape}')
    # print(f'test_img: {test_img}')
    print(f'test_img shape: {test_img.shape}')
    input_data = (np.copy(test_img) * 2**fixpos).astype(np.int8)
    print(f'input_data[0]: {input_data[0]}')
    print(f'test_img: {test_img}')
    print(f'all zeros test_img: {np.all(test_img == 0)}')
    print(f'all zeros input_data[0]: {np.all(input_data[0] == 0)}')
    print(f'input_tensor_buffers: {np.asarray(input_tensor_buffers[0])}')

    job_id = runner.execute_async([input_data], output_tensor_buffers)
    runner.wait(job_id)

    pre_output_size = int(output_tensor_buffers[0].get_tensor().get_data_size() / batch)
    
    print(f'pre_output_size: {pre_output_size}')
    print(f'output_tensor buffers: {output_tensor_buffers}')
    output_data = np.asarray(output_tensor_buffers[0])
    print(f'output_data raw: {output_data}')
    output_data = output_data.astype(np.float32) * 2**(-output_fixpos)
    print(f'output_data after fix_float: {output_data}')
    output_data = Sigmoid(output_data)
    print(f'output_data after sigmoid: {output_data}')
    print(f'first_output: {output_data.shape}')
    output_data = (output_data[0, :, :, 0]).astype(np.float32)
    # output_data = np.squeeze(output_data, axis=0)
    print(f'output_data shape: {output_data.shape}')
    print(f'all zeros: {np.all(output_data == 0)}')
    print(f'output_data: {output_data}')
    # Testing with only one image and label  firsnt

    pred_dir_path = os.path.join(PRED_FOLDER, pred_dir)
    os.makedirs(pred_dir_path, exist_ok=True)
    filename = os.path.basename(str(test_ids[0]))
    
    tiff.imsave(os.path.join(PRED_FOLDER, pred_dir, filename), output_data)
    print(f'saving at {os.path.join(PRED_FOLDER, pred_dir, filename)}')
    # print(f'test_ids: {test_ids[0]}')
    del runner


if __name__ == "__main__":

    #if len(sys.argv) != 3:
    if len(sys.argv) != 2:
        print("please  model file and input file.")
    else:
        main(sys.argv)
