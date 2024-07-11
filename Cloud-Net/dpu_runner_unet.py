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
from utils import get_input_image_names


GLOBAL_PATH = 'path to 38-cloud dataset'
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'Training')
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'Test')
PRED_FOLDER = os.path.join(GLOBAL_PATH, 'Predictions')


in_rows = 384
in_cols = 384
num_of_channels = 4
num_of_classes = 1
batch_sz = 10
max_bit = 65535  # maximum gray level in landsat 8 images
experiment_name = "Cloud-Net_trained_on_38-Cloud_training_patches"


# getting input images names

test_patches_csv_name = 'test_patches_38-cloud.csv'
df_test_img = pd.read_csv(os.path.join(TEST_FOLDER, test_patches_csv_name))
test_img, test_ids = get_input_image_names(df_test_img, TEST_FOLDER, if_train=False)
pred_dir = experiment_name + '_train_192_test_384'





"""
 obtain dpu subgrah
"""
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
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
    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1  # only one DPU kernel
    runner = vart.RunnerExt.create_runner (subgraphs[0], "run")
    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()

    input_ndim = tuple(input_tensor_buffers[0].get_tensor().dims)
    batch = input_ndim[0]
    print(f'batch_sz: {batch}')
    width = input_ndim[1]
    height = input_ndim[2]
    fixpos = input_tensor_buffers[0].get_tensor().get_attr("fix_point")

    # image = preprocess_one_image(argv[2], width, height, MEANS, SCALES, fixpos)
    input_data = np.asarray(input_tensor_buffers[0])
    input_data[0] = test_img[0]

    job_id = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
    runner.wait(job_id)

    pre_output_size = int(output_tensor_buffers[0].get_tensor().get_data_size() / batch)
    output_data = np.asarray(output_tensor_buffers[0])
    output_data = (output_data[:, :, 0]).astype(np.int32)
    tiff.imsave(os.path.join(PRED_FOLDER, pred_dir, str(test_ids[0])), output_data)
    del runner


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("please  model file and input file.")
    else:
        main(sys.argv)