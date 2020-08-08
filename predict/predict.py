from predict.clovaa import get_text_boxes
from predict.dataloader.dataloader import DataLoader
from predict.utils import vis_bbox
from config.model_config import model_params
from predict.grpc_client import get_model_output

import os
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np

def predict(file):
    content_type = file.content_type

    if content_type == 'application/pdf':
        pages = convert_from_bytes(file.file.read())
        image = pages[0]
    elif content_type.startswith('image'):
        image = Image.open(file.file)
    else:
        return 'unknown file type'

    json_data = get_text_boxes(image, file.filename)

    data_loader = DataLoader(json_data, model_params, update_dict=False, load_dictionary=True) # False to provide a path with only test data
    num_words = max(20000, data_loader.num_words)
    num_classes = data_loader.num_classes

    data = data_loader.fetch_validation_data()

    model_output_val = get_model_output(data)

    shape = data['shape']
    file_name = data['file_name'][0] # use one single file_name
    bboxes = data['bboxes'][file_name]
    
    vis_bbox(data_loader, image, np.array(data['grid_table'])[0], 
            np.array(data['gt_classes'])[0], np.array(model_output_val)[0], file_name, 
            np.array(bboxes), shape)