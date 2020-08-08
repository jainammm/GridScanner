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
    model_output_val = np.array(model_output_val)[0]

    shape = data['shape']
    file_name = data['file_name'][0] # use one single file_name
    bboxes = data['bboxes'][file_name]
    
    vis_bbox(data_loader, image, np.array(data['grid_table'])[0], 
            np.array(data['gt_classes'])[0], model_output_val, file_name, 
            np.array(bboxes), shape)

    logits = model_output_val.reshape([-1, data_loader.num_classes])

    grid_table = np.array(data['grid_table'])[0] 
    gt_classes = np.array(data['gt_classes'])[0]
    word_ids = data['word_ids'][file_name]
    data_input_flat = grid_table.reshape([-1])

    c_threshold = 0.5

    for i in range(len(data_input_flat)):
            if max(logits[i]) > c_threshold:
                inf_id = np.argmax(logits[i])
                if inf_id:
                    try:
                        print('----------')
                        print(data_loader.classes[inf_id])
                        print(idTotext(word_ids[i], json_data))
                        print(max(logits[i]))
                    except:
                        pass

def idTotext(id, json_data):
    for text_box in json_data['text_boxes']:
        if text_box['id'] == id:
            return text_box['text']
    
    return None