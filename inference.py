import numpy as np
import onnxruntime
import onnx
import numpy as np
import os
import copy
from preprocess import resize_image, normalize_image
from postprocess import postProcess, sorted_boxes, filter_tag_det_res, get_rotate_crop_image, resize_norm_img, LabelDecoder
from zookeeper import get_model

def load_onnx_model(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession(model_path)
    input_tensor = session.get_inputs()[0]
    return session, input_tensor

def detect_text(image, model_path):
    session, input_tensor = load_onnx_model(model_path)
    
    image, shape_list = resize_image(image)
    image = image.astype(np.float32)
        
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = normalize_image(image)

    input_dict = {}
    input_dict[input_tensor.name] = image
    
    return session.run(None, input_dict), shape_list

def predict(detection_model_path, recognition_model_path, image):
    detection_result, shape_list = detect_text(image, detection_model_path)

    prediction = {}
    prediction['maps'] = detection_result[0]

    post_result = postProcess(prediction, [shape_list])

    boxes = post_result[0]['points']
    boxes = filter_tag_det_res(boxes, image.shape)

    img_crop_list = []
    boxes = sorted_boxes(boxes)
    ori_img = np.asarray(image)

    for bno in range(len(boxes)):
        tmp_box = copy.deepcopy(boxes[bno])
        img_crop = get_rotate_crop_image(ori_img, np.float32(tmp_box))
        img_crop_list.append(img_crop)

    batch_num = 6
    ctcLabelDecode = LabelDecoder(use_space_char=True)

    img_num = len(img_crop_list)

    width_list = []
    for img in img_crop_list:
        width_list.append(img.shape[1] / float(img.shape[0]))

    indices = np.argsort(np.array(width_list))
    rec_res = [['', 0.0]] * img_num

    imgH = 48
    imgW = 320
    max_wh_ratio = imgW / imgH

    for beg_img_no in range(0, img_num, batch_num):
        end_img_no = min(img_num, beg_img_no + batch_num)
        
        norm_img_batch = []
        wh_ratio_list = []
        
        for ino in range(beg_img_no, end_img_no):
            h, w = img_crop_list[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
            wh_ratio_list.append(wh_ratio)

        for ino in range(beg_img_no, end_img_no):
            norm_img = resize_norm_img(img_crop_list[indices[ino]], max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)
        
        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()
        
        session, input_tensor = load_onnx_model(recognition_model_path)
        
        input_dict = {}
        input_dict[input_tensor.name] = norm_img_batch
        outputs = session.run(None, input_dict)
        prediction = outputs[0]
        
        rec_result = ctcLabelDecode(prediction, return_word_box=False, wh_ratio_list=wh_ratio_list, max_wh_ratio=max_wh_ratio)
        
        for rno in range(len(rec_result)):
            rec_res[indices[beg_img_no + rno]] = rec_result[rno]

    return rec_res


def ocr(image, model_name, model_version='latest'):
    if model_name == 'PP-OCRv3':
        files, path = get_model(model_name, model_version)
        assert 'det_model.onnx' in files, 'det_model.onnx not found'
        assert 'rec_model.onnx' in files, 'rec_model.onnx not found'

        detection_model_path = os.path.join(path, 'det_model.onnx')
        recognition_model_path = os.path.join(path, 'rec_model.onnx')

        return predict(detection_model_path, recognition_model_path, image)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))
