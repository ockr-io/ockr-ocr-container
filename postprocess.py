import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import math
import re


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]
    return rect

def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points

def filter_tag_det_res(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box in dt_boxes:
        if type(box) is list:
            box = np.array(box)
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new)
    return dt_boxes

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

def unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def boxes_from_bitmap(pred, _bitmap, dest_width, dest_height):
    max_candidates=1000
    min_size = 3
    box_thresh=0.6
    unclip_ratio=1.5
    bitmap = _bitmap
    height, width = bitmap.shape

    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 3:
        img, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]

    num_contours = min(len(contours), max_candidates)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        points, sside = get_mini_boxes(contour)
        if sside < min_size:
            continue
        points = np.array(points)
        score = box_score_fast(pred, points.reshape(-1, 2))
        
        if box_thresh > score:
            continue

        box = unclip(points, unclip_ratio).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)
        if sside < min_size + 2:
            continue
        box = np.array(box)

        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.astype("int32"))
        scores.append(score)
    return np.array(boxes, dtype="int32"), scores

def postProcess(outs_dict, shape_list, thresh=0.3):
    pred = outs_dict['maps']
    pred = pred[:, 0, :, :]
    segmentation = pred > thresh

    boxes_batch = []
    for batch_index in range(pred.shape[0]):
        src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
        mask = segmentation[batch_index]
        boxes, scores = boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
        boxes_batch.append({'points': boxes})
    return boxes_batch

def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def resize_norm_img(img, max_wh_ratio):
    imgC, imgH, imgW = [3, 48, 320]
    assert imgC == img.shape[2]
    imgW = int((imgH * max_wh_ratio))
    
    w = 1024
    if w is not None and w > 0:
        imgW = w

    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

class LabelDecoder():
    def __init__(self, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        with open('en_dict.txt', "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        if use_space_char:
            self.character_str.append(" ")
        dict_character = list(self.character_str)
        if 'arabic' in 'en_dict.txt':
            self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def get_word_info(self, text, selection):
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection==True)[0]

        for c_i, char in enumerate(text):
            if '\u4e00' <= char <= '\u9fff':
                c_state = 'cn'
            elif bool(re.search('[a-zA-Z0-9]', char)):
                c_state = 'en&num'
            else:
                c_state = 'splitter'
            
            if char == '.' and state == 'en&num' and c_i + 1 < len(text) and bool(re.search('[0-9]', text[c_i+1])): # grouping floting number
                c_state = 'en&num'
            if char == '-' and state == "en&num": # grouping word with '-', such as 'state-of-the-art'
                c_state = 'en&num'
            
            if state == None:
                state = c_state

            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            if state != "splitter":
                word_content.append(char)
                word_col_content.append(valid_col[c_i])

        if len(word_content) != 0: 
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False, return_word_box=False):
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
    
            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]
    
            text = ''.join(char_list)
    
            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)
            
            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(text, selection)
                result_list.append((text, np.mean(conf_list).tolist(), [len(text_index[batch_idx]), word_list, word_col_list, state_list]))
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, return_word_box=False, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
    
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True, return_word_box=False)
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs['wh_ratio_list'][rec_idx]
                max_wh_ratio = kwargs['max_wh_ratio']
                rec[2][0] = rec[2][0]*(wh_ratio/max_wh_ratio)
        return text

