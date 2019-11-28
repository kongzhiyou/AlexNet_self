import os
import numpy as np
import cv2 as cv
import math
from cfg import config
import os

def deformation_aug(self):
    images_path = config.IMAGE_PATH
    class_list = os.listdir(images_path)
    for cls in class_list:
        cls_img = os.listdir(os.path.join(images_path, cls))
        for img in cls_img:
            img_path = images_path + '\\' + os.path.join(cls, img)
            img_arr = np.array(cv.imread(img_path))
            arr_mean = np.mean(img_arr)
            img_arr = img_arr - arr_mean
            h, w, c = img_arr.shape
            if h > w and h > 256:
                img_re = cv.resize(img_arr, (256, h), interpolation=cv.INTER_CUBIC)
                re_h, re_w = img_re.shape[:2]
                flt = math.floor((re_h - 256) / 2)
                wrap = img_re[flt:(256 + flt), 0:re_w]
            elif w > h and w > 256:
                img_re = cv.resize(img_arr, (w, 256), interpolation=cv.INTER_CUBIC)
                re_h, re_w = img_re.shape[:2]
                flt = math.floor((re_w - 256) / 2)
                wrap = img_re[0:re_h, flt:(256 + flt)]
            elif math.fabs(w - h) < 30 and w < 256 and h < 256:
                img_re = cv.resize(img_arr, (256, 256), interpolation=cv.INTER_CUBIC)
                wrap = img_re
            else:
                mean = np.mean(img_arr)
                img_1 = img_arr[:, :, 0]
                img_2 = img_arr[:, :, 1]
                img_3 = img_arr[:, :, 2]
                np.pad(img_arr, ((16, 16), (16, 16), (16, 16)), 'constant', constant_values=(mean, mean))
                #             np.pad(img_2,((16,16),(16,16)),'constant',constant_values=(mean,mean))
                #             np.pad(img_3,((16,16),(16,16)),'constant',constant_values=(mean,mean))
                wrap = cv.resize(img_arr, (256, 256), interpolation=cv.INTER_CUBIC)
            cv.imwrite(img_path, wrap)


