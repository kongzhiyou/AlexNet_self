import os
import numpy as np
import cv2 as cv
import math
from cfg import config
import os

def data_press(file_dir=r'F:\data\dogs-vs-cats\train'):
    filename_list = []
    for root,f_dir,filename in os.walk(file_dir):
        filename_list = filename
    return filename_list

class ImageGenerator(object):
    def __init__(self):
        self.file_dir = r'F:\data\dogs-vs-cats\train'
        self.batch_size = 16
        self.data_set = data_press(self.file_dir)
        self.num_examples = len(self.data_set)
        self.num_epochs = 1000
        self.num_step_epoch = self.num_examples/self.batch_size
        self.all_step = self.num_epochs*self.num_step_epoch
        self.computed_batch = 0
        self.start_index = 0
        self.end_index = self.start_index+self.batch_size
        self.batch_img_set = []
        self.batch_label_set = []

    def one_hot(self,label):
        y_set_oh = np.zeros((len(label), 2))    #  其中的2为类别数目
        y_set_oh[np.arange(len(label)),label] = 1
        return y_set_oh

    def next_batch(self):
        self.batch_img_set = []
        self.batch_label_set = []
        if self.computed_batch==0 and self.start_index ==0:
            np.random.shuffle(self.data_set)
        if self.start_index+self.batch_size > self.num_examples:
            res_data_pre = self.data_set[self.start_index:self.num_examples]
            np.random.shuffle(self.data_set)
            res_num = self.start_index+self.batch_size-self.num_examples
            res_data_suf = self.data_set[:res_num]
            res_data_pre.extend(res_data_suf)
            label_list = []
            for file_name in self.data_set[self.start_index:self.end_index]:
                img = cv.imread(os.path.join(self.file_dir, file_name))
                img_arr = cv.resize(img, (256, 256))
                label = file_name.split('.')[0]
                if label == 'cat':
                    label_list.append(0)
                else:
                    label_list.append(1)
                self.batch_img_set.append(img_arr)
                print(file_name)
            self.batch_label_set = self.one_hot(label_list)
            self.start_index = self.end_index

        else:
            label_list = []
            for file_name in self.data_set[self.start_index:self.end_index]:
                img = cv.imread(os.path.join(self.file_dir,file_name))
                img_arr = cv.resize(img,(256,256))
                label = file_name.split('.')[0]
                if label == 'cat':
                    label_list.append(0)
                else:
                    label_list.append(1)
                self.batch_img_set.append(img_arr)
                print(file_name)
            self.batch_label_set = self.one_hot(label_list)
            self.start_index = self.end_index
            self.batch_img_set = self.reflect_image()
        return self.batch_img_set,self.batch_label_set

    def reflect_image(self):
        batch_img_arr = np.asarray(self.batch_img_set)
        img_list = []
        for img_arr in batch_img_arr:
            H, W = img_arr.shape[:2]
            x = np.random.randint(W - 227)
            y = np.random.randint(H - 227)
            img_arr = img_arr[y:y + 227, x:x + 227]
            img_arr = img_arr[:, ::-1]
            img_list.append(img_arr)
        return img_list

image_set = ImageGenerator()
image_set.next_batch(1)

#
# if __name__ == '__main__':
#     file_dir = r'F:\data\dogs-vs-cats\test\cat'
#     img_list = []
#     for img_name in os.listdir(file_dir):
#         img_arr = cv.imread(os.path.join(file_dir,img_name))
#         img_arr = cv.resize(img_arr, (256, 256))
#         H,W = img_arr.shape[:2]
#         print(H,W)
#         x = np.random.randint(W - 227)
#         y = np.random.randint(H - 227)
#         image = img_arr[y:y + 227, x:x + 227]
#         img_arr = img_arr[:,::-1]
#         img_list.append(img_arr)
#     img_list = np.stack(img_list)
#     cv.imshow('cat',img_list[0])
#     cv.waitKey(0)
#     cv.destroyAllWindows()


