import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os


class PIAPDataset:
    
    def __init__(self, root, transform=None, target_transform=None, is_test = False, keep_difficult=False, label_file=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform

        if is_test:
            image_sets_file = self.root / "piap/test.txt"
            print("is_test = TURE ===> ", image_sets_file)
        else:
            image_sets_file = self.root / "piap/train.txt"
            print("is_test = FALSE ===> ", image_sets_file)

        self.ids = PIAPDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        label_file_name = self.root / "piap/char_obj.names"
        classes = []
        with open(label_file_name, "r") as f:
            lines = f.readlines()

        for i in lines:
            if "\n" in i:
                classes.append(i.split("\n")[0])
            else:
                classes.append(i)
        classes.insert(0, "BACKGROUND")
        self.class_names = tuple(classes)
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"piap/img/{image_id}.txt"
        boxes = []
        labels = []
        is_difficult = []
        is_difficult_str = False

        # image rows, cols
        img = self._read_image(image_id)
        height, width = img.shape[:2]
        with open(annotation_file, "r") as f:
            while True:
                read_line = f.readline()
                line = read_line.split("\n")[0]
                line_info = line.split(" ")
                if line_info[0] == "": break
                
                labels.append(int(line_info[0]))
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
                
                # Relative Positions to Abs
                r_cx = float(line_info[1])
                r_cy = float(line_info[2])
                r_w = float(line_info[3])
                r_h = float(line_info[4])

                x1 = int((r_cx - (r_w / 2)) * width)
                y1 = int((r_cy - (r_h / 2)) * height)
                x2 = int((r_cx + (r_w / 2)) * width)
                y2 = int((r_cy + (r_h / 2)) * height)

                boxes.append([x1, y1, x2, y2])

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"piap/img/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
                










