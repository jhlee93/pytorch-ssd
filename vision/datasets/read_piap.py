import numpy
import pathlib
import cv2
import os

class PIAPDataset:
    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        is_test = False
        if is_test:
            image_sets_file = self.root / "piap/test.txt"
        else:
            image_sets_file = self.root / "piap/train.txt"
        self.ids = PIAPDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_path = self.root / "piap/char_obj.names"
        classes = []
        with open(label_file_path, "r") as f:
            lines = f.readlines()

        for i in lines:
            if "\n" in i:
                classes.append(i.split("\n")[0])
            else
                classes.append(i)

        self.class_names = tuple(classes)
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)


