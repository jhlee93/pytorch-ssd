import torch
from vision.datasets.piap_dataset import PIAPDataset
import argparse
import numpy as np
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.utils.misc import str2bool, Timer
import os
import cv2

parser = argparse.ArgumentParser(description="SSD Mobilenet test")
parser.add_argument('--use_cuda', type=str2bool, default=True)
parser.add_argument('--test_folder', type=str, default="/home/cvserver3/project/JH/Dataset/piap/Final_Testset")
parser.add_argument('--nms_method', type=str, default="hard")
parser.add_argument('--mb2_width_mult', type=float, default=1.0)
parser.add_argument('--trained_model', type=str, default="./models/mb2-ssd-lite-Epoch-5-Loss-0.9502851062069381.pth")
parser.add_argument('--prob_thresh', type=float, default=0.5)
args=parser.parse_args()



# x1, y1, x2, y2
def sort_boxes(boxes):
    y_list = []
    for i in boxes:
        y_list.append(i[1])

    y_list = sorted(y_list)
    for i in range(len(y_list)):
        boxes[i] = boxes[y_list.index(y_list[i])]

    for i in range(len(boxes)-1):
        for j in range(len(boxes)-1):
            if int(boxes[j][1] - boxes[j+1][1]) <  int((boxes[j][3] - boxes[j][1])/2):
                if boxes[j][0] > boxes[j+1][0]:
                    tmp = boxes[i]
                    boxes[j] = boxes[j+1]
                    boxes[j+1] = tmp



    return boxes


def main():
    timer = Timer()
    DEVICE = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() and args.use_cuda else "cpu")
   

    print("=== Use ", DEVICE)
    classes=[] 
    with open("./models/char_obj_EN.names", "r") as f:
        lines = f.readlines()

    for i in lines:
        if "\n" in i:
            classes.append(i.split("\n")[0])
        else:
            classes.append(i)
    class_names = classes
    print("label size is ", len(class_names))
    net = create_mobilenetv2_ssd_lite(len(class_names),width_mult=args.mb2_width_mult, is_test=True)
    
    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print(f'{timer.end("Load Model")} sec to load the model.')
    
    predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)

    test_list = [x for x in os.listdir(args.test_folder) if x.split(".")[1] == "jpg"]
    
    total_time = 0.0
    for i in test_list:
        # Read Images
        img_path = args.test_folder + "/" + i
        image = cv2.imread(img_path)
        cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(cvt_image)
        end_time = timer.end("Predict")
        print("Prediction: {:4f} sec.".format(end_time))
        total_time += end_time

        # to numpy
        bb = boxes.numpy()
        lb = labels.numpy()
        pb = probs.numpy()
        save_name = ""
        
        # resize
        s_factor = 4
        rows, cols = image.shape[:2]
        image = cv2.resize(image, (cols*s_factor, rows*s_factor))

        # score > 0.5
        for b in range(pb.size):
            if pb[b] > args.prob_thresh:
                cv2.rectangle(image, (int(bb[b][0]*s_factor), int(bb[b][1]*s_factor)),
                                        (int(bb[b][2]*s_factor), int(bb[b][3]*s_factor)),
                                         (0,255,0), 2)
                cv2.putText(image, class_names[int(lb[b])+1],
                                (int(bb[b][0]*s_factor), int(bb[b][1]*s_factor)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0,0,255),
                                2)

                save_name += class_names[int(lb[b])+1]

        cv2.imwrite("./output/Final_Testset/" + save_name + ".jpg", image)

        print("Input Image : {}  ====> Predict : {}".format(i, save_name))

    print("Avg Time is {:4f} sec.".format(total_time / len(test_list)))


if __name__ == '__main__':
    main()
