from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
#import config
from keras_retinanet import models
from imutils import paths
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image  
import PIL  
import matplotlib.image as mpimg
import os
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True, help='path to trained model')
ap.add_argument("-l", "--labels", required=True, help='path to class labels')
ap.add_argument("-i", "--input", required=True, help='path to input images directory')
ap.add_argument("-o", "--output", required=True, help='path to directory to store predictions')
ap.add_argument("-c", "--confidence", type=float, default=0.0, help="min probability to filter weak detections")

args = vars(ap.parse_args())

LABELS = open(args["labels"]).read().strip().split('\n')
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

model = models.load_model(args["model"], backbone_name='resnet50')
imagePaths = list(paths.list_images(args["input"]))

for (i, imagePath) in enumerate(imagePaths):

    print("[INFO] predicting on image {} of {}".format(i+1, imagePath))
    filename = (imagePath.split(os.path.sep)[-1]).split('.')[0]
    
    image = read_image_bgr(imagePath)
 
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    (image, scale) = resize_image(image)
        
    image = np.expand_dims(image, axis=0)

   
    
    (boxes, scores, labels) = model.predict_on_batch(image)
    boxes /= scale

    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):

        if score < args["confidence"]:
            continue
        
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
     
        caption = "{} {:.3f}".format(LABELS[label], score)
        draw_caption(draw, b, caption)
    draw = Image.fromarray(draw)
    draw.save("result_images/{}.jpg".format(filename))
    # plt.show()


