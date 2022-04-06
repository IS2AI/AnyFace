# import the necessary packages
from imutils import paths 
import numpy as np
import argparse 
import imutils
import cv2
import os

def extract_labels(image, points):
    H, W = image.shape[:2]
    
    # top-left and bottom-right coords
    # of bounding box
    xs = min(points[:, 0])
    ys = min(points[:, 1])
    xe = max(points[:, 0])
    ye = max(points[:, 1])
    
    # extend the bounding box
    # by adding a small margin
    mw = (xe - xs) * 0.1
    mh = (ye - ys) * 0.1
    
    xs = max(xs - mw, 0)
    ys = max(ys - mh * 2, 0)
    
    xe = min(xe + mw, W-1)
    ye = min(ye + mh, H-1)
    
    # convert to the the yolo format:
    # x center, y center, width, height 
    xc = (xs + xe) / 2
    yc = (ys + ye) / 2
    width = (xe - xs)
    height = (ye - ys)
    
    # coordinates of facial landmarks
    p1x = (points[0,0] + points[1,0]) / 2
    p1y = (points[0,1] + points[1,1]) / 2
    
    p2x = (points[2,0] + points[3,0]) / 2
    p2y = (points[2,1] + points[3,1]) / 2
    
    p3x = points[4,0]
    p3y = points[4,1]
    
    p4x = points[5,0]
    p4y = points[5,1]
    
    p5x = points[6,0]
    p5y = points[6,1]
    
    cv2.rectangle(image, (int(xs), int(ys)), (int(xe), int(ye)), (0, 255, 0), 2)
    cv2.circle(image, (int(p1x), int(p1y)), 3, (0, 0, 255), -1)
    cv2.circle(image, (int(p2x), int(p2y)), 3, (255, 0, 0), -1)
    cv2.circle(image, (int(p3x), int(p3y)), 3, (0, 255, 255), -1)
    cv2.circle(image, (int(p4x), int(p4y)), 3, (255, 0, 255), -1)
    cv2.circle(image, (int(p5x), int(p5y)), 3, (255, 255, 0), -1)
    
    return image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to the dataset")
args = vars(ap.parse_args())

# get path to the images and annotations
annotPaths = list(paths.list_files(args["dataset"], validExts="pts"))
annotPaths = sorted(annotPaths)

imagePaths = list(paths.list_files(args["dataset"], validExts="jpg"))
imagePaths = sorted(imagePaths)

print(len(annotPaths), len(imagePaths))

width = 2

# loop over the sorted images
for annotPath, imagePath in zip(annotPaths, imagePaths):
    image = cv2.imread(imagePath)
    points = np.loadtxt(annotPath, comments=("version:", "n_points:", "{", "}"))
    
    image = extract_labels(image, points)

    # show the image
    cv2.imshow("Image", imutils.resize(image, width=640))
    key = cv2.waitKey(0) & 0xFF

    # if the `q` key was pressed, 
    # break from the loop
    if key == ord("q"):
        break