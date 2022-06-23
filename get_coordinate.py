"""
This file is to get the cooridinate for locating the salient feature in the image and add it into the file's name.
"""

# import the necessary packages
import numpy as np
import argparse
import cv2
import os 
import re


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()  
ap.add_argument("--saliency_folder", type=str, help='saliency maps of the images')
ap.add_argument("--img_folder", type=str, help='image folder')
ap.add_argument("--radius", type=int, default=51, help="radius of Gaussian blur; must be odd")
  

args = vars(ap.parse_args()) 

np.random.seed(1)

 

new_folder = args['img_folder'].rstrip("/") + "_with_coordinate"
if(not os.path.exists(new_folder)):
    os.mkdir(new_folder)


img_files = []
for r, d, f in os.walk(args['img_folder']):
    for file in f: 
        img_files.append(os.path.join(r, file))



for i in range(len(img_files)):  
    imgFile = img_files[i]
    regexp = re.compile('\d+' + '|' + '-\d+' + "|" + "_\d+" )
    tmp = regexp.findall(imgFile.replace(args['img_folder'], '')) 
    saliency_imgFile = args['saliency_folder'].rstrip("/") + "/" + tmp[0] + tmp[1] + "_0049.png"
    
    #print(i , imgFile, saliency_imgFile)

    saliency_img = cv2.imread(saliency_imgFile)
    source_img = cv2.imread(imgFile)    
 
    gray = cv2.cvtColor(saliency_img, cv2.COLOR_BGR2GRAY)
    # perform a naive attempt to find the (x, y) coordinates of
    # the area of the image with the largest intensity value
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray) 

    # apply a Gaussian blur to the image then find the brightest
    # region 
    gray = cv2.blur(gray, (args["radius"], args["radius"]) ) 
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    newFileName = imgFile.replace(".npy", "=" + str(maxLoc[0])+"="+str(maxLoc[1]) + ".npy"  ).replace(args['img_folder'], new_folder)
    #print(imgFile, ' ===> ', newFileName)

    #print( "\t\tcp {} {}".format(imgFile, newFileName) ) 
    os.system( "cp {} {}".format(imgFile, newFileName) )
     
 







