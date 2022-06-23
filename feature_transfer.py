"""
This file is to identify the salient feature in one image, extract the salient feature
 and transplant it into least-salient region of a new image 
 the input and output are all in npy format"
"""
 
# import the necessary packages
import numpy as np
import argparse
import cv2
import os 
import re

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-r", "--radius", type = int, help = "radius of Gaussian blur; must be odd")
ap.add_argument("--saliency_folder", type=str, required=True, help='folder for the saliency maps of the source imgs' )
ap.add_argument("--img_folder", type=str, help='source img folder, whose salient features will be extracted and transplanted')
ap.add_argument("--adv_input", type=int, default=0, help='indicate whether the source img is adv img or not. This is only for labeling the output folder') 
ap.add_argument("--held_out_input_folder", type=str, required=True, help='folder for held-out images')
ap.add_argument("--held_out_saliency", type=str, required=True, help='folder for the saliency maps of the held-out images')
ap.add_argument("--noise_percentage", type=str, required=True, help='size of the patch size. This is only for labeling the output folder')
ap.add_argument('--dataset', type=str, required=True, help='dataset name')

ap.add_argument('--target', type=str, required=True, help='target label')
ap.add_argument('--patch_type', type=str, default='square', help="patch type: rectangle or square")
ap.add_argument('--random_seed' , type = int, default=100, help='random seed. Needed if you want to perform feature transfer for multiple times, so that each times different random hold-out images would be chosen')
ap.add_argument('--num_of_feature_trans', type=int, default=1, help='tag to differentiate folders when performing feature transfer multiple times')
ap.add_argument('--save_folder', type=str, default='.',  help="directory to save the resuluting images")


args = vars(ap.parse_args()) 


np.random.seed(args['random_seed'])



# held_out_input for attaching the salient feature from other images
if(not args['held_out_input_folder'] ==None):
    held_out_input_folder = []    
    for r, d, f in os.walk(args['held_out_input_folder']):
        for file in f: 
            if(".jpg" in file or '.png' in file):
                held_out_input_folder.append(os.path.join(r, file))


files = []
for r, d, f in os.walk(args['saliency_folder']):
    for file in f:
        #if( ".JPEG" in file ):
        files.append(os.path.join(r, file))



def get_coordinate_for_recetange(x, y, radius, x_max, y_max, patch_type):
    # get the start / end coordinate for extracting the salient features

    start = []
    end = [] # coordinate of the rectangle

    if(patch_type == "square"):
        start.append( x - radius )
        start.append( y - radius )
        end.append( x + radius )
        end.append( y + radius )
    elif(patch_type=="rectangle"):
        start.append( x - int(radius* 1.23) )
        start.append( y - int(radius*0.82) )
        end.append( x + int(radius*1.23) )
        end.append( y + int(radius*0.82) )


    if(start[0] < 0): 
        end[0] += abs(start[0]) 
        start[0] = 0
    if(start[1] < 0):
        end[1] += abs(start[1])
        start[1] = 0
    if(end[0] > x_max):
        start[0] -= (end[0] - x_max)
        end[0] = x_max
    if(end[1] > y_max):
        start[1] -= (end[1] - y_max)
        end[1] = y_max

    # start[y, x], end[y, x]
    return tuple(start), tuple(end)


for i in range(len(files)): 
 
    saliency_imgFile = files[i]  
    regexp1 = re.compile('\d+' + '|' + '-\d+' + "|" + "_\d+" )
    fileName = saliency_imgFile.replace(args["saliency_folder"], '')
    tmp = regexp1.findall( fileName ) 

    clean_label = tmp[1][1:]


    "Get the source file name based on the saliency file's name"
    source_imgFile = args["img_folder"].rstrip("/") + "/" + tmp[0] + tmp[1] + ".npy"
    #print(i , saliency_imgFile, source_imgFile, flush=True)
    #print()
    try:
        saliency_img = cv2.imread(saliency_imgFile)
        source_img = np.load(source_imgFile) 
    except:
        continue

    orig = saliency_img.copy()
    gray = cv2.cvtColor(saliency_img, cv2.COLOR_BGR2GRAY)
    # perform a naive attempt to find the (x, y) coordinates of
    # the area of the image with the largest intensity value
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray) 
 

    #"averaging to identify the high density region."
    #"outlier large point will be smoothed out after averaging"
    gray = cv2.blur(gray, (args["radius"], args["radius"]) )
    #gray = cv2.blur(gray, (args["radius"], args["radius"]) )

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)


    source_img = source_img[0]
    img_shape = source_img.shape 
    start, end = get_coordinate_for_recetange(maxLoc[0], maxLoc[1], args['radius'], img_shape[1], img_shape[2], args["patch_type"])

    if(args['dataset'] != 'vggface'):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = [0.489, 0.409, 0.372]
        std = [1, 1, 1]
         
    # read held_out_input
    rand_indx = np.random.randint(low=0, high= len(held_out_input_folder)-1 ) 


    "the held out input needs to be processed and transposed to match the source img (directly applicable for torch model inference)"
    held_out_input = cv2.imread( held_out_input_folder[rand_indx] )
    held_out_input = cv2.cvtColor(held_out_input, cv2.COLOR_BGR2RGB)
    held_out_input = np.asarray(held_out_input)
    held_out_input = held_out_input.astype(float)
    held_out_input /= 255


    if(held_out_input.shape[0] != 224):
        print("Warning: the input dimension is not 224")

    held_out_input = np.transpose(held_out_input, (2, 0, 1))

    # normalization 
    for i in range(3):
        held_out_input[i, :, :] -= mean[i]
        held_out_input[i, :, :] /= std[i]
 

    # transfer the salient feature to the least-salient region of the new image
    # first find the saliency map of the randomly chosen hold-out image
    selected_held_out_input = held_out_input_folder[rand_indx][:-4].replace(args['held_out_input_folder'], '').replace('/', '') 
    held_out_saliency = ''
    for r, d, f in os.walk(args['held_out_saliency']):
        for file in f: 
            if( selected_held_out_input in file ):
                held_out_saliency = file 
                break
    saliency_held_out_input = cv2.imread(args['held_out_saliency'] + "/" + held_out_saliency) 

    gray = cv2.cvtColor(saliency_held_out_input, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)
    gray = cv2.blur(gray, (args["radius"], args["radius"]) )
    gray = cv2.blur(gray, (args["radius"], args["radius"]) ) 

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # put the salient features from the source image to the least-salient regions in the hold-out image
    held_out_start, held_out_end = get_coordinate_for_recetange(minLoc[0], minLoc[1], args['radius'], img_shape[1], img_shape[2], args["patch_type"])
    held_out_input[ :, held_out_start[1]:held_out_end[1], held_out_start[0]:held_out_end[0] ] = \
                            source_img[ :, start[1]:end[1], start[0]:end[0] ]

    
    if(args['adv_input'] == 0 ):
        SAVE = "{}/{}_{}_{}_{}feature_transfer_org_comp_{}".format(args['save_folder'].rstrip("/"), args['dataset'], args['target'], args['patch_type'], args['num_of_feature_trans'], args['noise_percentage'].replace('.', ''))
    elif( args['adv_input'] == 1 ):
        SAVE = "{}/{}_{}_{}_{}feature_transfer_adv_comp_{}".format(args['save_folder'].rstrip("/"), args['dataset'], args['target'], args['patch_type'], args['num_of_feature_trans'], args['noise_percentage'].replace('.', '')) 

    if(not os.path.exists(SAVE)):
        os.mkdir(SAVE)

    held_out_input = held_out_input[np.newaxis, ...] 
    np.save( SAVE + "/" + tmp[0] + tmp[1] + ".npy",  held_out_input)


 








