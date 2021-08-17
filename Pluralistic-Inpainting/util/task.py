import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from random import randint
import numpy as np
import cv2
from PIL import Image
import random
from random import randint

###################################################################
# random mask generation
###################################################################
random.seed(10)


def dot_mask(img, coordinate= (112,112) , radius=51, percentage_of_pert=0.25, seed=1, is_square_patch=1): 
    "CAREFUL about the shape (in the code the channel is changed)"
 
    def get_coordinate_for_recetange(x, y, radius, x_max, y_max, is_square_patch):
        # get the start / end coordinate for the rectangle, centering at (x,y) with length of 2*radius

        start = []
        end = [] # coordinate of the rectangle

        if(is_square_patch):
            start.append( x - radius )
            start.append( y - radius )
            end.append( x + radius )
            end.append( y + radius )
        else:
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


        return tuple(start), tuple(end)

    

    def gencoordinates(size, x_min, x_max, y_min, y_max):
        seen = set()

        x_set = []
        y_set = []

 
        x, y = randint(x_min, x_max), randint(y_min, y_max)
        x_set.append(x)
        y_set.append(y) 

        cnt = 1
        while (not size==cnt ):
            seen.add((x, y)) 
            x, y = randint(x_min, x_max), randint(y_min, y_max)
            while (x, y) in seen:
                x, y = randint(x_min, x_max), randint(y_min, y_max)

            x_set.append(x)
            y_set.append(y) 
            cnt += 1 

        return x_set, y_set



    mask = np.ones(shape= img.shape )

    "NOTE: the coordiante returned is for RGB, y-axis downward, x-axis right-ward"
    start, end = get_coordinate_for_recetange(coordinate[0], coordinate[1], radius, img.shape[1]-1, img.shape[2]-1, is_square_patch)

    #print( coordinate ) 
    if(is_square_patch):
        len_for_random_inx = 2*radius #"length of the area to derive the random index for masking within the salient region"      
        pts_for_selection = int( (len_for_random_inx*len_for_random_inx) * percentage_of_pert )
    else:
        xlength = int(radius*2.46)
        ylength = int(radius*1.64)
        pts_for_selection = int( xlength * ylength * percentage_of_pert )

    if(is_square_patch):
        x_masked_index, y_masked_index = gencoordinates(size= pts_for_selection, x_min=start[0], x_max=start[0] + int(radius * 2) -1 , \
                                                        y_min= start[1], y_max=start[1] + int(radius * 2) -1)
    else: 
        x_max = int(float(radius) * 2.46) -1
        y_max = int(float(radius) * 1.64) -1



        x_masked_index, y_masked_index = gencoordinates(size= pts_for_selection, x_min=start[0], x_max=start[0] + x_max , \
                                                        y_min= start[1], y_max=start[1] + y_max )


    #x_masked_index = np.random.randint(low= start[0] , high= start[0] + radius * 2 -1, size=pts_for_selection )
    #y_masked_index = np.random.randint(low= start[1] , high= start[1] + radius * 2 -1, size=pts_for_selection )
 

    "Y axis first" 
    mask[ :, y_masked_index, x_masked_index] = 0



    mask = torch.from_numpy(mask)
    mask = mask.type(torch.FloatTensor)
    return mask

    
def random_regular_mask(img):
    """Generates a random regular hole"""
    mask = torch.ones_like(img)
    s = img.size()

    '''
    N_mask = random.randint(1, 5)
    limx = s[1] - s[1] / (N_mask + 1)
    limy = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        range_x = x + random.randint(int(s[1] / (N_mask + 7)), int(s[1] - x))
        range_y = y + random.randint(int(s[2] / (N_mask + 7)), int(s[2] - y))
        mask[:, int(x):int(range_x), int(y):int(range_y)] = 0
    '''

    # draw small masks
    mask_width = int(s[1]/2)
    mask_height = int(s[2]/2)

    N_mask = 5
    for _ in range(N_mask):

        mask_width_random = random.randint(0, mask_width)
        mask_height_random = random.randint(0, mask_height)

        left_corner_coordinate_x = random.randint(0, s[1]-mask_width_random)
        left_corner_coordinate_y = random.randint(0, s[2]- mask_height_random)

        mask[:, left_corner_coordinate_x : (left_corner_coordinate_x+mask_width), left_corner_coordinate_y : left_corner_coordinate_y+mask_height] = 0
 


    return mask


def center_mask(img):
    """Generates a center hole with 1/4*W and 1/4*H"""
    mask = torch.ones_like(img)
    size = img.size()
    x = int(size[1] / 4)
    y = int(size[2] / 4)
    range_x = int(size[1] * 3 / 4)
    range_y = int(size[2] * 3 / 4)
    mask[:, x:range_x, y:range_y] = 0

    return mask


def random_irregular_mask(img):
    """Generates a random irregular mask with lines, circles and elipses"""
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones_like(img)
    size = img.size()
    img = np.zeros((size[1], size[2], 1), np.uint8)

    # Set size scale
    max_width = 20
    if size[1] < 64 or size[2] < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    number = random.randint(16, 64)
    for _ in range(number):
        model = random.random()
        if model < 0.6:
            # Draw random lines
            x1, x2 = randint(1, size[1]), randint(1, size[1])
            y1, y2 = randint(1, size[2]), randint(1, size[2])
            thickness = randint(4, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        elif model > 0.6 and model < 0.8:
            # Draw random circles
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            radius = randint(4, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        elif model > 0.8:
            # Draw random ellipses
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            s1, s2 = randint(1, size[1]), randint(1, size[2])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(4, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    for j in range(size[0]):
        mask[j, :, :] = img_mask < 1

    return mask

###################################################################
# multi scale for image generation
###################################################################


def scale_img(img, size):
    scaled_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    return scaled_img


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = scale_img(img, size=[nh, nw])
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs

