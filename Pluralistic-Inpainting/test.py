"""
Change to the original code:
(1) read input in npy format
(2) output inpainted image in npy format (pixel in 0,1 but un-normalized)
(3) support drawing random mask based on a center coordinate
"""




from options import test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice


# python test.py --name imagenet_random --mask_type 4 --img_file 

if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)

    cnt = 0 
    for i, data in enumerate(dataset): 
        cnt += 1
        
        model.set_input(data)
        inpainted_img = model.test()
 
        print("{} input".format(cnt))

        
        # iteratively place mask on the imaage 
        for j in range(1, opt.iterative_mask ):
            model.set_input(inpainted_img)
            inpainted_img = model.test()

