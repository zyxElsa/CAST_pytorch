import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def makeVideo2(result_frames, output_filename, framerate):
    # result_frames: list of numpy arrays (frame)
    print('Stack transferred frames back to video...')
    height,width,dim = result_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(os.path.join(output_filename,'transfer.mp4'),fourcc,framerate,(width,height))
    for j in range(len(result_frames)):
        frame = result_frames[j] * 255.0
        frame = frame.astype('uint8')
        video.write(frame)
    video.release()
    print('Transferred video saved at %s.'%output_filename)


def denorm_img(img):
    min_val = np.amin(img)
    max_val = np.amax(img)
    interval = max_val - min_val
    img = (img - min_val) / interval
    return img

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    ######## use video transfer ########
    use_video = opt.use_video
    if use_video:
        #### read video, save each frame to ./temp_vid_frame/ ###
        # temp_vid_frame_path = './temp_vid_frame/'
        temp_vid_frame_path = './placeholder/testA/'
        store_path = './placeholder/testA/'
        if not os.path.exists(temp_vid_frame_path):
            os.makedirs(temp_vid_frame_path)
        temp_dir = glob.glob(temp_vid_frame_path+'*.jpg') + glob.glob(temp_vid_frame_path+'*.png')
        # make sure to remove previous video's frame
        for frames in temp_dir: 
            os.remove(frames)

        caps = cv2.VideoCapture(opt.video_name)
        framerate = int(caps.get(5))

        success, frame = caps.read()
        count = 0
        while success:
            cv2.imwrite(f"{store_path}{format(count,'05d')}.jpg", frame)     # save frame as JPEG file      
            success, frame = caps.read()
            count += 1
            print(f'total frame read is {count}', end='\r')
        print(f'total frame read is {count}')
        
        # opt.content_path = temp_vid_frame_path
        opt.num_test = count
    ########################################

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    video_frame = []
    for i, data in enumerate(dataset):
        if i == 0:
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
            # break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        if use_video:
            visual_np = visuals['fake_B'].cpu().detach().numpy()
            visual_np = cv2.cvtColor(np.transpose(np.squeeze(visual_np), (1,2,0)), cv2.COLOR_RGB2BGR)
            print(f'processing {i}-th frame', end='\r')
            visual_np = denorm_img(visual_np)
            video_frame.append(visual_np)
            # print(visuacls

        img_path = model.get_image_paths()     # get image paths
        if not use_video:
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, width=opt.display_winsize)
    if not use_video:
        webpage.save()  # save the HTML
    
    print(f'framerate is {framerate}')
    output_video_path = './output_transfer_video/'
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)
    makeVideo2(video_frame, output_video_path , framerate)
