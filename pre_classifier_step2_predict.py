# import libraries
import os
import shutil
import time
import argparse
import logging
import gc
from gluoncv.utils.filesystem import try_import_decord
import glob 
import cv2

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms

from gluoncv.data import Kinetics400Attr, UCF101Attr, SomethingSomethingV2Attr, HMDB51Attr, VideoClsCustom
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs
#mx.__version__

def get_file_names_from_dir(dir_path, ext_str):
    return glob.glob(dir_path + ext_str)

# pre-set classifier parameters
class Args: 
    interval = 30
    data_dir = '/Users/carechen/poc2/classifier_clip/'
    save_dir = data_dir + 'predictions/'
    logging_file = 'predictions.log'
    save_logits = False
    save_preds = False
    data_list = data_dir + 'mini_clips_' + str(interval) + 's.txt'
    need_root = False
    
    num_segments = 1
    new_length = 32
    new_step = 20
    ten_crop = False 
    three_crop = False
    num_crop = 1
    
    use_decord = True
    use_pretrained = True
    video_loader = True 

    input_size = 224
    new_width = 340
    new_height = 256

    data_aug = 'v1'    
    log_interval = 10
    resume_params = ''
 
    gpu_id = -1
    dtype = 'float32'

    #slowfast_8x8_resnet101_kinetics400, fbde1a7c
    #i3d_nl10_resnet101_v1_kinetics400, 59186c31
    #i3d_resnet50_v1_hmdb51, 2ec6bf01
    #i3d_resnet50_v1_ucf101
    #i3d_slow_resnet101_f16s4_kinetics700, 299b1d9d
    model = 'i3d_nl10_resnet101_v1_kinetics400'
    num_classes = 400 
    hashtag = '59186c31' 
    mode = 'hybrid' 
    
    slowfast = False
    slow_temporal_stride = 8
    fast_temporal_stride = 2

def read_data(opt, video_name, transform, video_utils):
    print(video_name)
    decord = try_import_decord()
    decord_vr = decord.VideoReader(video_name, width=opt.new_width, height=opt.new_height)
    duration = len(decord_vr)

    opt.skip_length = opt.new_length * opt.new_step
    segment_indices, skip_offsets = video_utils._sample_test_indices(duration)

    if opt.video_loader:
        if opt.slowfast:
            clip_input = video_utils._video_TSN_decord_slowfast_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
        else:
            clip_input = video_utils._video_TSN_decord_batch_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
    else:
        raise RuntimeError('We only support video-based inference.')

    clip_input = transform(clip_input)

    if opt.slowfast:
        sparse_sampels = len(clip_input) // (opt.num_segments * opt.num_crop)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, opt.input_size, opt.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    else:
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (opt.new_length, 3, opt.input_size, opt.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    if opt.new_length == 1:
        clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case

    return nd.array(clip_input)

def main():
    opt = Args()

    makedirs(opt.save_dir)

    filehandler = logging.FileHandler(os.path.join(opt.save_dir, opt.logging_file))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)

    gc.set_threshold(100, 5, 5)

    # set env
    if opt.gpu_id == -1:
        context = mx.cpu()
    else:
        gpu_id = opt.gpu_id
        context = mx.gpu(gpu_id)

    # get data preprocess
    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
    if opt.ten_crop:
        transform_test = transforms.Compose([
            video.VideoTenCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 10
    elif opt.three_crop:
        transform_test = transforms.Compose([
            video.VideoThreeCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 3
    else:
        transform_test = video.VideoGroupValTransform(size=opt.input_size, mean=image_norm_mean, std=image_norm_std)
        opt.num_crop = 1

    # get model
    if opt.use_pretrained and len(opt.hashtag) > 0:
        opt.use_pretrained = opt.hashtag
    classes = opt.num_classes
    model_name = opt.model
    net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained,
                    num_segments=opt.num_segments, num_crop=opt.num_crop)
    net.cast(opt.dtype)
    net.collect_params().reset_ctx(context)
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    if opt.resume_params != '' and not opt.use_pretrained:
        net.load_parameters(opt.resume_params, ctx=context)
        logger.info('Pre-trained model %s is successfully loaded.' % (opt.resume_params))
    else:
        logger.info('Pre-trained model is successfully loaded from the model zoo.')
    logger.info("Successfully built model {}".format(model_name))

    # get classes list, if we are using a pretrained network from the model_zoo
    classes = None
    if opt.use_pretrained:
        if "kinetics400" in model_name:
            classes = Kinetics400Attr().classes
        elif "ucf101" in model_name:
            classes = UCF101Attr().classes
        elif "hmdb51" in model_name:
            classes = HMDB51Attr().classes
        elif "sthsth" in model_name:
            classes = SomethingSomethingV2Attr().classes

    # get data
    anno_file = opt.data_list
    f = open(anno_file, 'r')
    data_list = f.readlines()
    logger.info('Load %d video samples.' % len(data_list))

    # build a pseudo dataset instance to use its children class methods
    video_utils = VideoClsCustom(root=opt.data_dir,
                                 setting=opt.data_list,
                                 num_segments=opt.num_segments,
                                 num_crop=opt.num_crop,
                                 new_length=opt.new_length,
                                 new_step=opt.new_step,
                                 new_width=opt.new_width,
                                 new_height=opt.new_height,
                                 video_loader=opt.video_loader,
                                 use_decord=opt.use_decord,
                                 slowfast=opt.slowfast,
                                 slow_temporal_stride=opt.slow_temporal_stride,
                                 fast_temporal_stride=opt.fast_temporal_stride,
                                 data_aug=opt.data_aug,
                                 lazy_init=True)

    start_time = time.time()
    for vid, vline in enumerate(data_list):
        video_path = vline.split()[0]
        video_name = video_path.split('/')[-1]
        if opt.need_root:
            video_path = os.path.join(opt.data_dir, video_path)
        video_data = read_data(opt, video_path, transform_test, video_utils)
        video_input = video_data.as_in_context(context)        
        
        pred = net(video_input.astype(opt.dtype, copy=False))
        if opt.save_logits:
            logits_file = '%s_%s_logits.npy' % (model_name, video_name)
            np.save(os.path.join(opt.save_dir, logits_file), pred.asnumpy())
        pred_label = np.argmax(pred.asnumpy())
        if opt.save_preds:
            preds_file = '%s_%s_preds.npy' % (model_name, video_name)
            np.save(os.path.join(opt.save_dir, preds_file), pred)
                
        # Report topK classes
        if classes:
            topK = 3
            ind = nd.topk(pred, k=topK)[0].astype('int')
        logger.info('The input %s is classified to be' % video_name)
        for i in range(topK):
            child_class = classes[ind[i].asscalar()] 
            logger.info('\t[%s], with probability %.3f.'%
                      (child_class, nd.softmax(pred)[0][ind[i]].asscalar()))
                            
    end_time = time.time()
    logger.info('Total inference time is %4.2f minutes' % ((end_time - start_time) / 60))


if __name__ == '__main__':
    main()



