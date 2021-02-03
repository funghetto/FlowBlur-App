import torch
import cv2

import sys

import argparse
import os

import glob
import numpy as np

from PIL import Image
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

from MotionBlur import BlurIt


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    #print(img.shape)
    #print(img[None].shape)
    return img[None].to(DEVICE)


def viz(img, blur, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    blur = blur[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, blur, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.imwrite("image.png", img_flo[:, :, [2,1,0]])
    cv2.waitKey()

def LoadModel(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    return model

def RunImage(img1, img2):
    padder = InputPadder(img1.shape)
    image1, image2 = padder.pad(img1, img2)
    
    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        
        #print(flow_up.shape)
        #flow = model(image1, image2, iters=20, test_mode=False)[0]
        blurImage, normalizedFlow = MotionBlur(image1, flow_up)
    return image1, blurImage, flow_up, normalizedFlow
    viz(image1, withBlur, flow_up)

def MotionBlur(img, flow, tresh, force, strength, smooth, intepolation):
    #blurInput = np.transpose(cur_output_rectified[flen,:,:,:].numpy(), (2, 1, 0))
    #blurFlow = np.transpose(onlyAll.cpu().numpy(), (2, 1, 0))

    breetingRoom = tresh
    onlyMax = torch.clamp(flow[0,:,:,:] - breetingRoom, min=0, max=100)
    onlyMin = torch.clamp(flow[0,:,:,:] + breetingRoom, min=-100, max=0)

    onlyAll = (onlyMax + onlyMin)

    im = np.transpose( img[0].cpu().numpy(),(2, 1, 0))
    onlyAll = np.transpose(onlyAll.cpu().numpy(), (2, 1, 0))


    #REMOVE BLUR FROM HERE
    # SET BLUR INSIDE OF BlurIT
    # BLUR THE BLURED PIXELS AND ADD TO THE REGULAR IMAGE.

    
    #if smooth > 0: 
        #onlyAll = cv2.blur(onlyAll,(smooth,smooth))
        #s = smooth + (smooth % 2 - 1)
        #onlyAll = cv2.GaussianBlur(onlyAll,(s, s),0)

    blurred = BlurIt(im, onlyAll, force / 10, intepolation, strength)

    if smooth > 0: 
        #onlyAll = cv2.blur(onlyAll,(smooth,smooth))
        s = smooth + (smooth % 2 - 1)
        blurred = cv2.GaussianBlur(blurred,(s, s),0)


    im[:, :, 0] = (im[:, :, 0] + blurred[:, :, 0]) / (blurred[:,:,3] + 1)
    im[:, :, 1] = (im[:, :, 1] + blurred[:, :, 1]) / (blurred[:,:,3] + 1)
    im[:, :, 2] = (im[:, :, 2] + blurred[:, :, 2]) / (blurred[:,:,3] + 1)
    
    return (
        torch.from_numpy(np.transpose(im, (2, 1, 0))).unsqueeze(0),
        torch.from_numpy(np.transpose(onlyAll, (2, 1, 0)))
    )

def RunVideo(video_path):
    print("Running video")

def demoImg(args):
    image1 = load_image(args.i1)
    image2 = load_image(args.i2)
    RunImage(image1, image2)

def load_model(args):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    return model

def run_pair(model, im1, im2, tresh, force, strength, smooth, intepolation):
    image1 = load_image(im1)
    image2 = load_image(im2)

    blur, flo1, flo2 = run_pair_tensor(model, image1, image2, tresh, force, strength, smooth, intepolation)

    cv2.imwrite("blur.png", blur[:, :, [2,1,0]])
    cv2.imwrite("flow.png", flo1[:, :, [2,1,0]])
    cv2.imwrite("nflow.png", flo2[:, :, [2,1,0]])


def run_pair_tensor(model, image1, image2, tresh, force, strength, smooth, intepolation):
    with torch.no_grad():
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

        image1 = padder.unpad(image1)
        flow_up = padder.unpad(flow_up)

        blurImage, normalizedFlow = MotionBlur(image1, flow_up, tresh, force, strength, smooth, intepolation)
        #return image1, blurImage, flow_up, normalizedFlow

    #img = image1[0].permute(1,2,0).cpu().numpy()
    blur = blurImage[0].permute(1,2,0).cpu().numpy()

    flo1 = flow_up[0].permute(1,2,0).cpu().numpy()
    flo2 = normalizedFlow.permute(1,2,0).cpu().numpy()

    flo1 = flow_viz.flow_to_image(flo1)
    flo2 = flow_viz.flow_to_image(flo2)

    del image1, image2, padder, flow_low, flow_up, blurImage, normalizedFlow

    return blur, flo1, flo2

    #flo = flow_viz.flow_to_image(flo)
    #print(blur)
    #cv2.imwrite("blur.png", blur[:, :, [2,1,0]])
    #cv2.imwrite("flow.png", flo1[:, :, [2,1,0]])
    #cv2.imwrite("nflow.png", flo2[:, :, [2,1,0]])
    #return image1, flow_up 

    

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)


if __name__ == "__main__":
    print("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_false', help='use small model')
    parser.add_argument('--mixed_precision', action='store_false', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_false', help='use efficent correlation implementation')
    parser.add_argument('--i1', help="First Image")
    parser.add_argument('--i2', help="Second Image")
    args = parser.parse_args()

    LoadModel(args)
    demoImg(args)