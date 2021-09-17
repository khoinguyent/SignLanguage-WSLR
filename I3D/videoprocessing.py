import json
import math
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl
import videotransforms

class VideoProcessing():
    @staticmethod
    def raw_numpy_array(video_file, nframes=None):
        """
        Loads a video from the given file. Will set the number
        of frames to `nframes` if this parameter is not `None`.

        Returns:
        - (width, height, arr): The width and height of the video,
        and a numpy array with the parsed contents of the video.
        """

        # Read video
        cap = cv2.VideoCapture(video_file)

        # Get properties of the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Min allowed height or width (whatever is smaller), in pixels
        min_dimension = 256.0

        # Determine scaling factors of width and height
        assert min(w, h) > 0, 'Cannot resize {} with W={}, H={}'.format(video_file, w, h)
        scale = min_dimension / min(w, h)
        w = int(w * scale)
        h = int(h * scale)

        buf = np.zeros((1, frame_count, h, w, 3), np.dtype('float32'))
        fc, flag = 0, True

        while fc < frame_count and flag:
            flag, image = cap.read()

            if flag:
                image = cv2.resize(image, (w, h))
                buf[0, fc] = image

            fc += 1

        cap.release()

        if nframes is not None:
            if nframes < frame_count:
                fc = frame_count
                t1, t2 = int(fc/2) - int(nframes/2), int(fc/2) + int(nframes/2)
                buf = buf[:, t1:t2, :, :, :]
            elif nframes > frame_count:
                buf = np.resize(buf, (1, nframes, h, w, 3))

            return w, h, buf 

    @staticmethod
    def crop_video(numpy_video, size, desired_size):
        """
        Crop a video of the given size (WIDTH, HEIGHT) into a square of `desired_size`.
        The video is represented as a numpy array. This func is for internal usage.
        """

        w, h = size
        h1, h2 = int(h/2) - int(desired_size/2), int(h/2) + int(desired_size/2)
        w1, w2 = int(w/2) - int(desired_size/2), int(w/2) + int(desired_size/2)
        return numpy_video[:, :, h1:h2, w1:w2, :]

    @staticmethod
    def video_to_tensor(pic):
        """Convert a ``numpy.ndarray`` to tensor.
        Converts a numpy.ndarray (T x H x W x C)
        to a torch.FloatTensor of shape (C x T x H x W)
        
        Args:
            pic (numpy.ndarray): Video to be converted to tensor.
        Returns:
            Tensor: Converted video.
        """
        return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

    @staticmethod
    def load_rgb_frames(image_dir, vid, start, num):
        frames = []
        for i in range(start, start + num):
            try:
                img = cv2.imread(os.path.join(image_dir, vid, "image_" + str(i).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
            except:
                print(os.path.join(image_dir, vid, str(i).zfill(6) + '.jpg'))
            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            img = (img / 255.) * 2 - 1
            frames.append(img)
        return np.asarray(frames, dtype=np.float32)

    @staticmethod
    def load_rgb_frames_from_video(vid_root, vid, start, num, resize=(256, 256)):
        video_path = ""
        if('.mp4' not in vid):
            video_path = os.path.join(vid_root, vid + '.mp4')
        else:
            video_path = os.path.join(vid_root, vid)

        vidcap = cv2.VideoCapture(video_path)

        frames = []

        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for offset in range(min(num, int(total_frames - start))):
            success, img = vidcap.read()

            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

            if w > 256 or h > 256:
                img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

            img = (img / 255.) * 2 - 1

            frames.append(img)

        return np.asarray(frames, dtype=np.float32)

    @staticmethod
    def load_flow_frames_upd(image_dir, vid, start, num, rate = 1):
        video_path = ""
        if('.mp4' not in vid):
            video_path = os.path.join(image_dir, vid + '.mp4')
        else:
            video_path = os.path.join(image_dir, vid)#print(video_path)
        vidcap = cv2.VideoCapture(video_path)

        frames = []

        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        isFirst = True
    
        prev_gray = []
        # i = 0
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for offset in range(start, min(num, int(total_frames - start)), rate):
            success, img = vidcap.read()

            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

            if w > 256 or h > 256:
                img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

            # if(i < 31):
            #   cv2_imshow(img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if(isFirst):
                prev_gray = img
                mask = np.zeros((img.shape[0], img.shape[1],3))
                # mask = mask[...,np.newaxis, np.newaxis, np.newaxis]
                # mask = mask.reshape(img.shape[0], img.shape[1],3)
                #print(mask.shape)
                # Sets image saturation to maximum
                mask[..., 1] = 255
                isFirst = False
                continue

            # Calculates dense optical flow by Farneback method
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, img, 
                                                None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            #optical_flow = cv2.optflow.createOptFlow_DualTVL1()
            #flow = optical_flow.calc(prev_gray, img, None)

            #print('flow shape:', flow.shape)
            # Computes the magnitude and angle of the 2D vectors
            #magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #print('angle shape: ', angle.shape)
            #print(magnitude.shape)
            # Sets image hue according to the optical flow 
            # direction
            # o day ha
            #mask[..., 0] = angle * 180 / np.pi / 2
            
            # Sets image value according to the optical flow
            # magnitude (normalized)
            #mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            # Converts HSV to RGB (BGR) color representation
            #img_float32 = np.float32(mask)
            #lab_image = cv.cvtColor(img_float32, cv.COLOR_RGB2HSV)
            #rgb = cv2.cvtColor(img_float32, cv2.COLOR_HSV2BGR)
            #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            #rgb = np.asarray([rgb, prev_gray]).transpose([1, 2, 0])
            #prev_gray = img
            for i in range(0, rate):
                frames.append(flow)
        
        return np.asarray(frames, dtype=np.float32)


    @staticmethod
    def load_flow_frames_upd1(image_dir, vid, start, num, rate = 1):
        video_path = ""
        if('.mp4' not in vid):
            video_path = os.path.join(image_dir, vid + '.mp4')
        else:
            video_path = os.path.join(image_dir, vid)#print(video_path)
        vidcap = cv2.VideoCapture(video_path)

        frames = []

        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        isFirst = True
    
        prev_gray = []
        # i = 0
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for offset in range(start, min(num, int(total_frames - start)), rate):
            success, img = vidcap.read()

            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

            if w > 256 or h > 256:
                img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

            # if(i < 31):
            #   cv2_imshow(img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if(isFirst):
                prev_gray = img
                mask = np.zeros((img.shape[0], img.shape[1],3))
                # mask = mask[...,np.newaxis, np.newaxis, np.newaxis]
                # mask = mask.reshape(img.shape[0], img.shape[1],3)
                #print(mask.shape)
                # Sets image saturation to maximum
                mask[..., 1] = 255
                isFirst = False
                continue

            # Calculates dense optical flow by Farneback method
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, img, 
                                                None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            #optical_flow = cv2.optflow.createOptFlow_DualTVL1()
            #flow = optical_flow.calc(prev_gray, img, None)

            #print('flow shape:', flow.shape)
            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #print('angle shape: ', angle.shape)
            #print(magnitude.shape)
            # Sets image hue according to the optical flow 
            # direction
            mask[..., 0] = angle * 180 / np.pi / 2
            
            # Sets image value according to the optical flow
            # magnitude (normalized)
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            # Converts HSV to RGB (BGR) color representation
            img_float32 = np.float32(mask)
            #lab_image = cv.cvtColor(img_float32, cv.COLOR_RGB2HSV)
            rgb = cv2.cvtColor(img_float32, cv2.COLOR_HSV2BGR)
            #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            #rgb = np.asarray([rgb, prev_gray]).transpose([1, 2, 0])
            #prev_gray = img
            frames.append(rgb)
        
        return np.asarray(frames, dtype=np.float32)