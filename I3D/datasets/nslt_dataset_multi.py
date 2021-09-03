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


def crop_video(numpy_video, size, desired_size):
    """
    Crop a video of the given size (WIDTH, HEIGHT) into a square of `desired_size`.
    The video is represented as a numpy array. This func is for internal usage.
    """

    w, h = size
    h1, h2 = int(h/2) - int(desired_size/2), int(h/2) + int(desired_size/2)
    w1, w2 = int(w/2) - int(desired_size/2), int(w/2) + int(desired_size/2)
    return numpy_video[:, :, h1:h2, w1:w2, :]

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


def load_rgb_frames_from_video(vid_root, vid, start, num, resize=(256, 256)):
    video_path = os.path.join(vid_root, vid + '.mp4')

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

def video_to_flow_data(vid_root, video_file, size, nframes=None):
    """
    Loads a numpy array of shape (1, nframes, size, size, 2) from a video file.
    Values contained in the array are based on optical flow of the video.
    https://docs.opencv.org/3.1.0/d6/d39/classcv_1_1cuda_1_1OpticalFlowDual__TVL1.html

    Parameter `size` should be an integer (pixels) for a square cropping of the video.
    Omitting the parameter `nframes` will preserve the original # frames in the video.
    """
    video_path = video_path = os.path.join(vid_root, video_file + '.mp4')
    print('video path using to convert:', video_path)
    # Load video into numpy array, and crop the video
    w, h, buf = raw_numpy_array(video_path, nframes=nframes)
    buf = crop_video(buf, (w, h), size)
    print(buf)
    num_frames = buf.shape[1]
    flow = np.zeros((1, num_frames, size, size, 2), dtype='float32')
    print(flow)

    # Convert to grayscale
    buf = np.dot(buf, np.array([0.2989, 0.5870, 0.1140]))

    # Apply optical flow algorithm
    for i in range(1, num_frames):
        prev, cur = buf[0, i - 1], buf[0, i]
        cur_flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Truncate values to [-20, 20] and scale from [-1, 1]
        cur_flow[cur_flow < -20] = -20
        cur_flow[cur_flow > 20] = 20
        cur_flow /= 20
        flow[0, i] = cur_flow
    print(np.asarray(flow, dtype=np.float32))
    return np.asarray(flow, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)

def load_flow_frames1(image_dir, vid, start, num):
    video_path = os.path.join(image_dir, vid + '.mp4')
    vidcap = cv2.VideoCapture(video_path)

    frames = []
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    isFirst = True

    prev_gray = []
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
        #print('flow shape:', flow.shape)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        #print('angle shape: ', angle.shape)
        #print(magnitude.shape)
        # Sets image hue according to the optical flow 
        # direction
        # o day ha
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        img_float32 = np.float32(mask)
        #lab_image = cv.cvtColor(img_float32, cv.COLOR_RGB2HSV)
        rgb = cv2.cvtColor(img_float32, cv2.COLOR_HSV2BGR)

        rgb = np.asarray(rgb).transpose([1,2,0])
        prev_gray = img
        frames.append(rgb)

    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    print('data len make dataset:', len(data))
    i = 0
    count_skipping = 0
    for vid in data.keys():
        if split == 'train':
            if data[vid]['subset'] not in ['train', 'val']:
                continue
        else:
            if data[vid]['subset'] != 'test':
                continue

        vid_root = root['word']
        src = 0

        video_path = os.path.join(vid_root, vid + '.mp4')
        if not os.path.exists(video_path):
            continue

        num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

        #if mode == 'flow':
        #    num_frames = num_frames // 2

        #if num_frames - 0 < 9:
        #    print("Skip video ", vid)
        #    count_skipping += 1
        #    continue

        label = np.zeros((num_classes, num_frames), np.float32)

        for l in range(num_frames):
            c_ = data[vid]['action'][0]
            label[c_][l] = 1

        if len(vid) == 5:
            dataset.append((vid, label, src, 0, data[vid]['action'][2] - data[vid]['action'][1]))
        elif len(vid) == 6:  ## sign kws instances
            dataset.append((vid, label, src, data[vid]['action'][1], data[vid]['action'][2] - data[vid]['action'][1]))

        i += 1
    print("Skipped videos: ", count_skipping)
    return dataset

def get_num_class(split_file):
    classes = set()

    content = json.load(open(split_file))

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)

    return len(classes)

class NSLT(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None):
        self.num_classes = get_num_class(split_file)

        self.data = make_dataset(split_file, split, root, mode, num_classes=self.num_classes)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, src, start_frame, nf = self.data[index]
        print('video name:', vid)

        total_frames = 64

        try:
            start_f = random.randint(0, nf - total_frames - 1) + start_frame
        except ValueError:
            start_f = start_frame

        if self.mode == 'rgb':
            imgs = load_rgb_frames_from_video(self.root['word'], vid, start_f, total_frames)
        else:
            imgs = load_flow_frames1(self.root['word'], vid, start_f, total_frames)

        imgs, label = self.pad(imgs, label, total_frames)

        imgs = self.transforms(imgs)

        ret_lab = torch.from_numpy(label)
        ret_img = video_to_tensor(imgs)

        return ret_img, ret_lab, vid

    def __len__(self):
        return len(self.data)

    def pad(self, imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                prob = np.random.random_sample()
                if prob > 0.5:
                    pad_img = imgs[0]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
                else:
                    pad_img = imgs[-1]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            padded_imgs = imgs

        label = label[:, 0]
        label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_imgs, label

    @staticmethod
    def pad_wrap(imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                pad = imgs[:min(num_padding, imgs.shape[0])]
                k = num_padding // imgs.shape[0]
                tail = num_padding % imgs.shape[0]

                pad2 = imgs[:tail]
                if k > 0:
                    pad1 = np.array(k * [pad])[0]

                    padded_imgs = np.concatenate([imgs, pad1, pad2], axis=0)
                else:
                    padded_imgs = np.concatenate([imgs, pad2], axis=0)
        else:
            padded_imgs = imgs

        label = label[:, 0]
        label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_imgs, label