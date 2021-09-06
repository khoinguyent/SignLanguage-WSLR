import torch.utils.data as data_utl
import cv2
import numpy as np
import os
from videoprocessing import VideoProcessing as vp

import random
import torch

def get_num_class():
    return 64

def make_dataset(path, split, num_classes):
    dataset = []
    #get list of mp4 files
    videos = [fname for fname in os.listdir(path) if fname.endswith('mp4')]

    #each word has 5 signers, each signer records 5 videos
    for video in videos:
        name = (video.split(".")[0]).split("_")[-1]

        if split == 'train' and name in ('003', '004'):
            continue
        if split == 'test' and name not in ('003', '004'):
            continue
        
        video_path = os.path.join(path, video)
        num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

        label = np.zeros((num_classes, num_frames), np.float32)

        for l in range(num_frames):
            c_ = int((video.split(".")[0]).split("_")[0]) - 1
            label[c_][l] = 1
        
        dataset.append((video, label, 0, 0, num_frames))
    
    return dataset

class ASL(data_utl.Dataset):
    def __init__(self, path, split, transforms=None):
        self.num_classes = get_num_class()
        self.data = make_dataset(path, split, num_classes=self.num_classes)
        self.transforms = transforms
        self.path = path

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, src, start_frame, nf = self.data[index]
        total_frames = 64

        try:
            start_f = random.randint(0, nf - total_frames - 1) + start_frame
        except ValueError:
            start_f = start_frame

        imgs_rgb = vp.load_rgb_frames_from_video(self.path, vid, start_f, total_frames)
        imgs_flow = vp.load_flow_frames_upd(self.path, vid, start_f, total_frames)
        
        imgs_rgb, label = self.pad(imgs_rgb, label, total_frames)
        imgs_flow, label = self.pad(imgs_flow, label, total_frames)
        #print(imgs)
        imgs_rgb = self.transforms(imgs_rgb)
        imgs_flow = self.transforms(imgs_flow)

        ret_lab = torch.from_numpy(label)
        ret_img = {
            'rgb': vp.video_to_tensor(imgs_rgb),
            'flow': vp.video_to_tensor(imgs_flow)
        }

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