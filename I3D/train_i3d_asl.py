import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torchvision import transforms
import videotransforms
import pandas as pd
import numpy as np

from config import Config
from pytorch_i3d import InceptionI3d

# from datasets.nslt_dataset import NSLT as Dataset
from datasets.asl_dataset import ASL as Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None,
        rate=1):
    print(configs)

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(root, 'train', int(train_split), mode, train_transforms, rate)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    val_dataset = Dataset(root, 'test', int(train_split), mode, test_transforms, rate)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    num_classes = dataset.num_classes
    i3d.replace_logits(num_classes)

    if weights:
        print('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    last_epoch = 0
    #load weights
    prefix = "nlst_"
    log_file = prefix + mode + "_" + str(num_classes) + "_log.csv"
    log_file = os.path.join(save_model, log_file)
    if(weights == None):
        if(os.path.exists(log_file)):
            os.remove(log_file)

        with open (log_file,'a') as logs:
            line = 'epoch\tacc_train\ttot_loss_train\tacc_val\ttotal_loss_val\n'
            logs.writelines(line)
    else:
        print("load weights {}".format(weights))
        weights = torch.load(weights)
        print(weights)
        i3d.load_state_dict(weights)

        #continue write files
        #load the last epoch
        load_logs_data = pd.read_csv(log_file, sep='\t', engine='python')
        last_epoch = int(load_logs_data.tail(1).values.tolist()[0][0])
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step  # accum gradient
    steps = 0
    epoch = 0

    best_val_score = 0

    acc_train = 0.0
    tot_loss_train = 0.0

    acc_val = 0.0
    tot_loss_val = 0.0
    # train it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    while steps < configs.max_steps and epoch < 400:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            collected_vids = []

            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                if data == -1: # bracewell does not compile opencv with ffmpeg, strange errors occur resulting in no video loaded
                    continue

                # inputs, labels, vid, src = data
                inputs, labels, vid = data

                # wrap them in Variable
                inputs = inputs.cuda()
                t = inputs.size(2)
                labels = labels.cuda()

                per_frame_logits = i3d(inputs, pretrained=False)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()
                #if num_iter == num_steps_per_update // 2:
                #    print(epoch, steps, loss.data.item())
                loss.backward()

                #acc_train = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                tot_loss_train = tot_loss / 10

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        acc_train = acc
                        print(
                            'Epoch {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                                                 phase,
                                                                                                                 tot_loc_loss / (10 * num_steps_per_update),
                                                                                                                 tot_cls_loss / (10 * num_steps_per_update),
                                                                                                                 tot_loss / 10,
                                                                                                                 acc))
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'test':
                if not os.path.exists(os.path.join(os.getcwd(), save_model)):
                    os.mkdir(os.path.join(os.getcwd(), save_model))

                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if val_score > best_val_score or epoch % 2 == 0:
                    best_val_score = val_score
                    model_name = save_model + "nslt_" + mode + str(num_classes) + "_" + str(steps).zfill(
                                6) + '_%3f.pt' % val_score

                    torch.save(i3d.module.state_dict(), model_name)
                    

                print('VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                            tot_loc_loss / num_iter,
                                                                                                            tot_cls_loss / num_iter,
                                                                                                            (tot_loss * num_steps_per_update) / num_iter,
                                                                                                            val_score
                                                                                                            ))
                acc_val = val_score
                tot_loss_val = (tot_loss * num_steps_per_update) / num_iter
                scheduler.step(tot_loss * num_steps_per_update / num_iter)
        
        with open (log_file,'a') as logs:
            line = '{}\t{}\t{}\t{}\t{}\n'.format(epoch + last_epoch, acc_train, tot_loss_train, acc_val, tot_loss_val)
            logs.writelines(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='rgb or flow', default='rgb')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--save_model', type=str, default='checkpoints/')
    parser.add_argument('--root', type=str, default={'word': '../data/WLASL2000'})
    parser.add_argument('--num_class', type=int, default=2000)
    parser.add_argument('--config', type=str, default='configfiles/asl2000.ini')
    parser.add_argument('--train_split', type=str, default='preprocess/nslt_2000.json')
    parser.add_argument('--dataset_name', type=str, default='WLASL')
    parser.add_argument('--rate', type=int, default=1)

    args = parser.parse_args()

    mode = args.mode
    root = args.root
    weights = args.weights
    save_model = args.save_model
    num_class = args.num_class
    config_file = args.config
    train_split = args.train_split
    dataset_name = args.dataset_name
    rate = args.rate

    # WLASL setting
    # mode = 'rgb'
    # root = {'word': '../../data/WLASL2000'}

    # save_model = 'checkpoints/'
    # train_split = 'preprocess/nslt_2000.json'

    # weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    # weights = None
    # config_file = 'configfiles/asl2000.ini'

    configs = Config(config_file)
    print(root, train_split)
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights, rate=rate)