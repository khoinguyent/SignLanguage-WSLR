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
from pytorch_mlp import MLP

# from datasets.nslt_dataset import NSLT as Dataset
from datasets.nslt_dataset_multi import NSLT as Dataset
from datasets.asl_dataset import ASL as ASL_Dataset

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
        dataset = 'WLASL'):
    print(configs)

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                            videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = None
    dataloader = None
    val_dataset = None
    val_dataloader = None

    prefix = ''

    if(dataset == 'WLASL'):
    #RGB data stream
        dataset = Dataset(train_split, 'train', root, train_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True,
                                                    pin_memory=True)

        val_dataset = Dataset(train_split, 'test', root, test_transforms)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=True,
                                                    pin_memory=False)

        prefix = 'nslt_'
    elif(dataset == 'ASL'):
        print('ASL dataset')
        dataset = ASL_Dataset(root, 'train', 10, train_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True,
                                                    pin_memory=True)

        val_dataset = ASL_Dataset(root, 'test', 10, test_transforms)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=True,
                                                    pin_memory=False)

        prefix = 'asl_'

    dataloaders = {'train': dataloader, 'test': val_dataloader}

    num_classes = dataset.num_classes
    
    #load models
    i3d_flow = InceptionI3d(400, in_channels=2)
    i3d_flow.load_state_dict(torch.load('weights/flow_imagenet.pt'))

    i3d_rgb = InceptionI3d(400, in_channels=3)
    i3d_rgb.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    mlp = MLP(num_classes, 64)

    #replace logits
    i3d_rgb.replace_logits(num_classes)
    i3d_flow.replace_logits(num_classes)

    last_epoch = 0
    #load weights
    if(weights == None):
        if(os.path.exists(prefix + "logs.csv")):
            os.remove(prefix + "logs.csv")

        with open (prefix + "logs.csv",'a') as logs:
            line = 'epoch\tacc_train\ttot_loss_train\tacc_val\ttotal_loss_train\n'
            logs.writelines(line)
    else:
        print("load weights {}".format(weights))
        weights = torch.load(weights)
        print(weights)
        i3d_rgb.load_state_dict(weights['rgb'], strict=False)
        i3d_flow.load_state_dict(weights['flow'], strict=False)
        mlp.load_state_dict(weights['mlp'], strict=False)

        #continue write files
        #load the last epoch
        load_logs_data = pd.read_csv("logs.csv", sep='\t', engine='python')
        last_epoch = int(load_logs_data.tail(1).values.tolist()[0][0])

    #cuda
    i3d_rgb.cuda()
    i3d_rgb = nn.DataParallel(i3d_rgb)

    i3d_flow.cuda()
    i3d_flow = nn.DataParallel(i3d_flow)

    mlp.cuda()
    mlp = nn.DataParallel(mlp)
    
    #load config
    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    
    optimizer = optim.Adam([
        {'params': i3d_rgb.parameters()},
        {'params': i3d_flow.parameters()},
        {'params': mlp.parameters()}
    ], lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step  # accum gradient
    steps = 0
    epoch = 0

    best_val_score = 0

    acc_train = 0.0
    tot_loss_train = 0.0

    acc_val = 0.0
    tot_loss_val = 0.0
    
    #train
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    #train epoch
    while steps < configs.max_steps and epoch < 400: #why400
        epoch += 1

        for phase in ["train", "test"]:
            if phase == 'train':
                i3d_rgb.train(True)
                i3d_flow.train(True)
                mlp.train(True)
            else:
                i3d_rgb.train(False)
                i3d_flow.train(False)
                mlp.train(False)

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype = np.int)

            isDataloaderEnd = False
            video_number = 0
            #while(not isDataloaderEnd):
            for i, input in enumerate(dataloaders[phase]):
                num_iter += 1

                data_input, labels, vid = input

                inputs_rgb = data_input['rgb']
                inputs_flow = data_input['flow']

                #print('video: ',labels_rgb, vid, labels_flow, vid_flow)
                # wrap them in Variable
                inputs_rgb = inputs_rgb.cuda()
                t_rgb = inputs_rgb.size(2)
                labels = labels.cuda()

                per_frame_logits_rgb = i3d_rgb(inputs_rgb, pretrained=False)
                # upsample to input size
                per_frame_logits_rgb = F.upsample(per_frame_logits_rgb, t_rgb, mode='linear')

                inputs_flow = inputs_flow.cuda()
                t_flow = inputs_flow.size(2)

                per_frame_logits_flow = i3d_flow(inputs_flow, pretrained=False)
                # upsample to input size
                per_frame_logits_flow = F.upsample(per_frame_logits_flow, t_flow, mode='linear')
                
                outputs = None
                #put output of rgb stream and flow stream through MLP network
                for i in range(0, per_frame_logits_flow.shape[2]):
                    input_mlp = torch.cat((per_frame_logits_rgb[:,:,i], per_frame_logits_flow[:,:,i]), 1)
                    output = mlp(input_mlp)

                    if i == 0:
                        outputs = output
                    else:
                        outputs = torch.cat((outputs, output))

                outputs = outputs.unsqueeze(0)
                outputs = torch.transpose(outputs, 1, 2)
                
                #comput localization loss
                loc_loss = F.binary_cross_entropy_with_logits(outputs, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(outputs, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                #comput classification loos( with max-pooling along time B X C X T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(outputs, dim=2)[0],
                                                            torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data.item()

                for i in range(outputs.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()

                loss.backward()

                acc_train = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                tot_loss_train = tot_loss / 10

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()

                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        print(
                            'Epoch {} Step{} Video#{} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                                                steps,
                                                                                                                video_number,
                                                                                                                phase,
                                                                                                                tot_loc_loss / (10 * num_steps_per_update),
                                                                                                                tot_cls_loss / (10 * num_steps_per_update),
                                                                                                                tot_loss / 10,
                                                                                                                acc))
                            
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.0
                
            if phase == 'test':
                if not os.path.exists(os.path.join(os.getcwd(), save_model)):
                    os.mkdir(os.path.join(os.getcwd(), save_model))

                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if val_score > best_val_score or epoch % 2 == 0:
                    best_val_score = val_score
                    model_name = save_model + prefix + str(num_classes) + "_" + str(steps).zfill(
                                6) + '_%3f.pt' % val_score

                    torch.save({
                        'rgb': i3d_rgb.state_dict(),
                        'flow': i3d_flow.state_dict(),
                        'mlp': mlp.state_dict()
                    }, model_name)

                    print(model_name)
                
                    print('VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                        tot_loc_loss / num_iter,
                                                                                                        tot_cls_loss / num_iter,
                                                                                                        (tot_loss * num_steps_per_update) / num_iter,
                                                                                                        val_score
                                                                                                        ))

                acc_val = val_score
                tot_loss_val = (tot_loss * num_steps_per_update) / num_iter
                scheduler.step(tot_loss * num_steps_per_update / num_iter)
            

        with open (prefix + "logs.csv",'a') as logs:
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
    parser.add_argument('--dataset', type=str, default='WLASL')

    args = parser.parse_args()

    mode = args.mode
    root = args.root
    weights = args.weights
    save_model = args.save_model
    num_class = args.num_class
    config_file = args.config
    train_split = args.train_split
    dataset = args.dataset

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
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights, dataset=dataset)