import os

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import argparse


### This is for Original CompNet

parser = argparse.ArgumentParser(
        description="PSFed-Palm"
    )

parser.add_argument("--batch_size",type=int,default = 2048)
parser.add_argument("--epoch_num",type=int,default = 3)
parser.add_argument("--com",type= int,default=100)
parser.add_argument("--temp", type=float, default= 0.07)
parser.add_argument("--weight1",type=float,default = 0.7)
parser.add_argument("--weight2",type=float,default = 0.15)
parser.add_argument("--weight3",type=float,default = 100)
parser.add_argument("--mu",type=float,default = 1e-2)

parser.add_argument("--id_num",type=int, default = 600, help = "IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500 IITD_Right 230 No_Delete_PolyU 386 Tongji_LR 300")
parser.add_argument("--gpu_id",type=str, default='0')
parser.add_argument("--lr",type=float, default=0.001)
parser.add_argument("--redstep",type=int, default=30)
parser.add_argument("--mode", type=str, default='fedavg', help="fedavg|fedprox|fedpdf")

parser.add_argument("--test_interval",type=str,default = 500)
parser.add_argument("--save_interval",type=str,default = 200)  ## 200 for Multi-spec 500 for RED

##Training Path
parser.add_argument("--train_set_file",type=str,default='./data/train_all_server.txt')
parser.add_argument("--test_set_file",type=str,default='./data/test_server.txt')

##Store Path
parser.add_argument("--des_path",type=str,default='/data/YZY/Palm_DOC/Tongji_add/checkpoint/')
parser.add_argument("--path_rst",type=str,default='/data/YZY/Palm_DOC/Tongji_add/rst_test/')
parser.add_argument("--save_path",type=str,default='./cross-db-checkpoint/PolyU_1')
parser.add_argument("--seed",type=int,default=42)
args = parser.parse_args()

# print(args.gpu_id)
# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id



import time
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision import models

# print(torch.cuda.is_available())
# print(os.getcwd())
# import pickle
import numpy as np
from PIL import Image
import cv2 as cv
from loss import SupConLoss

import matplotlib.pyplot as plt

from utils.util import plotLossACC, saveLossACC, saveGaborFilters, saveParameters, saveFeatureMaps

plt.switch_backend('agg')

from models import MyDataset
# from models.compnet_original import compnet
# from models.co3 import compnet2 as co3net
from models import ccnet
from utils import *

import copy
import random

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        elif args.mode.lower() == 'fedper':
            for key in server_model.state_dict().keys():
##                if 'weight_fed' not in key and 'MLP' not in key :
#                if 'keys' not in key:
                if 'fc' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def communication_sub(s_model, models):
    with torch.no_grad():
        # aggregate params
        for key in s_model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                s_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:
                temp = torch.zeros_like(s_model.state_dict()[key])
                for client_idx in range(len(models)):
                    temp += 1/2 * models[client_idx].state_dict()[key]
                s_model.state_dict()[key].data.copy_(temp)
    return s_model



def test(model, gallery_file, query_file, path_rst):
    # finished training
    # torch.save(net.state_dict(), 'net_params.pth')
    # torch.save(net, 'net.pkl')

    # print('Finished Trainning')
    # print('the best training acc is: ', bestacc, '%')
    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    ### Calculate EER

    # path_rst = './Tongji2others/Tongji2IITD/rst_test/'
    # if not os.path.exists(path_rst):
    #     os.makedirs(path_rst)

    path_hard = os.path.join(path_rst, 'rank1_hard')

    # train_set_file = './data/train_IITD.txt'
    # test_set_file = './data/test_IITD.txt'

    trainset = MyDataset(txt=gallery_file, transforms=None, train=False)
    testset = MyDataset(txt=query_file, transforms=None, train=False)

    batch_size = 512  # 128

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=2)

    fileDB_train = getFileNames(gallery_file)
    fileDB_test = getFileNames(query_file)

    # output dir
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    # num_classes = 600  # IITD: 460    KTU: 145    Tongji: 600    REST: 358    XJTU: 200
    net = model

    # device = torch.device("cuda")
    net.cuda()
    net.eval()

    # feature extraction:

    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):
        # break

        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]
    # assert num_training_samples % classNumel == 0
    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_test):

        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    print('completed feature extraction for test set.')
    print('featDB_test.shape: ', featDB_test.shape)

    print('\nfeature extraction done!')
    print('\n\n')

    print('start feature matching ...\n')

    print('Verification EER of the test set ...')

    print('Start EER for Train-Test Set ! Its wrong? \n')

    # verification EER of the test set
    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]

        for j in range(ntrain):
            feat2 = featDB_train[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    if not os.path.exists(path_rst+'veriEER'):
        os.makedirs(path_rst+'veriEER')
    if not os.path.exists(path_rst+'veriEER/rank1_hard/'):
        os.makedirs(path_rst+'veriEER/rank1_hard/')

    with open(path_rst+'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    # rank-1 acc
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]

        dis = np.zeros((ntrain, 1))

        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1

        idx = np.argmin(dis[:])

        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1
        else:
            testname = fileDB_test[i]
            trainname = fileDB_train[idx]
            # store similar inter-class samples
            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)
            img = np.concatenate((im_test, im_train), axis=1)
            cv.imwrite(path_rst + 'veriEER/rank1_hard/%6.4f_%s_%s.png' % (
                np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)

    print('\n\nReal EER of the test set...')
    # dataset EER of the test set (the gallery set is not used)
    s = []  # matching score
    l = []  # genuine / impostor matching
    n = featDB_test.shape[0]
    for i in range(n - 1):
        feat1 = featDB_test[i]

        for jj in range(n - i - 1):
            j = i + jj + 1
            feat2 = featDB_test[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_test[j]:
                l.append(1)
            else:
                l.append(-1)

    print('feature extraction about real EER done!\n')

    with open(path_rst + 'veriEER/scores_EER_test.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')

def fit(epoch, model, data_loader, optimize=None, server_model=None, mode = 'fedavg', aux_model = None, phase='training'):

    if phase != 'training' and phase != 'testing':
        raise TypeError('input error!')

    if phase == 'training':
        model.train()
        aux_model.train()
        server_model.train()
    if phase == 'testing':
        # print('test')
        model.eval()
        aux_model.eval()
        server_model.eval()

    running_loss = 0
    entro_loss = 0
    supcon_loss = 0
    prox_loss = 0
    mse_loss = 0

    running_correct = 0

    cri_mse = nn.MSELoss().cuda()

    for batch_id, (datas, target) in enumerate(data_loader):

        data = datas[0]
        data = data.cuda()

        data_con = datas[1]
        data_con = data_con.cuda()

        target = target.cuda()
        if phase == 'training':
            optimize.zero_grad()
            output, fe1 = model(data, target)
            _, fe2 = model(data_con, target)
            _, fe3 = aux_model(data,target)

            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                output, fe1 = model(data, None)
                _, fe2 = model(data_con, None)
                _, fe3 = aux_model(data, None)
                fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

        ce = criterion(output, target)
        ce2 = con_criterion(fe, target)
        ce3 = cri_mse(fe1,fe3.detach())

        w_diff = torch.tensor(0.).cuda()
        for w, w_t in zip(server_model.parameters(), model.parameters()):
            w_diff += torch.pow(torch.norm(w - w_t), 2)
        w_diff = torch.sqrt(w_diff)
        loss2 = mu / 2. * w_diff

        w_diff = torch.tensor(0.).cuda()
        for w, w_t in zip(aux_model.parameters(), model.parameters()):
            w_diff += torch.pow(torch.norm(w - w_t), 2)
        w_diff = torch.sqrt(w_diff)
        loss3 = mu / 2. * w_diff
        
        loss2 = loss2 + loss3
        # loss2 = loss3 

        loss = weight1*ce + weight2*ce2 + weight3 * ce3 + loss2
        # loss = weight1*ce + weight2*ce2 + loss2
        # loss = weight1*ce + weight2*ce2 + weight3 * ce3

        running_loss += loss.data.cpu().numpy()
        entro_loss += ce.data.cpu().numpy()
        supcon_loss += ce2.data.cpu().numpy()
        prox_loss += loss2.data.cpu().numpy() * weight3
        mse_loss += ce3.data.cpu().numpy()

        preds = output.data.max(dim=1, keepdim=True)[1]  # max returns (value, index)
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()

        ## update
        if phase == 'training':
            loss.backward(retain_graph=None)  #
            optimize.step()

    ## log info of this epoch
    total = len(data_loader.dataset)
    loss = running_loss / total
    entroloss = entro_loss / total
    supconloss = supcon_loss / total
    proxloss = prox_loss / total
    mseloss = mse_loss *weight3 / total
    accuracy = (100.0 * running_correct) / total

    if epoch % 5 == 0:
        print('epoch %d: \t%s loss is \t%7.5f Entropy_loss is \t%7.5f SupCon_loss is \t%7.5f Prox_loss is \t%7.5f MSE_loss is \t%7.5f ;\t%s accuracy is \t%d/%d \t%7.3f%%' % (
        epoch, phase, loss, entroloss, supconloss, proxloss, mseloss, phase, running_correct, total, accuracy))

    return loss, accuracy

def Dataset():

    src_dataset_1 = DataLoader(MyDataset(txt='./data/train_MSRed.txt', transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=2, shuffle=True)
    src_dataset_2 = DataLoader(MyDataset(txt='./data/train_MSGreen.txt', transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=2, shuffle=True)
    src_dataset_3 = DataLoader(MyDataset(txt='./data/train_MSBLUE.txt', transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=2, shuffle=True)
    src_dataset_4 = DataLoader(MyDataset(txt='./data/train_MSNIR.txt', transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=2, shuffle=True)
    
    dataloaders = []
    dataloaders.append(src_dataset_1)
    dataloaders.append(src_dataset_2)
    dataloaders.append(src_dataset_3)
    dataloaders.append(src_dataset_4)

    return dataloaders

if __name__== "__main__" :

    set_seed(args.seed)
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    num_classes = args.id_num  # IITD: 460    KTU: 145    Tongji: 600    REST: 358    XJTU: 200 POLYU 378
    weight1 = args.weight1
    weight2 = args.weight2
    weight3 = args.weight3
    mu = args.mu
    communications = args.com
    ##Checkpoint Path
    print('seed:',args.seed)
    print('weight of cross:', weight1)
    print('weight of contra:', weight2)
    print('weight of mse:', weight3)
    print('mu:', mu)
    print('tempture:', args.temp)
    des_path = args.des_path
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    # path
    # train_set_file = args.train_set_file
    # test_set_file = args.test_set_file
    train_set_files = ['./data/train_MSRed.txt','./data/train_MSGreen.txt' ,'./data/train_MSBLUE.txt','./data/train_MSNIR.txt']
    test_set_files = ['./data/test_MSRed.txt','./data/test_MSGreen.txt' ,'./data/test_MSBLUE.txt','./data/test_MSNIR.txt']

    names = ['red','green','blue','nir']
    
    path_rst = args.path_rst

    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    train_datas = Dataset()
    server_model = ccnet(num_classes=num_classes).cuda()
    best_net = ccnet(num_classes=num_classes).cuda()

    visib_net = ccnet(num_classes=num_classes).cuda()   ### Green & Blue
    invis_net = ccnet(num_classes=num_classes).cuda() 

    models = [copy.deepcopy(server_model) for idx in range(4)]
    optimizers = [torch.optim.Adam(models[idx].parameters(), lr=args.lr) for idx in range(4)]
    schedulers = [lr_scheduler.StepLR(optimizers[idx], step_size=args.redstep, gamma=0.8) for idx in range(4)]
    client_weights = [1 / 4 for i in range(4)]
    #
    criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=args.temp, base_temperature=args.temp)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    
    bestacc = 0


    for com in range(communications):
    
        temp_val_acc = []
        visb_models = []
        invis_models = []
    
        for idx in range(4):
            print(com,names[idx])
            for epoch in range(epoch_num):

                if idx == 1 or idx ==2:   ### visble
                    epoch_loss, epoch_accuracy = fit(epoch, models[idx], train_datas[idx], optimize=optimizers[idx], server_model= server_model, mode=args.mode.lower(), aux_model = invis_net, phase='training')
                else:
                    epoch_loss, epoch_accuracy = fit(epoch, models[idx], train_datas[idx], optimize=optimizers[idx], server_model= server_model, mode=args.mode.lower(), aux_model = visib_net, phase='training')
                
                if idx == 1 or idx ==2:
                    val_epoch_loss, val_epoch_accuracy = fit(epoch, models[idx], train_datas[idx], server_model= server_model, aux_model= invis_net , phase='testing')
                else:
                    val_epoch_loss, val_epoch_accuracy = fit(epoch, models[idx], train_datas[idx], server_model= server_model, aux_model= visib_net, phase='testing')
                schedulers[idx].step()

        # ------------------------logs----------------------
                train_losses.append(epoch_loss)
                train_accuracy.append(epoch_accuracy)

                val_losses.append(val_epoch_loss)
                val_accuracy.append(val_epoch_accuracy)

                temp_val_acc.append(val_epoch_accuracy)

            if idx==1 or idx ==2:
                visb_models.append(models[idx])
            else:
                invis_models.append(models[idx])

        visib_net = communication_sub(visib_net,visb_models)
        invis_net = communication_sub(invis_net, invis_models)

        server_model, models = communication(args, server_model, models, client_weights)

        val_epoch_accuracy = sum(temp_val_acc)/temp_val_acc.__len__()
        # save the best model
        if val_epoch_accuracy >= bestacc:
            bestacc = val_epoch_accuracy
            torch.save(server_model.state_dict(), des_path + 'net_params_best.pth')
            for key in server_model.state_dict().keys():
                best_net.state_dict()[key].data.copy_(server_model.state_dict()[key])

        # save the current model and log info:
        if com % 10 == 0 or com == (epoch_num - 1) and com != 0:
            torch.save(server_model.state_dict(), des_path + 'net_params.pth')
            saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, bestacc, path_rst)

        if com % args.save_interval == 0:
            torch.save(server_model.state_dict(), des_path + 'com_' + str(com) + '_net_params.pth')

        ###第一次也测一下 看看有没有bug
        if com % args.test_interval == 0 and com != 0:
            for source_id in range(4):
                for target_id in range(4):
                    print(names[source_id],'->',names[target_id])
                    path_rst = args.path_rst + names[source_id] + '2' + names[target_id] + '_best/'
                    test(server_model,train_set_files[source_id],test_set_files[target_id], path_rst)

