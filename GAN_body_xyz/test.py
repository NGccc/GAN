import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pdb
from mydataset import MyDataset
from utils import initialize_weights
import torch.autograd as autograd
import cv2,math
from WGAN_GP_IN import G

def getCompleteAction8(x, tofile):
    left  = [(i+3) for i in [0,2,4,8,12,14,16,17]]
    right = [(i+3) for i in [1,3,5,9,13,15,16,18]]
    x = x.squeeze()
    out = np.zeros((64,54)).astype('float')
    #pdb.set_trace()
    
    out[:,left] = x
    out[:,right]= x
    np.savetxt(tofile, out)

def getCompleteAction64(x, tofile):
    left  = [(i+3) for i in [0,2,4,8,12,14,16,17]]
    right = [(i+3) for i in [1,3,5,9,13,15,16,18]]
    x = x.squeeze()
    xx  = np.zeros((64,8)).astype('float')
    out = np.zeros((64,54)).astype('float')
    #pdb.set_trace()
    for i in range(8,64,8):
        xx[:,int(i/8)-1] = x[:,i-8:i].mean(1)
    
    out[:,left] = xx
    out[:,right]= xx
    np.savetxt(tofile, out)

def getCompleteAction48(x, tofile):
    #left  = [(i+3) for i in [0,2,4,8,12,14,16,17]]
    #right = [(i+3) for i in [1,3,5,9,13,15,16,18]]
    x = x.squeeze()
    ALL = [(i+3) for i in range(21)]
    for i in range(27,54):
        ALL.append(i)
    #import pdb
    #pdb.set_trace()
    out = np.zeros((96,54)).astype('float')
    xx  = np.zeros((96,48)).astype('float')
    for i in range(2,96,2):
        xx[:,int(i/2)-1] = x[:,i-2:i].mean(1)
    #pdb.set_trace()
    
    out[:,ALL] = xx
    np.savetxt(tofile, out)

def euler2quaternion(euler):
    X, Y, Z = euler[0], euler[1], euler[2]
    X, Y, Z =   (X - 360) * math.pi / 180 if X > 180 else X * math.pi / 180, \
                (Y - 360) * math.pi / 180 if Y > 180 else Y * math.pi / 180, \
                (Z - 360) * math.pi / 180 if Z > 180 else Z * math.pi / 180
    c1 = math.cos(Y * 0.5)
    s1 = math.sin(Y * 0.5)

    c2 = math.cos(Z * 0.5)
    s2 = math.sin(Z * 0.5)

    c3 = math.cos(X * 0.5)
    s3 = math.sin(X * 0.5)

    w = c1 * c2 * c3 + s1 * s2 * s3
    x = s1 * s2 * c3 + c1 * c2 * s3
    y = s1 * c2 * c3 - c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3
    return [x, y, z, w]

def getCompleteAction128(x, tofile):
    #left  = [(i+3) for i in [0,2,4,8,12,14,16,17]]
    #right = [(i+3) for i in [1,3,5,9,13,15,16,18]]
    
    ALL = [i for i in range(21)]
    for i in range(24,51):
        ALL.append(i)
    #import pdb
    #pdb.set_trace()
    out = np.zeros((128,51)).astype('float')
    xx  = np.zeros((128,48)).astype('float')
    for i in range(2,98,2):
        xx[:,int(i/2)-1] = x[:,i-2:i].mean(1)
    #pdb.set_trace()
    out[:,ALL] = xx
    np.savetxt(tofile, out)
    used = set([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 58, 59, 60, 61, 62, 63, 64, 65, 69, 70, 71, 72, 73, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 88, 89, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 123, 124, 125, 126, 127, 128, 129, 130, 131, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 148, 149, 150, 151, 152, 153, 154, 155, 159, 160, 161, 162, 163, 164, 165, 166, 167, 171, 172, 173, 174, 175, 176, 177, 178, 179, 183, 184, 185, 188, 189, 190, 191, 198, 199, 200, 203, 204, 205, 206])
    f = open(tofile.replace('blendshape', 'skeleton'),'w+',encoding='utf-8-sig')
    ske = ['Bip001', 'Bip001 Footsteps', 'Bip001 Pelvis', 'Bip001 Spine', 'Bip001 Spine1', 'Bip001 Spine2', 'Bip001 Neck', 'Bip001 L Clavicle', 'Bip001 L UpperArm', 'Bip001 L Forearm', 'Bip001 L Hand', 'Bip001 L Finger0', 'Bip001 L Finger01', 'Bip001 L Finger02', 'Bip001 L Finger0Nub', 'Bip001 L Finger1', 'Bip001 L Finger11', 'Bip001 L Finger12', 'Bip001 L Finger1Nub', 'Bip001 L Finger2', 'Bip001 L Finger21', 'Bip001 L Finger22', 'Bip001 L Finger2Nub', 'Bip001 L Finger3', 'Bip001 L Finger31', 'Bip001 L Finger32', 'Bip001 L Finger3Nub', 'Bip001 L Finger4', 'Bip001 L Finger41', 'Bip001 L Finger42', 'Bip001 L Finger4Nub', 'Bip001 L ForeTwist', 'Bip001 L ForeTwist1', 'Bip001 R Clavicle', 'Bip001 R UpperArm', 'Bip001 R Forearm', 'Bip001 R Hand', 'Bip001 R Finger0', 'Bip001 R Finger01', 'Bip001 R Finger02', 'Bip001 R Finger0Nub', 'Bip001 R Finger1', 'Bip001 R Finger11', 'Bip001 R Finger12', 'Bip001 R Finger1Nub', 'Bip001 R Finger2', 'Bip001 R Finger21', 'Bip001 R Finger22', 'Bip001 R Finger2Nub', 'Bip001 R Finger3', 'Bip001 R Finger31', 'Bip001 R Finger32', 'Bip001 R Finger3Nub', 'Bip001 R Finger4', 'Bip001 R Finger41', 'Bip001 R Finger42', 'Bip001 R Finger4Nub', 'Bip001 R ForeTwist', 'Bip001 R ForeTwist1', 'Bip001 Head', 'Bip001 HeadNub', 'Bip001 L Thigh', 'Bip001 L Calf', 'Bip001 L Foot', 'Bip001 L Toe0', 'Bip001 L Toe0Nub', 'Bip001 R Thigh', 'Bip001 R Calf', 'Bip001 R Foot', 'Bip001 R Toe0', 'Bip001 R Toe0Nub']
    default = ['0.0', '-0.0', '90.00003546027156', '0.0', '0.0', '0.0', '-0.0', '-0.0', '0.0', '-0.0', '-0.0', '0.0', '-0.0', '-0.0', '0.0', '-0.0', '-0.0', '0.0', '-0.0', '-0.0', '0.0', '0.0', '0.0', '0.0', '180.0', '0.0', '0.0', '180.0', '0.0', '0.0', '180.0', '0.0', '-0.0', '180.0', '0.0', '-0.0', '180.0', '0.0', '-0.0', '-0.0', '0.0', '0.0', '0.0', '0.0', '269.9999645397786', '0.0', '0.0', '180.0', '0.0', '0.0', '-0.0', '0.0', '269.9999645397786', '0.0', '-0.0', '-0.0']
    #euler = []
    for r in range(x.shape[0]):
        res = ''
        th  = 0
        default_index = 0
        i = 98
        dt = 2
        import pdb
        #pdb.set_trace()
        euler = []
        while int(th/3) <= 71:
            if i >= 104 and dt==2:
                i = 103
                dt = 1
            if th % 3 == 0:
                if euler:
                    quaternion = euler2quaternion(euler)
                    for q in quaternion:
                        res = res + str(q) + ','
                if int(th/3) == 71:
                    break
                res = res + ske[int(th/3)] + ','
                euler = []
                #print(ske[int(th/3)])
            if th == 0:
                res = res + '-1.317390,75.786125,64.521378,'
            if th in used:
                #res = res + str(x[r,i-dt:i].mean()) + ',' #96 97,98 99, 100 101, 102,103...255, == 154
                euler.append(x[r,i-dt:i].mean())
            else:
                #res = res + str(default[default_index]) + ','
                euler.append(float(default[default_index]))
                default_index += 1 
                th += 1
                continue
            i += dt
            th += 1
        f.write(res + '\n')


def getBodyAction64(x, path):
    f = open(path,'w+',encoding='utf-8-sig')
    '''
    ske = ['Bip001 Spine', 'Bip001 Spine1', 'Bip001 Spine2', 'Bip001 Neck','Bip001 Head',\
    'Bip001 L Clavicle', 'Bip001 L UpperArm', 'Bip001 L Forearm', 'Bip001 L Hand', 'Bip001 L Finger0', 'Bip001 L Finger1', 'Bip001 L Finger2', 'Bip001 L Finger3', 'Bip001 L Finger4',\
    'Bip001 R Clavicle', 'Bip001 R UpperArm', 'Bip001 R Forearm', 'Bip001 R Hand', 'Bip001 R Finger0', 'Bip001 R Finger1', 'Bip001 R Finger2', 'Bip001 R Finger3', 'Bip001 R Finger4',\
    'Bip001 L Thigh', 'Bip001 L Calf', 'Bip001 L Foot', 'Bip001 L Toe0', \
    'Bip001 R Thigh', 'Bip001 R Calf', 'Bip001 R Foot', 'Bip001 R Toe0',]
    pdb.set_trace()
    res = 'Bip001,0.0,0.0,0.0,-0.511119,-0.451622,-0.487246,0.545330,Bip001 Pelvis,0.0,0.0,0.0,0.0,0.0,0.0,'
    for i in range(x.shape[0]):
        for j in range(0, x.shape[1], 3):
            res=res+ske[j//3]+','
            res=res+str(x[i][j])+','+str(x[i][j+1])+','+str(x[i][j+2])+','
        f.write(res+'\n')
        res = 'Bip001,0.0,0.0,0.0,-0.511119,-0.451622,-0.487246,0.545330,Bip001 Pelvis,0.0,0.0,0.0,0.0,0.0,0.0,'
    '''
    res = 'Bip001,0.0,0.0,0.0,-0.511119,-0.451622,-0.487246,0.545330,Bip001 Pelvis,-0.499519,0.483693,0.500690,0.515589,'
    ske = ['Bip001 L Foot','Bip001 L Calf','Bip001 L Thigh', \
    'Bip001 L Hand','Bip001 L Forearm','Bip001 L UpperArm','Bip001 L Clavicle',\
    'Bip001 Neck','Bip001 Spine2','Bip001 Spine1','Bip001 Spine',\
    'Bip001 R Clavicle', 'Bip001 R UpperArm', 'Bip001 R Forearm', 'Bip001 R Hand',\
    'Bip001 R Thigh', 'Bip001 R Calf', 'Bip001 R Foot'
    ]
    from rotation2quart import rotvector2quart
    '''
    minx = -81.36
    maxx = 79.6
    miny = -45.0
    maxy = 72.1
    minz = -55.1
    maxz = 53.8
    '''
    for i in range(x.shape[0]):
        for j in range(0,54,3):
            res = res + ske[j//3] + ','
            #pdb.set_trace()
            #x[i][j]=(x[i][j]+1)/2.0
            #x[i][j+1]=(x[i][j+1]+1)/2.0
            #x[i][j+2]=(x[i][j+2]+1)/2.0
            quart = rotvector2quart([x[i][j],x[i][j+1],x[i][j+2]])
            #
            for k in range(4): 
                y = np.complex128(quart[k])
                #pdb.set_trace()
                #if type(quart[k]) == np.complex128:
                #print(type(quart[k]))
                #print(type(quart[k]))
                #print(quart[k])
                #pdb.set_trace()
                #x   = complex(quart[k])
                #print(x)
                res = res + str(y.real) + ','
        f.write(res+'\n')
        res = 'Bip001,0.0,0.0,0.0,-0.511119,-0.451622,-0.487246,0.545330,Bip001 Pelvis,-0.499519,0.483693,0.500690,0.515589,'
    pass

def test(model, epoch):
    import pdb
    #pdb.set_trace()
    for xxx in range(1):
        z_ = torch.rand((1, 100, 1, 1)).float().cuda()
        G_ = model(z_).squeeze()
        getBodyAction64(G_.cpu().detach().numpy(), r'C:\Users\sunyi03\Desktop\FaceRecognitionaudio\Assets\GAN_All_body\%d_skeleton.txt' % xxx)
        #pdb.set_trace()
        '''
        x  = np.zeros((G_.shape[0],G_.shape[1])).astype('float')
        x[:,:96] = (0.5*G_.cpu().detach().numpy()[:,:96]+0.5)*100.0
        x[:,96:] = (0.5*G_.cpu().detach().numpy()[:,96:]+0.5)*360.0
        for i in range(x.shape[0]):
            for j in range(96,x.shape[1],1):
                if x[i,j] >= 180:
                    x[i,j] -= 360
        '''
        #getCompleteAction128(x, r'C:\Users\sunyi03\Desktop\FaceRecognitionaudio\Assets\GAN_body\%d_blendshape.txt' % xxx)

G = G(True).cuda()
dic_G = torch.load(r'D:\GAN_TEST\GAN_body_xyz\models_notanh\epoch_499_WD_10410.7324_Gloss_5589.8442_Dloss_-6815.6318_G.pth')
G.load_state_dict(dic_G['state_dict'])
test(G,499)