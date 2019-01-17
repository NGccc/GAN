import pdb
import math
from math import cos,sin
import queue
import numpy as np
from math import cos,sin
from quart2rotation import quart2Rotationvector
import torch
def dfs(dic, node):
    dic[node]['pos'] = np.array(dic[node]['pos'])
    for child in dic[node]['children']:
        dfs(dic, child)

def getTopoTree():
    dic = {}
    dic['Bip001'] = {'father':'', 'children':['Bip001 Pelvis'], 'pos':[0,0,0]}
    dic['Bip001 Pelvis'] = {'father':'Bip001', 'children':['Bip001 Spine'], 'pos':[0,0,0]}
    dic['Bip001 Spine'] = {'father':'Bip001 Pelvis', 'children':['Bip001 L Thigh','Bip001 R Thigh','Bip001 Spine1'], 'pos':[-11.16231, 1.71701, 9.612973e-06]}
    
    dic['Bip001 L Thigh'] = {'father':'Bip001 Spine', 'children':['Bip001 L Calf'], 'pos':[11.23972, -1.101868, 6.513383]}
    dic['Bip001 R Thigh'] = {'father':'Bip001 Spine', 'children':['Bip001 R Calf'], 'pos':[11.23971, -1.101831, -6.513384]}
    dic['Bip001 Spine1']  = {'father':'Bip001 Spine', 'children':['Bip001 Spine2'], 'pos':[-12.52701, -0.008990765, -2.464731e-08]}
    
    dic['Bip001 L Calf'] = {'father':'Bip001 L Thigh', 'children':['Bip001 L Foot'], 'pos':[-32.66669, 1.192093e-07, 0]}
    dic['Bip001 R Calf'] = {'father':'Bip001 R Thigh', 'children':['Bip001 R Foot'], 'pos':[-32.66669, 0, 0]}
    dic['Bip001 Spine2']  = {'father':'Bip001 Spine1', 'children':['Bip001 Neck'],   'pos':[-11.69302, -0.004831314, -1.340959e-08]}
    
    dic['Bip001 L Foot'] = {'father':'Bip001 L Calf', 'children':['Bip001 L Toe0'], 'pos':[-36.31458, -9.536743e-07, 0]}
    dic['Bip001 R Foot'] = {'father':'Bip001 R Calf', 'children':['Bip001 R Toe0'], 'pos':[-36.31458, -4.768372e-07, -4.768372e-07]}
    dic['Bip001 Neck']  = {'father':'Bip001 Spine2', 'children':['Bip001 Head', 'Bip001 L Clavicle', 'Bip001 R Clavicle'],   'pos':[-7.388115, -2.221376, 6.07915e-06]}
    
    dic['Bip001 L Toe0'] = {'father':'Bip001 L Foot', 'children':[], 'pos':[-8.059542, 8.277123, 0]}
    dic['Bip001 R Toe0'] = {'father':'Bip001 R Foot', 'children':[], 'pos':[-8.059542, 8.277124, -4.768372e-07]}
    dic['Bip001 Head']  = {'father':'Bip001 Neck', 'children':[], 'pos':[-5.895767, 0.09526443, 0.0003307685]}
    dic['Bip001 L Clavicle']  = {'father':'Bip001 Neck', 'children':['Bip001 L UpperArm'], 'pos':[4.509056, 1.610937, 5.190379]}
    dic['Bip001 R Clavicle']  = {'father':'Bip001 Neck', 'children':['Bip001 R UpperArm'], 'pos':[4.509048, 1.610966, -5.190396]}
    
    dic['Bip001 L UpperArm'] = {'father':'Bip001 L Clavicle', 'children':['Bip001 L Forearm'], 'pos':[-5.375591, 0, 7.629395e-06]}
    dic['Bip001 R UpperArm'] = {'father':'Bip001 R Clavicle', 'children':['Bip001 R Forearm'], 'pos':[-5.37559,  0, 0]}
    
    dic['Bip001 L Forearm'] = {'father':'Bip001 L UpperArm', 'children':['Bip001 L Hand', 'Bip001 L ForeTwist'], 'pos':[-18.61573, -9.536743e-07, -7.629395e-06]}
    dic['Bip001 R Forearm'] = {'father':'Bip001 R UpperArm', 'children':['Bip001 R Hand', 'Bip001 R ForeTwist'], 'pos':[-18.61573,  0, 0]}
    
    dic['Bip001 L Hand']      = {'father':'Bip001 L Forearm', 'children':['Bip001 L Finger0', 'Bip001 L Finger1', 'Bip001 L Finger2', 'Bip001 L Finger3', 'Bip001 L Finger4'], 'pos':[-18.90229, 0, 0]}
    dic['Bip001 L ForeTwist'] = {'father':'Bip001 L Forearm', 'children':['Bip001 L ForeTwist1'], 'pos':[0, 0, 1.0252e-05]}
    dic['Bip001 R Hand']      = {'father':'Bip001 R Forearm', 'children':['Bip001 R Finger0', 'Bip001 R Finger1', 'Bip001 R Finger2', 'Bip001 R Finger3', 'Bip001 R Finger4'], 'pos':[-18.9023, -1.907349e-06, 0]}
    dic['Bip001 R ForeTwist'] = {'father':'Bip001 R Forearm', 'children':['Bip001 R ForeTwist1'], 'pos':[-1.084655e-05,  0, -1.28746e-05]}
    
    dic['Bip001 L Finger0']    = {'father':'Bip001 L Hand', 'children':['Bip001 L Finger01'], 'pos':[-1.719231, 0.5651627, -1.910677]}
    dic['Bip001 L Finger1']    = {'father':'Bip001 L Hand', 'children':['Bip001 L Finger11'], 'pos':[-5.540304, -0.2278214,-1.871119]}
    dic['Bip001 L Finger2']    = {'father':'Bip001 L Hand', 'children':['Bip001 L Finger21'], 'pos':[-5.579481, -0.4572525,-0.4858208]}
    dic['Bip001 L Finger3']    = {'father':'Bip001 L Hand', 'children':['Bip001 L Finger31'], 'pos':[-5.428627, -0.4348068,0.7903614]}
    dic['Bip001 L Finger4']    = {'father':'Bip001 L Hand', 'children':['Bip001 L Finger41'], 'pos':[-5.161129, -0.2240829,1.925165]}
    dic['Bip001 L ForeTwist1'] = {'father':'Bip001 L ForeTwist', 'children':[], 'pos':[-9.451145, 0, 0]}
    dic['Bip001 R Finger0']    = {'father':'Bip001 R Hand', 'children':['Bip001 R Finger01'], 'pos':[-1.719234, 0.5651703, 1.910675]}
    dic['Bip001 R Finger1']    = {'father':'Bip001 R Hand', 'children':['Bip001 R Finger11'], 'pos':[-5.540302, -0.2278137,1.871119]}
    dic['Bip001 R Finger2']    = {'father':'Bip001 R Hand', 'children':['Bip001 R Finger21'], 'pos':[-5.579479, -0.4572525,0.4858189]}
    dic['Bip001 R Finger3']    = {'father':'Bip001 R Hand', 'children':['Bip001 R Finger31'], 'pos':[-5.428623, -0.4348068,-0.7903633]}
    dic['Bip001 R Finger4']    = {'father':'Bip001 R Hand', 'children':['Bip001 R Finger41'], 'pos':[-5.161131, -0.2240753,-1.925165]}
    dic['Bip001 R ForeTwist1'] = {'father':'Bip001 R ForeTwist', 'children':[], 'pos':[-9.451149, 0, 0]}
    
    dic['Bip001 L Finger01']    = {'father':'Bip001 L Finger0', 'children':['Bip001 L Finger02'], 'pos':[-2.294796, -7.629395e-06, 0]}
    dic['Bip001 L Finger11']    = {'father':'Bip001 L Finger1', 'children':['Bip001 L Finger12'], 'pos':[-2.618149, 0,-1.907349e-06]}
    dic['Bip001 L Finger21']    = {'father':'Bip001 L Finger2', 'children':['Bip001 L Finger22'], 'pos':[-3.094715, 0,-1.907349e-06]}
    dic['Bip001 L Finger31']    = {'father':'Bip001 L Finger3', 'children':['Bip001 L Finger32'], 'pos':[-2.827309, 7.629395e-06,0]}
    dic['Bip001 L Finger41']    = {'father':'Bip001 L Finger4', 'children':['Bip001 L Finger42'], 'pos':[-2.183788, 0,1.907349e-06]}
    dic['Bip001 R Finger01']    = {'father':'Bip001 R Finger0', 'children':['Bip001 R Finger02'], 'pos':[-2.294792, -3.814697e-06, 0]}
    dic['Bip001 R Finger11']    = {'father':'Bip001 R Finger1', 'children':['Bip001 R Finger12'], 'pos':[-2.618153, -7.629395e-06,-1.430511e-06]}
    dic['Bip001 R Finger21']    = {'father':'Bip001 R Finger2', 'children':['Bip001 R Finger22'], 'pos':[-3.094719, 0,0]}
    dic['Bip001 R Finger31']    = {'father':'Bip001 R Finger3', 'children':['Bip001 R Finger32'], 'pos':[-2.827312, -7.629395e-06,0]}
    dic['Bip001 R Finger41']    = {'father':'Bip001 R Finger4', 'children':['Bip001 R Finger42'], 'pos':[-2.183777, -7.629395e-06,-1.907349e-06]}
    
    dic['Bip001 L Finger02']    = {'father':'Bip001 L Finger01', 'children':[], 'pos':[-1.90498, 3.814697e-06, 3.814697e-06]}
    dic['Bip001 L Finger12']    = {'father':'Bip001 L Finger11', 'children':[], 'pos':[-1.357204, 0,4.768372e-07]}
    dic['Bip001 L Finger22']    = {'father':'Bip001 L Finger21', 'children':[], 'pos':[-1.688278, 7.629395e-06,1.907349e-06]}
    dic['Bip001 L Finger32']    = {'father':'Bip001 L Finger31', 'children':[], 'pos':[-1.569027, -3.814697e-06,0]}
    dic['Bip001 L Finger42']    = {'father':'Bip001 L Finger41', 'children':[], 'pos':[-1.283112, -7.629395e-06,-1.907349e-06]}
    dic['Bip001 R Finger02']    = {'father':'Bip001 R Finger01', 'children':[], 'pos':[-1.904972, 0, 3.814697e-06]}
    dic['Bip001 R Finger12']    = {'father':'Bip001 R Finger11', 'children':[], 'pos':[-1.357208, 0, 9.536743e-07]}
    dic['Bip001 R Finger22']    = {'father':'Bip001 R Finger21', 'children':[], 'pos':[-1.688263, -7.629395e-06,1.907349e-06]}
    dic['Bip001 R Finger32']    = {'father':'Bip001 R Finger31', 'children':[], 'pos':[-1.569023, 0,1.907349e-06]}
    dic['Bip001 R Finger42']    = {'father':'Bip001 R Finger41', 'children':[], 'pos':[-1.283119, 3.814697e-06,0]}
    dfs(dic, 'Bip001') #list to array
    return dic

def getM(x):
    #last = torch.zeros((3,3)).cuda()
    theta = (x ** 2).sum()**0.5
    r = x / theta
    last = torch.FloatTensor([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
    return (cos(theta) * torch.eye(3) + (1-cos(theta)) * r.reshape(3,1).mm(r.reshape(1,3)) + sin(theta)*last).t()
    
def relate2abs_ori(topoTree, dic, node, plst, mlst, M=None, P=None):
    #absPos = np.array([0,0,0])
    #import pdb
    #pdb.set_trace()
    #pos = topoTree[node]['pos']
    P1  = topoTree[node]['pos']
    if type(P)!=type(None):
        P = M.mm(torch.from_numpy(P1).float().reshape(3,1)) + P
    else:
        P = torch.from_numpy(P1).float().reshape(3,1)
    M1  = getM(torch.FloatTensor(quart2Rotationvector(*dic[node]['quaternion'])))
    #print('M1:',M1)
    if type(M)!=type(None):
        M = torch.mm(M,M1)
    else:
        M = M1
    #pdb.set_trace()
    for i in range(len(plst)-1,-1,-1):
        tM  = mlst[i]
        tP  = plst[i]
        #pdb.set_trace()
        pos = np.dot(tM, pos) + tP
    #absPos = pos
    if (P.reshape(1,3).numpy() - pos).mean() > 0.0001:
        print('error.')
        pdb.set_trace()
    #print('pos:',pos,P)
    topoTree[node]['absPos'] = pos
    for child in topoTree[node]['children']:
        plst.append(topoTree[node]['pos'])
        mlst.append(quaternion2Matrix(dic[node]['quaternion']))
        relate2abs(topoTree, dic, child, plst, mlst, M, P)
        plst.pop()
        mlst.pop()

def relate2abs(topoTree, dic, node, M=None, P=None): #记录向下传,O(n)
    P1  = topoTree[node]['pos']
    if type(P)!=type(None):
        P = M.mm(torch.from_numpy(P1).float().reshape(3,1)) + P
    else:
        P = torch.from_numpy(P1).float().reshape(3,1)
    #vec = quart2Rotationvector(*dic[node]['quaternion'])
    #pdb.set_trace()
    ''' #四元数[0,0,-5e-6,1.0] => 轴角[0,0,0] 导致后面出现nan
    for i in range(3):
        if math.isnan(vec[i,0]):
            pdb.set_trace()
            print('????????????')
    '''
    M1 = quaternion2Matrix(dic[node]['quaternion'])
    #M1  = getM(torch.FloatTensor(vec))
    for i in range(3):
        for j in range(3):
            #pdb.set_trace()
            if math.isnan(M1[i,j].item()):
                pdb.set_trace()
                print('????????????')
    ske = ['Bip001 L Foot','Bip001 L Calf','Bip001 L Thigh', \
    'Bip001 L Hand','Bip001 L Forearm','Bip001 L UpperArm','Bip001 L Clavicle',\
    'Bip001 Neck','Bip001 Spine2','Bip001 Spine1','Bip001 Spine',\
    'Bip001 R Clavicle', 'Bip001 R UpperArm', 'Bip001 R Forearm', 'Bip001 R Hand',\
    'Bip001 R Thigh', 'Bip001 R Calf', 'Bip001 R Foot'
    ]
    '''
    for j in range(len(ske)):
        if node == ske[j]:
            z[0,j*3] = vec[0]
            z[0,j*3+1] = vec[1]
            z[0,j*3+2] = vec[2]
    '''
    #print(z)
    #pdb.set_trace()
    #quart2Rotationvector(*dic[node]['quaternion'])
    #print('M1:',M1)
    if type(M)!=type(None):
        M = torch.mm(M,M1)
    else:
        M = M1
    topoTree[node]['absPos'] = P.reshape(3).numpy().tolist()
    for child in topoTree[node]['children']:
        relate2abs(topoTree, dic, child, M, P)


def quaternion2Matrix(quaternion):
    #import pdb
    #pdb.set_trace()
    #这样求出来是左乘旋转矩阵，要转置
    #证明四元数转轴角的函数正确，而且用下面的公式从轴角到旋转矩阵的转换正确。
    '''下面是验证轴角转旋转矩阵的，与四元数相同
    v = quart2Rotationvector(*quaternion)
    v = np.array(v)
    theta = (v ** 2).sum() ** 0.5
    r = v / theta
    I = np.eye(3)
    last = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]]).astype('float')
    R = cos(theta) * I + (1-cos(theta))*r.dot(r.reshape(1,3)) + sin(theta) * last
    R = R.T
    x,y,z,w = quaternion[0],quaternion[1],quaternion[2],quaternion[3]
    
    M = [
        [cos(w)+(1-cos(w))*x*x, (1-cos(w))*x*y-sin(w)*z, (1-cos(w))*x*z+sin(w)*y],
        [(1-cos(w))*y*x+sin(w)*z, cos(w)+(1-cos(w))*y*y, (1-cos(w))*y*z-sin(w)*x],
        [(1-cos(w))*z*x-sin(w)*y, (1-cos(w))*z*y+sin(w)*x, cos(w)+(1-cos(w))*z*z],
    ]
    '''
    #右乘的旋转矩阵，要旋转的列向量在右侧
    #取转置就是左乘旋转矩阵
    x,y,z,w = quaternion[0],quaternion[1],quaternion[2],quaternion[3] #unity 四元数格式(x,y,z,w)
    M = [
        [w*w+x*x-y*y-z*z,2*(x*y-w*z),2*(w*y+x*z)],
        [2*(x*y+w*z),w*w+y*y-x*x-z*z,2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),w*w+z*z-x*x-y*y],
    ]
    '''
    M = [
        [1-2*y*y-2*z*z, 2*x*y+2*w*z,2*x*z-2*w*y],
        [2*x*y-2*w*z,1-2*x*x-2*z*z,2*y*z+2*w*x],
        [2*x*z+2*w*y,2*y*z-2*w*x,1-2*x*x-2*y*y]
    ]
    '''
    M = torch.from_numpy(np.array(M)).float()
    #xx = (M - R).mean()
    #print(xx)
    return M

def move(topoTree, node, dtp):
    for child in topoTree[node]['children']:
        topoTree[child]['pos'] = topoTree[child]['pos'] + dtp
        move(topoTree, child, dtp)

def rotate(topoTree, node, M):
    for child in topoTree[node]['children']:
        topoTree[child]['pos'] = np.matmul(M, topoTree[child]['pos'])
        rotate(topoTree, child, M)

def updatedfs(topoTree, node, dic):
    for child in topoTree[node]['children']:
        updatedfs(topoTree, child, dic)
    quaternion = dic[node]['quaternion']
    M = quaternion2Matrix(quaternion) #numpy M.shape=3*3
    move(topoTree, node, -topoTree[node]['pos']) #移动node的子树，不包含它自身
    rotate(topoTree, node, M) #旋转node的子树，不包含它自身
    move(topoTree, node, topoTree[node]['pos'])

def updateTreeDFS(dic):
    topoTree = getTopoTree()
    #q = queue.deque()
    #q.append('Bip001')
    #topoTree['Bip001']['pos'] = np.array(dic['Bip001']['pos']) #所有运动的Bip001坐标都变成0,0,0开始
    #z = np.zeros((1,64))
    
    relate2abs(topoTree, dic, 'Bip001 Spine')#把Bip001开始的整棵树节点的坐标转换为绝对坐标
    #print(z)
    #np.savetxt('test.txt',z)
    #pdb.set_trace()
    #updatedfs(topoTree, 'Bip001', dic) #递归旋转Bip001的子树,用世界坐标从叶子节点开始，逐层向上
    '''
    while q.__len__() != 0:
        node = q.pop()
        update(topoTree, node, dic[node]['quaternion']) #更新node的子树，旋转和平移
        for child in topoTree[node]['children']:
            q.append(child)
    '''
    return topoTree

def getStateDic(state):
    #pdb.set_trace()
    lst = state.split(',')
    dic = {}
    dic[lst[0]]={}
    dic[lst[0]]['pos'] = [float(lst[1]), float(lst[2]), float(lst[3])]
    dic[lst[0]]['quaternion'] = [-0.511119,-0.451622,-0.487246,0.545330]
    #pdb.set_trace()
    for i in range(8,len(lst),5):
        dic[lst[i]] = {}
        dic[lst[i]]['quaternion'] = [float(lst[i+1]), float(lst[i+2]), float(lst[i+3]), float(lst[i+4])]
    return dic

def updateState(state):
    dic = getStateDic(state)
    return updateTreeDFS(dic), dic

def l2(now):
    tmp = 0
    for i in range(3):
        tmp += now[i]**2
    return tmp**0.5

def check(t1, t2, node):
    #pdb.set_trace()
    for child in t1[node]['children']:
        tmp1 = t1[node]['pos'] - t1[child]['pos']
        tmp2 = np.array(t2[child]['pos']).astype('float')
        print(str(t1[node]['pos']), str(t1[child]['pos']))
        if abs(l2(tmp1) - l2(tmp2)) > 0.001:
            pdb.set_trace()
        print(l2(tmp1),l2(tmp2))
        check(t1, t2, child)
        
def getPos():
    ske = ['Bip001', 'Bip001 Footsteps', 'Bip001 Pelvis', 'Bip001 Spine', 'Bip001 Spine1', 'Bip001 Spine2', 'Bip001 Neck', 'Bip001 L Clavicle', 'Bip001 L UpperArm', 'Bip001 L Forearm', 'Bip001 L Hand', 'Bip001 L Finger0', 'Bip001 L Finger01', 'Bip001 L Finger02', 'Bip001 L Finger0Nub', 'Bip001 L Finger1', 'Bip001 L Finger11', 'Bip001 L Finger12', 'Bip001 L Finger1Nub', 'Bip001 L Finger2', 'Bip001 L Finger21', 'Bip001 L Finger22', 'Bip001 L Finger2Nub', 'Bip001 L Finger3', 'Bip001 L Finger31', 'Bip001 L Finger32', 'Bip001 L Finger3Nub', 'Bip001 L Finger4', 'Bip001 L Finger41', 'Bip001 L Finger42', 'Bip001 L Finger4Nub', 'Bip001 L ForeTwist', 'Bip001 L ForeTwist1', 'Bip001 R Clavicle', 'Bip001 R UpperArm', 'Bip001 R Forearm', 'Bip001 R Hand', 'Bip001 R Finger0', 'Bip001 R Finger01', 'Bip001 R Finger02', 'Bip001 R Finger0Nub', 'Bip001 R Finger1', 'Bip001 R Finger11', 'Bip001 R Finger12', 'Bip001 R Finger1Nub', 'Bip001 R Finger2', 'Bip001 R Finger21', 'Bip001 R Finger22', 'Bip001 R Finger2Nub', 'Bip001 R Finger3', 'Bip001 R Finger31', 'Bip001 R Finger32', 'Bip001 R Finger3Nub', 'Bip001 R Finger4', 'Bip001 R Finger41', 'Bip001 R Finger42', 'Bip001 R Finger4Nub', 'Bip001 R ForeTwist', 'Bip001 R ForeTwist1', 'Bip001 Head', 'Bip001 HeadNub', 'Bip001 L Thigh', 'Bip001 L Calf', 'Bip001 L Foot', 'Bip001 L Toe0', 'Bip001 L Toe0Nub', 'Bip001 R Thigh', 'Bip001 R Calf', 'Bip001 R Foot', 'Bip001 R Toe0', 'Bip001 R Toe0Nub']
    #ske = ['Bip001 Spine', 'Bip001 Spine1', 'Bip001 Spine2', 'Bip001 Neck', 'Bip001 L Clavicle', 'Bip001 L UpperArm', 'Bip001 L Forearm', 'Bip001 L Hand', 'Bip001 L Finger0', 'Bip001 L Finger01', 'Bip001 L Finger02', 'Bip001 L Finger0Nub', 'Bip001 L Finger1', 'Bip001 L Finger11', 'Bip001 L Finger12', 'Bip001 L Finger1Nub', 'Bip001 L Finger2', 'Bip001 L Finger21', 'Bip001 L Finger22', 'Bip001 L Finger2Nub', 'Bip001 L Finger3', 'Bip001 L Finger31', 'Bip001 L Finger32', 'Bip001 L Finger3Nub', 'Bip001 L Finger4', 'Bip001 L Finger41', 'Bip001 L Finger42', 'Bip001 L Finger4Nub', 'Bip001 L ForeTwist', 'Bip001 L ForeTwist1', 'Bip001 R Clavicle', 'Bip001 R UpperArm', 'Bip001 R Forearm', 'Bip001 R Hand', 'Bip001 R Finger0', 'Bip001 R Finger01', 'Bip001 R Finger02', 'Bip001 R Finger0Nub', 'Bip001 R Finger1', 'Bip001 R Finger11', 'Bip001 R Finger12', 'Bip001 R Finger1Nub', 'Bip001 R Finger2', 'Bip001 R Finger21', 'Bip001 R Finger22', 'Bip001 R Finger2Nub', 'Bip001 R Finger3', 'Bip001 R Finger31', 'Bip001 R Finger32', 'Bip001 R Finger3Nub', 'Bip001 R Finger4', 'Bip001 R Finger41', 'Bip001 R Finger42', 'Bip001 R Finger4Nub', 'Bip001 R ForeTwist', 'Bip001 R ForeTwist1', 'Bip001 Head', 'Bip001 HeadNub', 'Bip001 L Thigh', 'Bip001 L Calf', 'Bip001 L Foot', 'Bip001 L Toe0', 'Bip001 L Toe0Nub', 'Bip001 R Thigh', 'Bip001 R Calf', 'Bip001 R Foot', 'Bip001 R Toe0', 'Bip001 R Toe0Nub']
    
    f = open(r'D:\all_data\1102\nlp_005\211_1_skeleton.txt','r',encoding='utf-8-sig')
    lines = f.readlines()
    f.close()
    f = open('test1.txt','w+',encoding='utf-8-sig')
    for line in lines:
        line = line.strip().replace('\n','')[:-1]
        topoTree = updateState(line)
        #pdb.set_trace()
        oriTree  = getTopoTree()
        #check(topoTree, oriTree, 'Bip001') #验证Bip001与其儿子节点的相对位置
        res = ''
        #pdb.set_trace()
        topoTree = topoTree[0]
        for s in ske:
            res = res + s + ','
            #pdb.set_trace()
            
            if s in topoTree:
                if s == 'Bip001' or s == 'Bip001 Pelvis':
                    res = res + '0.0,0.0,0.0,-0.511119,-0.451622,-0.487246,0.545330,'
                else:
                    for j in range(3):
                        #if s == 'Bip001':
                        #    res = res + '0,'
                        #else:
                        res = res + str(topoTree[s]['absPos'][j]) + ','
                    for j in range(3):
                        res = res + '0.0,'
            else:
                for j in range(6):
                    res = res + '0.00,'
        #pdb.set_trace()
        f.write(res+'\n')
        #pdb.set_trace()
        print('one line done.')
    f.close()

if __name__ == '__main__':
    getPos()