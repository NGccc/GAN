import numpy as np
import cv2
import pdb

with open('./data/llbs.txt', 'r', encoding='utf-8-sig') as f:
    labelbs = f.readlines()
with open('./data/llskr.txt', 'r', encoding='utf-8-sig') as f:
    labelsk = f.readlines()

def saveImage(data, th):
    #pdb.set_trace()
    data = (data * 255).astype('uint8').squeeze()
    dt  = 10
    out = np.zeros((data.shape[0], data.shape[1] * dt)).astype('uint8')
    for j in range(data.shape[1]):
        for i in range(dt):
            out[:,j*dt+i] = data[:,j]
    cv2.imwrite('./results_GT/GT_%d.png' % th, out)

for index in range(len(labelbs)):
    labelbs1 = labelbs[index].replace('\n','').strip().split(' ')
    labelbs_lst = []
    for la in labelbs1:
        labelbs_lst.append(la.split('_'))

    labelsk1 = labelsk[index].replace('\n','').strip().split(' ')
    labelsk_lst = []
    for la in labelsk1:
        labelsk_lst.append(la.split('_'))
    used = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 58, 59, 60, 61, 62, 63, 64, 65, 69, 70, 71, 72, 73, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 88, 89, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 123, 124, 125, 126, 127, 128, 129, 130, 131, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 148, 149, 150, 151, 152, 153, 154, 155, 159, 160, 161, 162, 163, 164, 165, 166, 167, 171, 172, 173, 174, 175, 176, 177, 178, 179, 183, 184, 185, 188, 189, 190, 191, 198, 199, 200, 203, 204, 205, 206]
    label_1  = (np.array(labelbs_lst).astype('float') / 100.0)#[0,100] to [0,1] to [-1,1]
    label_2  = (np.array(labelsk_lst).astype('float') / 360.0)#[0,360] to [0,1] to [-1,1]      
    label_2  = label_2[:,used]
    label_12 = np.zeros((label_1.shape[0],label_1.shape[1]+label_2.shape[1])).astype('float')
    label_12[:,:48] = label_1
    label_12[:,48:] = label_2
    
    dt = 2
    th = 0
    labelr = np.zeros((128, 256))
    for j in range(label_12.shape[1]):
        if th >= 102: #前51维重复一次，凑256维 48+157=205    + 51 = 256
            break
            dt=1
        for i in range(dt):
            th += 1
            labelr[:,j*dt+i] = label_12[:,j]
    
    th = 51
    for j in range(102,256):
        labelr[:,j] = label_12[:,th]
        th += 1
    
    saveImage(labelr, index)
    #labelr = labelr.astype('uint8')
    #cv2.imwrite('./results48_GT/fake_epoch_%d.png' % index, labelr)