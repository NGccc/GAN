# -*- coding: utf-8 -*-
import numpy as np
from torch.utils import data
import pdb

class MyDataset(data.Dataset):
    def __init__(self, train=True, transform=None):
        #import pdb
        #pdb.set_trace()
        self.minx = -81.36
        self.maxx = 79.6
        self.miny = -45.0
        self.maxy = 72.1
        self.minz = -55.1
        self.maxz = 53.8
        self.cot  = 0
        self.data    = './data'
        self.train   = train
        self.maxlen  = 64
        with open('%s/train_new_test.txt' % self.data, 'r', encoding='utf-8-sig') as f:
            self.train_labelsk = f.readlines()
        self.train_length = len(self.train_labelsk)

    def __getitem__(self, index):
        #pdb.set_trace()
        if self.train:
            labelsk_lst = self.train_labelsk
        
        labelsk = labelsk_lst[index].replace('\n','').strip().split(' ')
        labelsk_lst = []
        for la in labelsk:
            labelsk_lst.append(la.split('_'))
        #res = np.zeros((64,93)).astype('float')
        res = np.array(labelsk_lst).astype('float')
        '''
        for i in range(93):
            if i % 3 == 0:
                res[:,i] = ((res[:,i] - self.minx) / (self.maxx - self.minx) - 0.5) * 2 
            elif i % 3 == 1:
                res[:,i] = ((res[:,i] - self.miny) / (self.maxy - self.miny) - 0.5) * 2
            else:
                res[:,i] = ((res[:,i] - self.minz) / (self.maxz - self.minz) - 0.5) * 2
        '''
        return res

    def __len__(self):
        if self.train:
            return self.train_length
        else:
            return self.val_length