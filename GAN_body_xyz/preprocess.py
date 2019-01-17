import pdb
import os
from getAbsPos import updateState 
from quart2rotation import quart2Rotationvector
ske_name = ['Bip001 Spine', 'Bip001 Spine1', 'Bip001 Spine2', 'Bip001 Neck', 'Bip001 Head', \
    'Bip001 L Clavicle', 'Bip001 L UpperArm', 'Bip001 L Forearm', 'Bip001 L Hand', 'Bip001 L Finger0', 'Bip001 L Finger1', 'Bip001 L Finger2', 'Bip001 L Finger3','Bip001 L Finger4', \
    'Bip001 R Clavicle', 'Bip001 R UpperArm', 'Bip001 R Forearm', 'Bip001 R Hand', 'Bip001 R Finger0', 'Bip001 R Finger1', 'Bip001 R Finger2', 'Bip001 R Finger3','Bip001 R Finger4',\
    'Bip001 L Thigh','Bip001 L Calf','Bip001 L Foot','Bip001 L Toe0',\
    'Bip001 R Thigh','Bip001 R Calf','Bip001 R Foot','Bip001 R Toe0'] #31

minx = miny = minz = 1000
maxx = maxy = maxz = -1000
def write2file(dic, ddic, f):
    #pdb.set_trace()
    global minx,miny,minz,maxx,maxy,maxz
    res = ''
    for ske in ske_name:
        if ske not in dic:
            print('error.')
            pdb.set_trace()
        else:
            for i in range(3):
                res = res + str(dic[ske]['absPos'][i]) + '_'
    #pdb.set_trace()
    f.write(res[:-1]+' ') #末尾'_' => ' '

def getSkeleton(root_dir, out_path, frameN):
    
    dlst = os.listdir(root_dir)
    fw = open(out_path, 'w+', encoding='utf-8-sig')
    #pdb.set_trace()
    l1 = 0
    for d in dlst:
        dpath = os.path.join(root_dir, d)
        if os.path.isfile(dpath):
            continue
        ddlst = os.listdir(dpath)
        for dd in ddlst:
            ddpath = os.path.join(dpath, dd)
            if os.path.isfile(ddpath):
                continue
            filelst = os.listdir(ddpath)
            for name in filelst:
                if name.find('skeleton.txt') < 0:
                    continue
                fname = os.path.join(ddpath, name)
                print(fname)
                f =  open(fname, 'r', encoding='utf-8-sig')
                lines = f.readlines()
                if len(lines) < frameN:
                    continue
                #pdb.set_trace()
                last = len(lines)
                cot  = 0 
                for line in lines:
                    line = line.strip()[:-1]
                    print('line:',l1)
                    #if l1 == 10:
                    #    pdb.set_trace()
                    #    print('error line.')
                    res, dic = updateState(line)
                    write2file(res, dic, fw)
                    cot += 1
                    #print(minx,maxx,miny,maxy,minz,maxz)
                    if cot >= frameN:
                        fw.write('\n')
                        #pdb.set_trace()
                        
                        l1 += 1
                        cot = 0
                        last -= frameN
                        if last < frameN:
                            break
                
    #print(minx,maxx,miny,maxy,minz,maxz)

if __name__ == '__main__':
    root_dir = r'D:\all_data'
    out_path = './data/train_new_test.txt'
    getSkeleton(root_dir, out_path, 64)