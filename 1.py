'''
先删除 txt  和 result 里面的残余文件 
'''

import os
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def xywh2x1y1w1h1(logpath,train_save,val_save,label_save):
    '''
    转化为 yolov3 坐标
    xywh 为顶点坐标，为像素值
    x1y1w1h1 为归一化坐标
    '''
    f = open(logpath,"r")   #设置文件对象
    data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
    f.close()  
    # data = data[:80000]  
    # ?+   data[72375:]         #   原来的数据格式为  72375  ，
    print(len(data))
    label = []
    filelist = []
    for file in data:
    #    try :   
                filecopy = file
            #    print(i)
                file = file.replace('\n','')
                bbox  = file.split(',')
                kinds = bbox[1]
                bbox[4] = int(file.split(',')[2]) +  int(file.split(',')[4])
                bbox[5] = int(file.split(',')[3]) +  int(file.split(',')[5])
                # name = filecopy.split(',')[0]
                name = filecopy.split(',')[0]
                box = (int(bbox[2]),bbox[4],int(bbox[3]),bbox[5])
                # print('old is :',file)
                # print('new is :',box)
                # print(box)
            #    if  110105026-5789-5885-00001084_nobody-51 in 
                # img = cv2.imread(name)
                img = Image.open(os.path.join('result_3',name))
                # print(img.size)
                w, h = img.size
                text = convert((w,h), box)
                Flag = True
                for i in text:
                    if i <=0:
                        Flag = False
                if Flag == False:
                    print(name  , '        error')
                    continue
                # print(text)
                
                filename = name
                label.append([filename,text,kinds])
                filelist.append('data/custom/images/'+filename )
                # print([filename,text],'\n')
                # print('data/custom/images/'+filename,'\n')
                
    filelist = list(set(filelist))
    train_X, test_X = train_test_split(filelist, test_size=0.05, random_state=0,shuffle=True)
    # print(train_X)
    # print(test_X)
    
    train_file = open(train_save, 'w')
    for x in train_X:
        x =  x.replace('.txt','.jpg')
        # print(x)
        train_file.write(x+'\n')
    train_file.close()
    test_file = open(val_save, 'w')
    for x in test_X:
        x = x.replace('.txt','.jpg')
        test_file.write(x+'\n')
    test_file.close()

    for bb in   label:
        with open(os.path.join(label_save,bb[0].replace('.jpg','.txt')),'a+') as f:
            x = bb[2] + ' ' + " ".join([str(a) for a in bb[1]]) + '\n'
            f.write(x)


if __name__ == '__main__':
    logpath = "log_3.txt"
    train_save = 'train_label/train_3.txt'
    val_save = 'train_label/valid_3.txt'
    label_save = ''
    xywh2x1y1w1h1(logpath,train_save,val_save,label_save)