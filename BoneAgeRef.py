import cv2
import os

os.mkdir('C:/Users/vuno/Desktop/BoneAgeRef/saved/')
path_saved = 'C:/Users/vuno/Desktop/BoneAgeRef/saved/'

i = 0
for i in range(1,32):
    boneage = cv2.imread('{}.PNG'.format(i), cv2.IMREAD_UNCHANGED)
    boneage2 = cv2.resize(boneage, dsize=(1080, 1350), interpolation=cv2.INTER_LINEAR)
    
    name = "male_{}.jpg".format(i)
    cv2.imwrite(os.path.join(path_saved, name), boneage2)
    boneage = i+1
print('male done')

j = 0
for j in range(33,59):
    boneage = cv2.imread('{}.PNG'.format(j), cv2.IMREAD_UNCHANGED)
    boneage2 = cv2.resize(boneage, dsize=(1080, 1350), interpolation=cv2.INTER_LINEAR)
    
    name = "female_{}.jpg".format(j)
    cv2.imwrite(os.path.join(path_saved, name), boneage2)
    boneage = j+1
print('female done')