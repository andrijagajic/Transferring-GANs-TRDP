# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:23:20 2019

@author: andrija
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:05:10 2019

@author: andrija
"""
import pickle
import numpy as np

with open("activations/Bordeaux/Places_source/places/activations.txt", "rb") as fp:   # Unpickling
    res_plc1 = pickle.load(fp)

filts1 = np.mean(res_plc1[0],0)
#filts1 = np.mean(filts1,1)
#filts1 = np.mean(filts1,1)
filts1 = np.reshape(filts1,(filts1.shape[0]*filts1.shape[1]*filts1.shape[2],))

del res_plc1

with open("activations/Bordeaux/Bedroom_source/Bedrooms/activations.txt", "rb") as fp:   # Unpickling
    res_bed1 = pickle.load(fp)

filts3 = np.mean(res_bed1[0],0)
#filts3 = np.mean(filts3,1)
#filts3 = np.mean(filts3,1)
filts3 = np.reshape(filts3,(filts3.shape[0]*filts3.shape[1]*filts3.shape[2],))

del res_bed1

with open("activations/Bordeaux/celebA_source/celebA/activations.txt", "rb") as fp:   # Unpickling
    res_cel1 = pickle.load(fp)
    
filts5 = np.mean(res_cel1[0],0)
#filts5 = np.mean(filts5,1)
#filts5 = np.mean(filts5,1)
filts5 = np.reshape(filts5,(filts5.shape[0]*filts5.shape[1]*filts5.shape[2],))


del res_cel1

with open("activations/Bordeaux/Imagenet_source/imagenet/activations.txt", "rb") as fp:   # Unpickling
    res_img1 = pickle.load(fp)    

filts7 = np.mean(res_img1[0],0)
#filts7 = np.mean(filts7,1)
#filts7 = np.mean(filts7,1)
filts7 = np.reshape(filts7,(filts7.shape[0]*filts7.shape[1]*filts7.shape[2],))


del res_img1

with open("activations/Bordeaux/Places_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    res_plc2 = pickle.load(fp)

filts2 = np.mean(res_plc2[0],0)
#filts2 = np.mean(filts2,1)
#filts2 = np.mean(filts2,1)
filts2 = np.reshape(filts2,(filts2.shape[0]*filts2.shape[1]*filts2.shape[2],))


del res_plc2

with open("activations/Bordeaux/Bedroom_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    res_bed2 = pickle.load(fp)
    
filts4 = np.mean(res_bed2[0],0)
#filts4 = np.mean(filts4,1)
#filts4 = np.mean(filts4,1)
filts4 = np.reshape(filts4,(filts4.shape[0]*filts4.shape[1]*filts4.shape[2],))


del res_bed2

with open("activations/Bordeaux/celebA_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    res_cel2 = pickle.load(fp)

filts6 = np.mean(res_cel2[0],0)
#filts6 = np.mean(filts6,1)
#filts6 = np.mean(filts6,1)
filts6 = np.reshape(filts6,(filts6.shape[0]*filts6.shape[1]*filts6.shape[2],))


del res_cel2

with open("activations/Bordeaux/Imagenet_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    res_img2 = pickle.load(fp)    

filts8 = np.mean(res_img2[0],0)
#filts8 = np.mean(filts8,1)
#filts8 = np.mean(filts8,1)
filts8 = np.reshape(filts8,(filts8.shape[0]*filts8.shape[1]*filts8.shape[2],))


del res_img2

feat_num = 100


#filts2 = filts2[np.argsort(filts1)[-1000:]]
#filts1 = filts1[np.argsort(filts1)[-1000:]]

#filts_diff = np.abs(filts1-filts2)
filts_diff = np.abs(filts1-filts2)


idx = np.argsort(filts_diff)[::-1]
idx = idx[0:feat_num]

filts1_red = filts1[idx]
filts2_red = filts2[idx]

a = np.mean(filts1_red)
b = np.mean(filts2_red)
diff_plc = a - b

filts_diff = filts_diff[idx]
diff_plc = np.mean(filts_diff)


filts4 = filts4[np.argsort(filts3)[-1000:]]
filts3 = filts3[np.argsort(filts3)[-1000:]]
filts_diff = np.abs(filts3-filts4)

idx = np.argsort(filts_diff)[::-1]
idx = idx[0:feat_num]

filts3_red = filts3[idx]
filts4_red = filts4[idx]

c = np.mean(filts3_red)
d = np.mean(filts4_red)
diff_bed = c - d

filts_diff = filts_diff[idx]
diff_bed = np.mean(filts_diff)


filts6 = filts6[np.argsort(filts5)[-1000:]]
filts5 = filts5[np.argsort(filts5)[-1000:]]
filts_diff = np.abs(filts5-filts6)

idx = np.argsort(filts_diff)[::-1]
idx = idx[0:feat_num]

filts5_red = filts5[idx]
filts6_red = filts6[idx]

e = np.mean(filts5_red)
f = np.mean(filts6_red)

diff_cel = e - f

filts_diff = filts_diff[idx]
diff_cel = np.mean(filts_diff)


filts8 = filts8[np.argsort(filts7)[-1000:]]
filts7 = filts7[np.argsort(filts7)[-1000:]]
filts_diff = np.abs(filts7-filts8)

idx = np.argsort(filts_diff)[::-1]
idx = idx[0:feat_num]

filts7_red = filts7[idx]
filts8_red = filts8[idx]

g = np.mean(filts7_red)
h = np.mean(filts8_red)

diff_img = g - h

filts_diff = filts_diff[idx]
diff_img = np.mean(filts_diff)