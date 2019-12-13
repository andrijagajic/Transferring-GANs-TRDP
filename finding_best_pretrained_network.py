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


feat_num = 100 # Number of filters with highest absolute differences to be kept

# Extracting activations
with open("activations/Bordeaux/Places_source/places/activations.txt", "rb") as fp:   # Unpickling
    plc_plc = pickle.load(fp)

plc_plc = np.mean(plc_plc[0],0)
plc_plc = np.reshape(plc_plc,(plc_plc.shape[0]*plc_plc.shape[1]*plc_plc.shape[2],))


with open("activations/Bordeaux/Bedroom_source/Bedrooms/activations.txt", "rb") as fp:   # Unpickling
    bed_bed = pickle.load(fp)

bed_bed = np.mean(bed_bed[0],0)
bed_bed = np.reshape(bed_bed,(bed_bed.shape[0]*bed_bed.shape[1]*bed_bed.shape[2],))


with open("activations/Bordeaux/celebA_source/celebA/activations.txt", "rb") as fp:   # Unpickling
    cel_cel = pickle.load(fp)
    
cel_cel = np.mean(cel_cel[0],0)
cel_cel = np.reshape(cel_cel,(cel_cel.shape[0]*cel_cel.shape[1]*cel_cel.shape[2],))


with open("activations/Bordeaux/Imagenet_source/imagenet/activations.txt", "rb") as fp:   # Unpickling
    ima_ima = pickle.load(fp)    

ima_ima = np.mean(ima_ima[0],0)
ima_ima = np.reshape(ima_ima,(ima_ima.shape[0]*ima_ima.shape[1]*ima_ima.shape[2],))



with open("activations/Bordeaux/Places_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    flo_plc = pickle.load(fp)

flo_plc = np.mean(flo_plc[0],0)
flo_plc = np.reshape(flo_plc,(flo_plc.shape[0]*flo_plc.shape[1]*flo_plc.shape[2],))


with open("activations/Bordeaux/Bedroom_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    flo_bed = pickle.load(fp)
    
flo_bed = np.mean(flo_bed[0],0)
flo_bed = np.reshape(flo_bed,(flo_bed.shape[0]*flo_bed.shape[1]*flo_bed.shape[2],))


with open("activations/Bordeaux/celebA_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    flo_cel = pickle.load(fp)

flo_cel = np.mean(flo_cel[0],0)
flo_cel = np.reshape(flo_cel,(flo_cel.shape[0]*flo_cel.shape[1]*flo_cel.shape[2],))


with open("activations/Bordeaux/Imagenet_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    flo_ima = pickle.load(fp)    

flo_ima = np.mean(flo_ima[0],0)
flo_ima = np.reshape(flo_ima,(flo_ima.shape[0]*flo_ima.shape[1]*flo_ima.shape[2],))





with open("activations/Bordeaux/Places_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
    kit_plc = pickle.load(fp)

kit_plc = np.mean(kit_plc[0],0)
kit_plc = np.reshape(kit_plc,(kit_plc.shape[0]*kit_plc.shape[1]*kit_plc.shape[2],))


with open("activations/Bordeaux/Bedroom_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
    kit_bed = pickle.load(fp)
    
kit_bed = np.mean(kit_bed[0],0)
kit_bed = np.reshape(kit_bed,(kit_bed.shape[0]*kit_bed.shape[1]*kit_bed.shape[2],))


with open("activations/Bordeaux/celebA_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
    kit_cel = pickle.load(fp)

kit_cel = np.mean(kit_cel[0],0)
kit_cel = np.reshape(kit_cel,(kit_cel.shape[0]*kit_cel.shape[1]*kit_cel.shape[2],))


with open("activations/Bordeaux/Imagenet_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
    kit_ima = pickle.load(fp)    

kit_ima = np.mean(kit_ima[0],0)
kit_ima = np.reshape(kit_ima,(kit_ima.shape[0]*kit_ima.shape[1]*kit_ima.shape[2],))




with open("activations/Bordeaux/Places_source/LFW/activations.txt", "rb") as fp:   # Unpickling
    lfw_plc = pickle.load(fp)

lfw_plc = np.mean(lfw_plc[0],0)
lfw_plc = np.reshape(lfw_plc,(lfw_plc.shape[0]*lfw_plc.shape[1]*lfw_plc.shape[2],))


with open("activations/Bordeaux/Bedroom_source/LFW/activations.txt", "rb") as fp:   # Unpickling
    lfw_bed = pickle.load(fp)
    
lfw_bed = np.mean(lfw_bed[0],0)
lfw_bed = np.reshape(lfw_bed,(lfw_bed.shape[0]*lfw_bed.shape[1]*lfw_bed.shape[2],))


with open("activations/Bordeaux/celebA_source/LFW/activations.txt", "rb") as fp:   # Unpickling
    lfw_cel = pickle.load(fp)

lfw_cel = np.mean(lfw_cel[0],0)
lfw_cel = np.reshape(lfw_cel,(lfw_cel.shape[0]*lfw_cel.shape[1]*lfw_cel.shape[2],))


with open("activations/Bordeaux/Imagenet_source/LFW/activations.txt", "rb") as fp:   # Unpickling
    lfw_ima = pickle.load(fp)    

lfw_ima = np.mean(lfw_ima[0],0)
lfw_ima = np.reshape(lfw_ima,(lfw_ima.shape[0]*lfw_ima.shape[1]*lfw_ima.shape[2],))




with open("activations/Bordeaux/Places_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
    cit_plc = pickle.load(fp)

cit_plc = np.mean(cit_plc[0],0)
cit_plc = np.reshape(cit_plc,(cit_plc.shape[0]*cit_plc.shape[1]*cit_plc.shape[2],))


with open("activations/Bordeaux/Bedroom_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
    cit_bed = pickle.load(fp)
    
cit_bed = np.mean(cit_bed[0],0)
cit_bed = np.reshape(cit_bed,(cit_bed.shape[0]*cit_bed.shape[1]*cit_bed.shape[2],))


with open("activations/Bordeaux/celebA_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
    cit_cel = pickle.load(fp)

cit_cel = np.mean(cit_cel[0],0)
cit_cel = np.reshape(cit_cel,(cit_cel.shape[0]*cit_cel.shape[1]*cit_cel.shape[2],))


with open("activations/Bordeaux/Imagenet_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
    cit_ima = pickle.load(fp)    

cit_ima = np.mean(cit_ima[0],0)
cit_ima = np.reshape(cit_ima,(cit_ima.shape[0]*cit_ima.shape[1]*cit_ima.shape[2],))



# Extracting activation differences

flo_plc_diff = np.abs(flo_plc - plc_plc)
flo_cel_diff = np.abs(flo_cel - cel_cel)
flo_ima_diff = np.abs(flo_ima - ima_ima)
flo_bed_diff = np.abs(flo_bed - bed_bed)

kit_plc_diff = np.abs(kit_plc - plc_plc)
kit_cel_diff = np.abs(kit_cel - cel_cel)
kit_ima_diff = np.abs(kit_ima - ima_ima)
kit_bed_diff = np.abs(kit_bed - bed_bed)

lfw_plc_diff = np.abs(lfw_plc - plc_plc)
lfw_cel_diff = np.abs(lfw_cel - cel_cel)
lfw_ima_diff = np.abs(lfw_ima - ima_ima)
lfw_bed_diff = np.abs(lfw_bed - bed_bed)

cit_plc_diff = np.abs(cit_plc - plc_plc)
cit_cel_diff = np.abs(cit_cel - cel_cel)
cit_ima_diff = np.abs(cit_ima - ima_ima)
cit_bed_diff = np.abs(cit_bed - bed_bed)

# Sorting activation differences in order to find filters that have the highest differences
flo_plc_idx = np.argsort(flo_plc_diff)[::-1]
flo_cel_idx = np.argsort(flo_cel_diff)[::-1]
flo_ima_idx = np.argsort(flo_ima_diff)[::-1]
flo_bed_idx = np.argsort(flo_bed_diff)[::-1]

kit_plc_idx = np.argsort(kit_plc_diff)[::-1]
kit_cel_idx = np.argsort(kit_cel_diff)[::-1]
kit_ima_idx = np.argsort(kit_ima_diff)[::-1]
kit_bed_idx = np.argsort(kit_bed_diff)[::-1]

lfw_plc_idx = np.argsort(lfw_plc_diff)[::-1]
lfw_cel_idx = np.argsort(lfw_cel_diff)[::-1]
lfw_ima_idx = np.argsort(lfw_ima_diff)[::-1]
lfw_bed_idx = np.argsort(lfw_bed_diff)[::-1]

cit_plc_idx = np.argsort(cit_plc_diff)[::-1]
cit_cel_idx = np.argsort(cit_cel_diff)[::-1]
cit_ima_idx = np.argsort(cit_ima_diff)[::-1]
cit_bed_idx = np.argsort(cit_bed_diff)[::-1]


# Selecting feat_num indices with highest differences
flo_plc_idx = flo_plc_idx[0:feat_num]
flo_cel_idx = flo_cel_idx[0:feat_num]
flo_ima_idx = flo_ima_idx[0:feat_num]
flo_bed_idx = flo_bed_idx[0:feat_num]

kit_plc_idx = kit_plc_idx[0:feat_num]
kit_cel_idx = kit_cel_idx[0:feat_num]
kit_ima_idx = kit_ima_idx[0:feat_num]
kit_bed_idx = kit_bed_idx[0:feat_num]

lfw_plc_idx = lfw_plc_idx[0:feat_num]
lfw_cel_idx = lfw_cel_idx[0:feat_num]
lfw_ima_idx = lfw_ima_idx[0:feat_num]
lfw_bed_idx = lfw_bed_idx[0:feat_num]

cit_plc_idx = cit_plc_idx[0:feat_num]
cit_cel_idx = cit_cel_idx[0:feat_num]
cit_ima_idx = cit_ima_idx[0:feat_num]
cit_bed_idx = cit_bed_idx[0:feat_num]


# Selecting filter differenes from calculated indices
flo_plc_diff = flo_plc_diff[flo_plc_idx]
flo_cel_diff = flo_cel_diff[flo_cel_idx]
flo_ima_diff = flo_ima_diff[flo_ima_idx]
flo_bed_diff = flo_bed_diff[flo_bed_idx]

kit_plc_diff = kit_plc_diff[kit_plc_idx]
kit_cel_diff = kit_cel_diff[kit_cel_idx]
kit_ima_diff = kit_ima_diff[kit_ima_idx]
kit_bed_diff = kit_bed_diff[kit_bed_idx]

lfw_plc_diff = lfw_plc_diff[lfw_plc_idx]
lfw_cel_diff = lfw_cel_diff[lfw_cel_idx]
lfw_ima_diff = lfw_ima_diff[lfw_ima_idx]
lfw_bed_diff = lfw_bed_diff[lfw_bed_idx]

cit_plc_diff = cit_plc_diff[cit_plc_idx]
cit_cel_diff = cit_cel_diff[cit_cel_idx]
cit_ima_diff = cit_ima_diff[cit_ima_idx]
cit_bed_diff = cit_bed_diff[cit_bed_idx]


# Calculating mean of corresponding filters that corresponds to prediction of distances between datasets
flo_plc_dist = np.mean(flo_plc_diff)
flo_cel_dist = np.mean(flo_cel_diff)
flo_ima_dist = np.mean(flo_ima_diff)
flo_bed_dist = np.mean(flo_bed_diff)

kit_plc_dist = np.mean(kit_plc_diff)
kit_cel_dist = np.mean(kit_cel_diff)
kit_ima_dist = np.mean(kit_ima_diff)
kit_bed_dist = np.mean(kit_bed_diff)

lfw_plc_dist = np.mean(lfw_plc_diff)
lfw_cel_dist = np.mean(lfw_cel_diff)
lfw_ima_dist = np.mean(lfw_ima_diff)
lfw_bed_dist = np.mean(lfw_bed_diff)

cit_plc_dist = np.mean(cit_plc_diff)
cit_cel_dist = np.mean(cit_cel_diff)
cit_ima_dist = np.mean(cit_ima_diff)
cit_bed_dist = np.mean(cit_bed_diff)

