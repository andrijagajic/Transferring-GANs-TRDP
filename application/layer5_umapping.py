# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 08:40:07 2019

@author: Nadja
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt
import umap
import os
import seaborn as sns       
        
        
os.chdir('../')

# Bedrooms target
with open("activations/Bordeaux/Bedroom_source/Bedrooms/activations.txt", "rb") as fp:   # Unpickling
    bed_bed = pickle.load(fp)[0]

bed_bed = np.max(bed_bed,(2,3)) # spatial

with open("activations/Bordeaux/Celeba_source/Bedrooms/activations.txt", "rb") as fp:   # Unpickling
    bed_cel = pickle.load(fp)[0]

bed_cel = np.max(bed_cel,(2,3)) # spatial

with open("activations/Bordeaux/Imagenet_source/Bedrooms/activations.txt", "rb") as fp:   # Unpickling
    bed_ima = pickle.load(fp)[0]

bed_ima = np.max(bed_ima,(2,3)) # spatial

with open("activations/Bordeaux/Places_source/Bedrooms/activations.txt", "rb") as fp:   # Unpickling
    bed_pla = pickle.load(fp)[0]

bed_pla = np.max(bed_pla,(2,3)) # spatial

bed = np.concatenate((bed_bed, bed_cel, bed_ima, bed_pla),1)

del bed_bed, bed_cel, bed_ima, bed_pla

# CelebA target
with open("activations/Bordeaux/Bedroom_source/Celeba/activations.txt", "rb") as fp:   # Unpickling
    cel_bed = pickle.load(fp)[0]

cel_bed = np.max(cel_bed,(2,3)) # spatial

with open("activations/Bordeaux/Celeba_source/Celeba/activations.txt", "rb") as fp:   # Unpickling
    cel_cel = pickle.load(fp)[0]

cel_cel = np.max(cel_cel,(2,3)) # spatial

with open("activations/Bordeaux/Imagenet_source/Celeba/activations.txt", "rb") as fp:   # Unpickling
    cel_ima = pickle.load(fp)[0]

cel_ima = np.max(cel_ima,(2,3)) # spatial

with open("activations/Bordeaux/Places_source/Celeba/activations.txt", "rb") as fp:   # Unpickling
    cel_pla = pickle.load(fp)[0]

cel_pla = np.max(cel_pla,(2,3)) # spatial

cel = np.concatenate((cel_bed, cel_cel, cel_ima, cel_pla),1)

del cel_bed, cel_pla, cel_ima, cel_cel



# Cityscapes target
with open("activations/Bordeaux/Bedroom_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
    cit_bed = pickle.load(fp)[0]

cit_bed = np.max(cit_bed,(2,3)) # spatial

with open("activations/Bordeaux/Celeba_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
    cit_cel = pickle.load(fp)[0]

cit_cel = np.max(cit_cel,(2,3)) # spatial

with open("activations/Bordeaux/Imagenet_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
    cit_ima = pickle.load(fp)[0]

cit_ima = np.max(cit_ima,(2,3)) # spatial

with open("activations/Bordeaux/Places_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
    cit_pla = pickle.load(fp)[0]

cit_pla = np.max(cit_pla,(2,3)) # spatial

cit = np.concatenate((cit_bed, cit_cel, cit_ima, cit_pla),1)

del cit_bed, cit_pla, cit_ima, cit_cel



# Flowers target
with open("activations/Bordeaux/Bedroom_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    flo_bed = pickle.load(fp)[0]

flo_bed = np.max(flo_bed,(2,3)) # spatial

with open("activations/Bordeaux/Celeba_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    flo_cel = pickle.load(fp)[0]

flo_cel = np.max(flo_cel,(2,3)) # spatial

with open("activations/Bordeaux/Imagenet_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    flo_ima = pickle.load(fp)[0]

flo_ima = np.max(flo_ima,(2,3)) # spatial

with open("activations/Bordeaux/Places_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
    flo_pla = pickle.load(fp)[0]

flo_pla = np.max(flo_pla,(2,3)) # spatial

flo = np.concatenate((flo_bed, flo_cel, flo_ima, flo_pla),1)

del flo_bed, flo_pla, flo_ima, flo_cel



# Imagenet target
with open("activations/Bordeaux/Bedroom_source/Imagenet/activations.txt", "rb") as fp:   # Unpickling
    ima_bed = pickle.load(fp)[0]

ima_bed = np.max(ima_bed,(2,3)) # spatial

with open("activations/Bordeaux/Celeba_source/Imagenet/activations.txt", "rb") as fp:   # Unpickling
    ima_cel = pickle.load(fp)[0]

ima_cel = np.max(ima_cel,(2,3)) # spatial

with open("activations/Bordeaux/Imagenet_source/Imagenet/activations.txt", "rb") as fp:   # Unpickling
    ima_ima = pickle.load(fp)[0]

ima_ima = np.max(ima_ima,(2,3)) # spatial

with open("activations/Bordeaux/Places_source/Imagenet/activations.txt", "rb") as fp:   # Unpickling
    ima_pla = pickle.load(fp)[0]

ima_pla = np.max(ima_pla,(2,3)) # spatial

ima = np.concatenate((ima_bed, ima_cel, ima_ima, ima_pla),1)

del ima_bed, ima_pla, ima_ima, ima_cel



# Kitchens target
with open("activations/Bordeaux/Bedroom_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
    kit_bed = pickle.load(fp)[0]

kit_bed = np.max(kit_bed,(2,3)) # spatial

with open("activations/Bordeaux/Celeba_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
    kit_cel = pickle.load(fp)[0]

kit_cel = np.max(kit_cel,(2,3)) # spatial

with open("activations/Bordeaux/Imagenet_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
    kit_ima = pickle.load(fp)[0]

kit_ima = np.max(kit_ima,(2,3)) # spatial

with open("activations/Bordeaux/Places_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
    kit_pla = pickle.load(fp)[0]

kit_pla = np.max(kit_pla,(2,3)) # spatial

kit = np.concatenate((kit_bed, kit_cel, kit_ima, kit_pla),1)

del kit_bed, kit_pla, kit_ima, kit_cel


# LFW target
with open("activations/Bordeaux/Bedroom_source/LFW/activations.txt", "rb") as fp:   # Unpickling
    lfw_bed = pickle.load(fp)[0]

lfw_bed = np.max(lfw_bed,(2,3)) # spatial

with open("activations/Bordeaux/Celeba_source/LFW/activations.txt", "rb") as fp:   # Unpickling
    lfw_cel = pickle.load(fp)[0]

lfw_cel = np.max(lfw_cel,(2,3)) # spatial

with open("activations/Bordeaux/Imagenet_source/LFW/activations.txt", "rb") as fp:   # Unpickling
    lfw_ima = pickle.load(fp)[0]

lfw_ima = np.max(lfw_ima,(2,3)) # spatial

with open("activations/Bordeaux/Places_source/LFW/activations.txt", "rb") as fp:   # Unpickling
    lfw_pla = pickle.load(fp)[0]

lfw_pla = np.max(lfw_pla,(2,3)) # spatial

lfw = np.concatenate((lfw_bed, lfw_cel, lfw_ima, lfw_pla),1)

del lfw_bed, lfw_pla, lfw_ima, lfw_cel    

# Places target
with open("activations/Bordeaux/Bedroom_source/Places/activations.txt", "rb") as fp:   # Unpickling
    pla_bed = pickle.load(fp)[0]

pla_bed = np.max(pla_bed,(2,3)) # spatial

with open("activations/Bordeaux/Celeba_source/Places/activations.txt", "rb") as fp:   # Unpickling
    pla_cel = pickle.load(fp)[0]

pla_cel = np.max(pla_cel,(2,3)) # spatial

with open("activations/Bordeaux/Imagenet_source/Places/activations.txt", "rb") as fp:   # Unpickling
    pla_ima = pickle.load(fp)[0]

pla_ima = np.max(pla_ima,(2,3)) # spatial

with open("activations/Bordeaux/Places_source/Places/activations.txt", "rb") as fp:   # Unpickling
    pla_pla = pickle.load(fp)[0]

pla_pla = np.max(pla_pla,(2,3)) # spatial

pla = np.concatenate((pla_bed, pla_cel, pla_ima, pla_pla),1)

del pla_bed, pla_pla, pla_ima, pla_cel

reducer = umap.UMAP()
comb = np.concatenate((bed,cel,cit,flo,ima,kit,lfw,pla),0)

embedding = reducer.fit_transform(comb)

bed_emb = embedding[0:600,:]
cel_emb = embedding[600:1200,:]
cit_emb = embedding[1200:1800,:]
flo_emb = embedding[1800:2400,:]
ima_emb = embedding[2400:3000,:]
kit_emb = embedding[3000:3600,:]
lfw_emb = embedding[3600:4200,:]
pla_emb = embedding[4200:4800,:]
    

