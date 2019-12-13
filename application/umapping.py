# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:56:53 2019

@author: andrija
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:25:50 2019

@author: andrija
"""

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
from matplotlib import pyplot as plt
import umap
import os
import seaborn as sns

os.chdir('../')


def calc_umap(lay_num):
    for k in range(lay_num,lay_num+1):
        with open("activations/Bordeaux/Bedroom_source/Bedrooms/activations_det.txt", "rb") as fp:   # Unpickling
            bed_bed = pickle.load(fp)[k]
        
        bed_bed = np.max(bed_bed,(2,3)) # spatial
        
        with open("activations/Bordeaux/Celeba_source/Bedrooms/activations_det.txt", "rb") as fp:   # Unpickling
            bed_cel = pickle.load(fp)[k]
        
        bed_cel = np.max(bed_cel,(2,3)) # spatial
        
        with open("activations/Bordeaux/Imagenet_source/Bedrooms/activations_det.txt", "rb") as fp:   # Unpickling
            bed_ima = pickle.load(fp)[k]
        
        bed_ima = np.max(bed_ima,(2,3)) # spatial
        
        with open("activations/Bordeaux/Places_source/Bedrooms/activations_det.txt", "rb") as fp:   # Unpickling
            bed_pla = pickle.load(fp)[k]
        
        bed_pla = np.max(bed_pla,(2,3)) # spatial
        
        bed = np.concatenate((bed_bed, bed_cel, bed_ima, bed_pla),1)
        
        del bed_bed, bed_cel, bed_ima, bed_pla
        
        # CelebA target
        with open("activations/Bordeaux/Bedroom_source/Celeba/activations_det.txt", "rb") as fp:   # Unpickling
            cel_bed = pickle.load(fp)[k]
        
        cel_bed = np.max(cel_bed,(2,3)) # spatial
        
        with open("activations/Bordeaux/Celeba_source/Celeba/activations_det.txt", "rb") as fp:   # Unpickling
            cel_cel = pickle.load(fp)[k]
        
        cel_cel = np.max(cel_cel,(2,3)) # spatial
        
        with open("activations/Bordeaux/Imagenet_source/Celeba/activations_det.txt", "rb") as fp:   # Unpickling
            cel_ima = pickle.load(fp)[k]
        
        cel_ima = np.max(cel_ima,(2,3)) # spatial
        
        with open("activations/Bordeaux/Places_source/Celeba/activations_det.txt", "rb") as fp:   # Unpickling
            cel_pla = pickle.load(fp)[k]
        
        cel_pla = np.max(cel_pla,(2,3)) # spatial
        
        cel = np.concatenate((cel_bed, cel_cel, cel_ima, cel_pla),1)
        
        del cel_bed, cel_pla, cel_ima, cel_cel
        
        
        
        # Cityscapes target
        with open("activations/Bordeaux/Bedroom_source/Cityscapes/activations_det.txt", "rb") as fp:   # Unpickling
            cit_bed = pickle.load(fp)[k]
        
        cit_bed = np.max(cit_bed,(2,3)) # spatial
        
        with open("activations/Bordeaux/Celeba_source/Cityscapes/activations_det.txt", "rb") as fp:   # Unpickling
            cit_cel = pickle.load(fp)[k]
        
        cit_cel = np.max(cit_cel,(2,3)) # spatial
        
        with open("activations/Bordeaux/Imagenet_source/Cityscapes/activations_det.txt", "rb") as fp:   # Unpickling
            cit_ima = pickle.load(fp)[k]
        
        cit_ima = np.max(cit_ima,(2,3)) # spatial
        
        with open("activations/Bordeaux/Places_source/Cityscapes/activations_det.txt", "rb") as fp:   # Unpickling
            cit_pla = pickle.load(fp)[k]
        
        cit_pla = np.max(cit_pla,(2,3)) # spatial
        
        cit = np.concatenate((cit_bed, cit_cel, cit_ima, cit_pla),1)
        
        del cit_bed, cit_pla, cit_ima, cit_cel
        
        
        
        # Flowers target
        with open("activations/Bordeaux/Bedroom_source/Flowers/activations_det.txt", "rb") as fp:   # Unpickling
            flo_bed = pickle.load(fp)[k]
        
        flo_bed = np.max(flo_bed,(2,3)) # spatial
        
        with open("activations/Bordeaux/Celeba_source/Flowers/activations_det.txt", "rb") as fp:   # Unpickling
            flo_cel = pickle.load(fp)[k]
        
        flo_cel = np.max(flo_cel,(2,3)) # spatial
        
        with open("activations/Bordeaux/Imagenet_source/Flowers/activations_det.txt", "rb") as fp:   # Unpickling
            flo_ima = pickle.load(fp)[k]
        
        flo_ima = np.max(flo_ima,(2,3)) # spatial
        
        with open("activations/Bordeaux/Places_source/Flowers/activations_det.txt", "rb") as fp:   # Unpickling
            flo_pla = pickle.load(fp)[k]
        
        flo_pla = np.max(flo_pla,(2,3)) # spatial
        
        flo = np.concatenate((flo_bed, flo_cel, flo_ima, flo_pla),1)
        
        del flo_bed, flo_pla, flo_ima, flo_cel
        
        
        
        # Imagenet target
        with open("activations/Bordeaux/Bedroom_source/Imagenet/activations_det.txt", "rb") as fp:   # Unpickling
            ima_bed = pickle.load(fp)[k]
        
        ima_bed = np.max(ima_bed,(2,3)) # spatial
        
        with open("activations/Bordeaux/Celeba_source/Imagenet/activations_det.txt", "rb") as fp:   # Unpickling
            ima_cel = pickle.load(fp)[k]
        
        ima_cel = np.max(ima_cel,(2,3)) # spatial
        
        with open("activations/Bordeaux/Imagenet_source/Imagenet/activations_det.txt", "rb") as fp:   # Unpickling
            ima_ima = pickle.load(fp)[k]
        
        ima_ima = np.max(ima_ima,(2,3)) # spatial
        
        with open("activations/Bordeaux/Places_source/Imagenet/activations_det.txt", "rb") as fp:   # Unpickling
            ima_pla = pickle.load(fp)[k]
        
        ima_pla = np.max(ima_pla,(2,3)) # spatial
        
        ima = np.concatenate((ima_bed, ima_cel, ima_ima, ima_pla),1)
        
        del ima_bed, ima_pla, ima_ima, ima_cel
        
        
        
        # Kitchens target
        with open("activations/Bordeaux/Bedroom_source/Kitchens/activations_det.txt", "rb") as fp:   # Unpickling
            kit_bed = pickle.load(fp)[k]
        
        kit_bed = np.max(kit_bed,(2,3)) # spatial
        
        with open("activations/Bordeaux/Celeba_source/Kitchens/activations_det.txt", "rb") as fp:   # Unpickling
            kit_cel = pickle.load(fp)[k]
        
        kit_cel = np.max(kit_cel,(2,3)) # spatial
        
        with open("activations/Bordeaux/Imagenet_source/Kitchens/activations_det.txt", "rb") as fp:   # Unpickling
            kit_ima = pickle.load(fp)[k]
        
        kit_ima = np.max(kit_ima,(2,3)) # spatial
        
        with open("activations/Bordeaux/Places_source/Kitchens/activations_det.txt", "rb") as fp:   # Unpickling
            kit_pla = pickle.load(fp)[k]
        
        kit_pla = np.max(kit_pla,(2,3)) # spatial
        
        kit = np.concatenate((kit_bed, kit_cel, kit_ima, kit_pla),1)
        
        del kit_bed, kit_pla, kit_ima, kit_cel
        
        
        # LFW target
        with open("activations/Bordeaux/Bedroom_source/LFW/activations_det.txt", "rb") as fp:   # Unpickling
            lfw_bed = pickle.load(fp)[k]
        
        lfw_bed = np.max(lfw_bed,(2,3)) # spatial
        
        with open("activations/Bordeaux/Celeba_source/LFW/activations_det.txt", "rb") as fp:   # Unpickling
            lfw_cel = pickle.load(fp)[k]
        
        lfw_cel = np.max(lfw_cel,(2,3)) # spatial
        
        with open("activations/Bordeaux/Imagenet_source/LFW/activations_det.txt", "rb") as fp:   # Unpickling
            lfw_ima = pickle.load(fp)[k]
        
        lfw_ima = np.max(lfw_ima,(2,3)) # spatial
        
        with open("activations/Bordeaux/Places_source/LFW/activations_det.txt", "rb") as fp:   # Unpickling
            lfw_pla = pickle.load(fp)[k]
        
        lfw_pla = np.max(lfw_pla,(2,3)) # spatial
        
        lfw = np.concatenate((lfw_bed, lfw_cel, lfw_ima, lfw_pla),1)
        
        del lfw_bed, lfw_pla, lfw_ima, lfw_cel    
        
        # Places target
        with open("activations/Bordeaux/Bedroom_source/Places/activations_det.txt", "rb") as fp:   # Unpickling
            pla_bed = pickle.load(fp)[k]
        
        pla_bed = np.max(pla_bed,(2,3)) # spatial
        
        with open("activations/Bordeaux/Celeba_source/Places/activations_det.txt", "rb") as fp:   # Unpickling
            pla_cel = pickle.load(fp)[k]
        
        pla_cel = np.max(pla_cel,(2,3)) # spatial
        
        with open("activations/Bordeaux/Imagenet_source/Places/activations_det.txt", "rb") as fp:   # Unpickling
            pla_ima = pickle.load(fp)[k]
        
        pla_ima = np.max(pla_ima,(2,3)) # spatial
        
        with open("activations/Bordeaux/Places_source/Places/activations_det.txt", "rb") as fp:   # Unpickling
            pla_pla = pickle.load(fp)[k]
        
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
        return bed_emb, cel_emb, cit_emb, flo_emb, ima_emb, kit_emb, lfw_emb, pla_emb
    
    colors = ['b', 'c', 'y', 'm', 'r', 'p', 'k', 'g']
    
    plt.figure()
    sc1 = plt.scatter(bed_emb[:,0], bed_emb[:,1], s=14, marker='o', color=sns.color_palette()[0])
    sc2 = plt.scatter(cel_emb[:,0], cel_emb[:,1], s=14, marker='o', color=sns.color_palette()[1])
    sc3 = plt.scatter(cit_emb[:,0], cit_emb[:,1], s=14, marker='o', color=sns.color_palette()[2])
    sc4 = plt.scatter(flo_emb[:,0], flo_emb[:,1], s=14, marker='o', color=sns.color_palette()[3])
    sc5 = plt.scatter(ima_emb[:,0], ima_emb[:,1], s=14, marker='o', color=sns.color_palette()[4])
    sc6 = plt.scatter(kit_emb[:,0], kit_emb[:,1], s=14, marker='o', color=sns.color_palette()[5])
    sc7 = plt.scatter(lfw_emb[:,0], lfw_emb[:,1], s=14, marker='o', color=sns.color_palette()[6])
    sc8 = plt.scatter(pla_emb[:,0], pla_emb[:,1], s=14, marker='o', color=sns.color_palette()[7])
    
        
    plt.legend((sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8),
               ('Bedrooms', 'Celeba', 'Cityscapes', 'Flowers', 'Imagenet', 'Kitchens', 'LFW', 'Places'),
               scatterpoints=3,
               loc='best',
               ncol=2,
               fontsize=10)
    plt.title('UMAP for Discriminator Layer ' + str(k+1))
    plt.show()
        
    bed_cen = np.median(bed_emb,0)
    cel_cen = np.median(cel_emb,0)
    cit_cen = np.median(cit_emb,0)
    flo_cen = np.median(flo_emb,0)
    ima_cen = np.median(ima_emb,0)
    kit_cen = np.median(kit_emb,0)
    lfw_cen = np.median(lfw_emb,0)
    pla_cen = np.median(pla_emb,0)
    
    cit_bed = np.linalg.norm(cit_cen-bed_cen)
    cit_cel = np.linalg.norm(cit_cen-cel_cen)
    cit_ima = np.linalg.norm(cit_cen-ima_cen)
    cit_pla = np.linalg.norm(cit_cen-pla_cen)
    
    flo_bed = np.linalg.norm(flo_cen-bed_cen)
    flo_cel = np.linalg.norm(flo_cen-cel_cen)
    flo_ima = np.linalg.norm(flo_cen-ima_cen)
    flo_pla = np.linalg.norm(flo_cen-pla_cen)
    
    kit_bed = np.linalg.norm(kit_cen-bed_cen)
    kit_cel = np.linalg.norm(kit_cen-cel_cen)
    kit_ima = np.linalg.norm(kit_cen-ima_cen)
    kit_pla = np.linalg.norm(kit_cen-pla_cen)
    
    
    lfw_bed = np.linalg.norm(lfw_cen-bed_cen)
    lfw_cel = np.linalg.norm(lfw_cen-cel_cen)
    lfw_ima = np.linalg.norm(lfw_cen-ima_cen)
    lfw_pla = np.linalg.norm(lfw_cen-pla_cen)
