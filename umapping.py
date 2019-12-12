# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:05:10 2019

@author: andrija
"""
import pickle
import numpy as np
from matplotlib import pyplot as plt
import umap
import seaborn as sns

def mle_estimation(test_data, mean, cov):
    k = test_data.shape[1]
    dist = 0
    for i in range(0,len(test_data)):
#        dist *= (2*np.pi)**(-k/2) * np.linalg.det(cov)**-0.5 * np.exp(-0.5*np.matmul(np.matmul(np.transpose(test_data[i,:]-mean),cov**-1),test_data[i,:]-mean))
        dist += (-k/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(cov)) - 0.5 * np.matmul(np.matmul(np.transpose(test_data[i,:]-mean),np.linalg.inv(cov)),test_data[i,:]-mean)
    return dist

def knn_estimation(target_data, source_data, k):
    res = [0, 0, 0, 0]
    for i in range(0,len(target_data)):
        dist = np.linalg.norm(target_data[i,:] - source_data, axis = 1)
        idx = np.argsort(dist)[0:k]
        idx = idx//600
        for j in range(k):
            res[idx[j]] += 1
    res /= np.sum(res)
    return res

supervised = False # Set supervised to True in order to get supervised UMAP
all_layers = False # Set all_layers to True in order to extract 5 UMAPs, one for each layer inside discriminator network 
histogram_features = False # SEt histogram_features to True in order to use histogram features instead of max pooling


if all_layers == True:
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title('UMAP for Discriminator Layer 1')
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.set_title('UMAP for Discriminator Layer 2')
    ax3 = fig.add_subplot(gs[0, 4:])
    ax3.set_title('UMAP for Discriminator Layer 3')
    ax4 = fig.add_subplot(gs[-1, 1:3])
    ax4.set_title('UMAP for Discriminator Layer 4')
    ax5 = fig.add_subplot(gs[-1, 3:5])
    ax5.set_title('UMAP for Discriminator Layer 5')
    
    for k in range(0,5):
        with open("activations/Bordeaux/Bedroom_source/Bedrooms/activations_det.txt", "rb") as fp:   # Unpickling
            bed_bed = pickle.load(fp)[k]
        
        bed_bed = np.max(bed_bed,(2,3)) # spatial
        num_img = bed_bed.shape[0]
        
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
        
        
        if (not supervised):
            comb = np.concatenate((bed,cel,cit,flo,ima,kit,lfw,pla),0)    
            embedding = reducer.fit_transform(comb)
            
            bed_emb = embedding[0:num_img,:]
            cel_emb = embedding[num_img:2*num_img,:]
            cit_emb = embedding[2*num_img:3*num_img,:]
            flo_emb = embedding[3*num_img:4*num_img,:]
            ima_emb = embedding[4*num_img:5*num_img,:]
            kit_emb = embedding[5*num_img:6*num_img,:]
            lfw_emb = embedding[6*num_img:7*num_img,:]
            pla_emb = embedding[7*num_img:8*num_img,:]
    
        
        else:
            num_train = int(5/6 * num_img)
            num_test = int(1/6 * num_img)
            bed_train = bed[0:num_train,:]
            cel_train = cel[0:num_train,:]
            cit_train = cit[0:num_train,:]
            flo_train = flo[0:num_train,:]
            ima_train = ima[0:num_train,:]
            kit_train = kit[0:num_train,:]
            lfw_train = lfw[0:num_train,:]
            pla_train = pla[0:num_train,:]
            comb_train = np.concatenate((bed_train,cel_train,cit_train,flo_train,ima_train,kit_train,lfw_train,pla_train),0)
            
            bed_test = bed[num_train:,:]
            cel_test = cel[num_train:,:]
            cit_test = cit[num_train:,:]
            flo_test = flo[num_train:,:]
            ima_test = ima[num_train:,:]
            kit_test = kit[num_train:,:]
            lfw_test = lfw[num_train:,:]
            pla_test = pla[num_train:,:]
            comb_test = np.concatenate((bed_test,cel_test,cit_test,flo_test,ima_test,kit_test,lfw_test,pla_test),0)
    
            y = []
            for k in range(0,len(comb_train)):
                color = k//num_train
                y.append(color)
            y = np.array(y)
            
            reducer.fit(comb_train,y)
            
            embedding = reducer.transform(comb_test)
            
            bed_emb = embedding[0:num_test,:]    
            cel_emb = embedding[num_test:2*num_test,:]
            cit_emb = embedding[2*num_test:3*num_test,:]
            flo_emb = embedding[3*num_test:4*num_test,:]
            ima_emb = embedding[4*num_test:5*num_test,:]
            kit_emb = embedding[5*num_test:6*num_test,:]
            lfw_emb = embedding[6*num_test:7*num_test,:]
            pla_emb = embedding[7*num_test:8*num_test,:]
            

        
        
        colors = ['b', 'c', 'y', 'm', 'r', 'p', 'k', 'g']
        
        if k == 0:
            ax = ax1
        elif k == 1:
            ax = ax2
        elif k == 2:
            ax = ax3
        elif k == 3:
            ax = ax4
        elif k == 4:
            ax = ax5
        
        sc1 = ax.scatter(bed_emb[:,0], bed_emb[:,1], s=14, marker='o', color=sns.color_palette()[0])
        sc2 = ax.scatter(cel_emb[:,0], cel_emb[:,1], s=14, marker='o', color=sns.color_palette()[1])
        sc3 = ax.scatter(cit_emb[:,0], cit_emb[:,1], s=14, marker='o', color=sns.color_palette()[2])
        sc4 = ax.scatter(flo_emb[:,0], flo_emb[:,1], s=14, marker='o', color=sns.color_palette()[3])
        sc5 = ax.scatter(ima_emb[:,0], ima_emb[:,1], s=14, marker='o', color=sns.color_palette()[4])
        sc6 = ax.scatter(kit_emb[:,0], kit_emb[:,1], s=14, marker='o', color=sns.color_palette()[5])
        sc7 = ax.scatter(lfw_emb[:,0], lfw_emb[:,1], s=14, marker='o', color=sns.color_palette()[6])
        sc8 = ax.scatter(pla_emb[:,0], pla_emb[:,1], s=14, marker='o', color=sns.color_palette()[7])
        
            
        ax.legend((sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8),
                   ('Bedrooms', 'Celeba', 'Cityscapes', 'Flowers', 'Imagenet', 'Kitchens', 'LFW', 'Places'),
                   scatterpoints=3,
                   loc='best',
                   ncol=2,
                   fontsize=10)
        #ax.title('UMAP for Discriminator Layer ' + str(k+1))
        #ax.show()
        
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
else:
    # Bedrooms target
    with open("activations/Bordeaux/Bedroom_source/Bedrooms/activations.txt", "rb") as fp:   # Unpickling
        bed_bed = pickle.load(fp)[0]
   
    num_img = bed_bed.shape[0]
    if not histogram_features:
        bed_bed = np.max(bed_bed,(2,3)) # spatial
    else:
        bed_bed = np.reshape(bed_bed,(600,512*4*4))

    
    with open("activations/Bordeaux/Celeba_source/Bedrooms/activations.txt", "rb") as fp:   # Unpickling
        bed_cel = pickle.load(fp)[0]
    if not histogram_features:
        bed_cel = np.max(bed_cel,(2,3)) # spatial
    else:
        bed_cel = np.reshape(bed_cel,(600,512*4*4))

    
    with open("activations/Bordeaux/Imagenet_source/Bedrooms/activations.txt", "rb") as fp:   # Unpickling
        bed_ima = pickle.load(fp)[0]
    
    if not histogram_features:
        bed_ima = np.max(bed_ima,(2,3)) # spatial
    else:
        bed_ima = np.reshape(bed_ima,(600,512*4*4))

    with open("activations/Bordeaux/Places_source/Bedrooms/activations.txt", "rb") as fp:   # Unpickling
        bed_pla = pickle.load(fp)[0]
    
    if not histogram_features:
        bed_pla = np.max(bed_pla,(2,3)) # spatial
    else:
        bed_pla = np.reshape(bed_pla,(600,512*4*4))

    bed = np.concatenate((bed_bed, bed_cel, bed_ima, bed_pla),1)
        
    # CelebA target
    with open("activations/Bordeaux/Bedroom_source/Celeba/activations.txt", "rb") as fp:   # Unpickling
        cel_bed = pickle.load(fp)[0]
    
    if not histogram_features:
        cel_bed = np.max(cel_bed,(2,3)) # spatial
    else:
        cel_bed = np.reshape(cel_bed,(600,512*4*4))

    with open("activations/Bordeaux/Celeba_source/Celeba/activations.txt", "rb") as fp:   # Unpickling
        cel_cel = pickle.load(fp)[0]
    
    if not histogram_features:
        cel_cel = np.max(cel_cel,(2,3)) # spatial
    else:
        cel_cel = np.reshape(cel_cel,(600,512*4*4))

    with open("activations/Bordeaux/Imagenet_source/Celeba/activations.txt", "rb") as fp:   # Unpickling
        cel_ima = pickle.load(fp)[0]
    
    if not histogram_features:
        cel_ima = np.max(cel_ima,(2,3)) # spatial
    else:
        cel_ima = np.reshape(cel_ima,(600,512*4*4))

    with open("activations/Bordeaux/Places_source/Celeba/activations.txt", "rb") as fp:   # Unpickling
        cel_pla = pickle.load(fp)[0]
    
    if not histogram_features:
        cel_pla = np.max(cel_pla,(2,3)) # spatial
    else:
        cel_pla = np.reshape(cel_pla,(600,512*4*4))

    cel = np.concatenate((cel_bed, cel_cel, cel_ima, cel_pla),1)
    
    
    
    
    # Cityscapes target
    with open("activations/Bordeaux/Bedroom_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
        cit_bed = pickle.load(fp)[0]
    
    if not histogram_features:
        cit_bed = np.max(cit_bed,(2,3)) # spatial
    else:
        cit_bed = np.reshape(cit_bed,(600,512*4*4))

    with open("activations/Bordeaux/Celeba_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
        cit_cel = pickle.load(fp)[0]
    
    if not histogram_features:
        cit_cel = np.max(cit_cel,(2,3)) # spatial
    else:
        cit_cel = np.reshape(cit_cel,(600,512*4*4))

    with open("activations/Bordeaux/Imagenet_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
        cit_ima = pickle.load(fp)[0]
    
    if not histogram_features:
        cit_ima = np.max(cit_ima,(2,3)) # spatial
    else:
        cit_ima = np.reshape(cit_ima,(600,512*4*4))

    with open("activations/Bordeaux/Places_source/Cityscapes/activations.txt", "rb") as fp:   # Unpickling
        cit_pla = pickle.load(fp)[0]
    
    if not histogram_features:
        cit_pla = np.max(cit_pla,(2,3)) # spatial
    else:
        cit_pla = np.reshape(cit_pla,(600,512*4*4))

    cit = np.concatenate((cit_bed, cit_cel, cit_ima, cit_pla),1)
    
    
    
    
    # Flowers target
    with open("activations/Bordeaux/Bedroom_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
        flo_bed = pickle.load(fp)[0]
    
    if not histogram_features:
        flo_bed = np.max(flo_bed,(2,3)) # spatial
    else:
        flo_bed = np.reshape(flo_bed,(600,512*4*4))

    with open("activations/Bordeaux/Celeba_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
        flo_cel = pickle.load(fp)[0]
    
    if not histogram_features:
        flo_cel = np.max(flo_cel,(2,3)) # spatial
    else:
        flo_cel = np.reshape(flo_cel,(600,512*4*4))

    with open("activations/Bordeaux/Imagenet_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
        flo_ima = pickle.load(fp)[0]
    
    if not histogram_features:
        flo_ima = np.max(flo_ima,(2,3)) # spatial
    else:
        flo_ima = np.reshape(flo_ima,(600,512*4*4))

    with open("activations/Bordeaux/Places_source/Flowers/activations.txt", "rb") as fp:   # Unpickling
        flo_pla = pickle.load(fp)[0]
    
    if not histogram_features:
        flo_pla = np.max(flo_pla,(2,3)) # spatial
    else:
        flo_pla = np.reshape(flo_pla,(600,512*4*4))

    flo = np.concatenate((flo_bed, flo_cel, flo_ima, flo_pla),1)
    
    
    
    
    # Imagenet target
    with open("activations/Bordeaux/Bedroom_source/Imagenet/activations.txt", "rb") as fp:   # Unpickling
        ima_bed = pickle.load(fp)[0]
    
    if not histogram_features:
        ima_bed = np.max(ima_bed,(2,3)) # spatial
    else:
        ima_bed = np.reshape(ima_bed,(600,512*4*4))

    
    with open("activations/Bordeaux/Celeba_source/Imagenet/activations.txt", "rb") as fp:   # Unpickling
        ima_cel = pickle.load(fp)[0]
    
    if not histogram_features:
        ima_cel = np.max(ima_cel,(2,3)) # spatial
    else:
        ima_cel = np.reshape(ima_cel,(600,512*4*4))

    with open("activations/Bordeaux/Imagenet_source/Imagenet/activations.txt", "rb") as fp:   # Unpickling
        ima_ima = pickle.load(fp)[0]
    
    if not histogram_features:
        ima_ima = np.max(ima_ima,(2,3)) # spatial
    else:
        ima_ima = np.reshape(ima_ima,(600,512*4*4))

    with open("activations/Bordeaux/Places_source/Imagenet/activations.txt", "rb") as fp:   # Unpickling
        ima_pla = pickle.load(fp)[0]
    
    if not histogram_features:
        ima_pla = np.max(ima_pla,(2,3)) # spatial
    else:
        ima_pla = np.reshape(ima_pla,(600,512*4*4))

    ima = np.concatenate((ima_bed, ima_cel, ima_ima, ima_pla),1)
    
    
    
    
    # Kitchens target
    with open("activations/Bordeaux/Bedroom_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
        kit_bed = pickle.load(fp)[0]
    
    if not histogram_features:
        kit_bed = np.max(kit_bed,(2,3)) # spatial
    else:
        kit_bed = np.reshape(kit_bed,(600,512*4*4))

    
    with open("activations/Bordeaux/Celeba_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
        kit_cel = pickle.load(fp)[0]
    
    if not histogram_features:
        kit_cel = np.max(kit_cel,(2,3)) # spatial
    else:
        kit_cel = np.reshape(kit_cel,(600,512*4*4))
    
    with open("activations/Bordeaux/Imagenet_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
        kit_ima = pickle.load(fp)[0]
    
    if not histogram_features:
        kit_ima = np.max(kit_ima,(2,3)) # spatial
    else:
        kit_ima = np.reshape(kit_ima,(600,512*4*4))

    with open("activations/Bordeaux/Places_source/Kitchens/activations.txt", "rb") as fp:   # Unpickling
        kit_pla = pickle.load(fp)[0]
    
    if not histogram_features:
        kit_pla = np.max(kit_pla,(2,3)) # spatial
    else:
        kit_pla = np.reshape(kit_pla,(600,512*4*4))

    
    kit = np.concatenate((kit_bed, kit_cel, kit_ima, kit_pla),1)
    
    
    
    # LFW target
    with open("activations/Bordeaux/Bedroom_source/LFW/activations.txt", "rb") as fp:   # Unpickling
        lfw_bed = pickle.load(fp)[0]
    
    if not histogram_features:
        lfw_bed = np.max(lfw_bed,(2,3)) # spatial
    else:
        lfw_bed = np.reshape(lfw_bed,(600,512*4*4))

    with open("activations/Bordeaux/Celeba_source/LFW/activations.txt", "rb") as fp:   # Unpickling
        lfw_cel = pickle.load(fp)[0]
    
    if not histogram_features:
        lfw_cel = np.max(lfw_cel,(2,3)) # spatial
    else:
        lfw_cel = np.reshape(lfw_cel,(600,512*4*4))

    with open("activations/Bordeaux/Imagenet_source/LFW/activations.txt", "rb") as fp:   # Unpickling
        lfw_ima = pickle.load(fp)[0]
    
    if not histogram_features:
        lfw_ima = np.max(lfw_ima,(2,3)) # spatial
    else:
        lfw_ima = np.reshape(lfw_ima,(600,512*4*4))

    with open("activations/Bordeaux/Places_source/LFW/activations.txt", "rb") as fp:   # Unpickling
        lfw_pla = pickle.load(fp)[0]
    
    if not histogram_features:
        lfw_pla = np.max(lfw_pla,(2,3)) # spatial
    else:
        lfw_pla = np.reshape(lfw_pla,(600,512*4*4))

    lfw = np.concatenate((lfw_bed, lfw_cel, lfw_ima, lfw_pla),1)
    
    
    # Places target
    with open("activations/Bordeaux/Bedroom_source/Places/activations.txt", "rb") as fp:   # Unpickling
        pla_bed = pickle.load(fp)[0]
    
    if not histogram_features:
        pla_bed = np.max(pla_bed,(2,3)) # spatial
    else:
        pla_bed = np.reshape(pla_bed,(600,512*4*4))

    with open("activations/Bordeaux/Celeba_source/Places/activations.txt", "rb") as fp:   # Unpickling
        pla_cel = pickle.load(fp)[0]
    
    if not histogram_features:
        pla_cel = np.max(pla_cel,(2,3)) # spatial
    else:
        pla_cel = np.reshape(pla_cel,(600,512*4*4))

    with open("activations/Bordeaux/Imagenet_source/Places/activations.txt", "rb") as fp:   # Unpickling
        pla_ima = pickle.load(fp)[0]
    
    if not histogram_features:
        pla_ima = np.max(pla_ima,(2,3)) # spatial
    else:
        pla_ima = np.reshape(pla_ima,(600,512*4*4))

    with open("activations/Bordeaux/Places_source/Places/activations.txt", "rb") as fp:   # Unpickling
        pla_pla = pickle.load(fp)[0]
    
    if not histogram_features:
        pla_pla = np.max(pla_pla,(2,3)) # spatial
    else:
        pla_pla = np.reshape(pla_pla,(600,512*4*4))

    pla = np.concatenate((pla_bed, pla_cel, pla_ima, pla_pla),1)
    
    
    # IPCV target
    with open("activations/Bordeaux/Bedroom_source/IPCV/activations.txt", "rb") as fp:   # Unpickling
        ipc_bed = pickle.load(fp)[0]
    
    if not histogram_features:
        ipc_bed = np.max(ipc_bed,(2,3)) # spatial
    else:
        ipc_bed = np.reshape(ipc_bed,(600,512*4*4))

    with open("activations/Bordeaux/Celeba_source/IPCV/activations.txt", "rb") as fp:   # Unpickling
        ipc_cel = pickle.load(fp)[0]
    
    if not histogram_features:
        ipc_cel = np.max(ipc_cel,(2,3)) # spatial
    else:
        ipc_cel = np.reshape(ipc_cel,(600,512*4*4))

    with open("activations/Bordeaux/Imagenet_source/IPCV/activations.txt", "rb") as fp:   # Unpickling
        ipc_ima = pickle.load(fp)[0]
    
    if not histogram_features:
        ipc_ima = np.max(ipc_ima,(2,3)) # spatial
    else:
        ipc_ima = np.reshape(ipc_ima,(600,512*4*4))

    with open("activations/Bordeaux/Places_source/IPCV/activations.txt", "rb") as fp:   # Unpickling
        ipc_pla = pickle.load(fp)[0]
    
    if not histogram_features:
        ipc_pla = np.max(ipc_pla,(2,3)) # spatial
    else:
        ipc_pla = np.reshape(ipc_pla,(600,512*4*4))

    ipc = np.concatenate((ipc_bed, ipc_cel, ipc_ima, ipc_pla),1)
    

    
    if histogram_features:
        bed_src = np.concatenate((bed_bed, cel_bed, cit_bed, flo_bed, ima_bed, kit_bed, lfw_bed, pla_bed), 0)
        cel_src = np.concatenate((cel_cel, bed_cel, cit_cel, flo_cel, ima_cel, kit_cel, lfw_cel, pla_cel), 0)
        ima_src = np.concatenate((ima_ima, bed_ima, cel_ima, cit_ima, flo_ima, kit_ima, lfw_ima, pla_ima), 0)
        pla_src = np.concatenate((pla_pla, bed_pla, cel_pla, cit_pla, flo_pla, ima_pla, kit_pla, lfw_pla), 0)
   
    
    # extracting histogram based features    
    if histogram_features:
        num_bins = 512
        min_bed = np.min(bed_src)
        max_bed = np.max(bed_src)
        min_cel = np.min(cel_src)
        max_cel = np.max(cel_src)
        min_ima = np.min(ima_src)
        max_ima = np.max(ima_src)
        min_pla = np.min(pla_src)
        max_pla = np.max(pla_src)
        
        bed_bed_hist = np.zeros((num_img,num_bins))
        cel_bed_hist = np.zeros((num_img,num_bins))
        cit_bed_hist = np.zeros((num_img,num_bins))
        flo_bed_hist = np.zeros((num_img,num_bins))
        ima_bed_hist = np.zeros((num_img,num_bins))
        kit_bed_hist = np.zeros((num_img,num_bins))
        lfw_bed_hist = np.zeros((num_img,num_bins))
        pla_bed_hist = np.zeros((num_img,num_bins))
        ipc_bed_hist = np.zeros((num_img,num_bins))
                
        
        bed_cel_hist = np.zeros((num_img,num_bins))
        cel_cel_hist = np.zeros((num_img,num_bins))
        cit_cel_hist = np.zeros((num_img,num_bins))
        flo_cel_hist = np.zeros((num_img,num_bins))
        ima_cel_hist = np.zeros((num_img,num_bins))
        kit_cel_hist = np.zeros((num_img,num_bins))
        lfw_cel_hist = np.zeros((num_img,num_bins))
        pla_cel_hist = np.zeros((num_img,num_bins))
        ipc_cel_hist = np.zeros((num_img,num_bins))
        
        bed_ima_hist = np.zeros((num_img,num_bins))
        cel_ima_hist = np.zeros((num_img,num_bins))
        cit_ima_hist = np.zeros((num_img,num_bins))
        flo_ima_hist = np.zeros((num_img,num_bins))
        ima_ima_hist = np.zeros((num_img,num_bins))
        kit_ima_hist = np.zeros((num_img,num_bins))
        lfw_ima_hist = np.zeros((num_img,num_bins))
        pla_ima_hist = np.zeros((num_img,num_bins))
        ipc_ima_hist = np.zeros((num_img,num_bins))
        
        bed_pla_hist = np.zeros((num_img,num_bins))
        cel_pla_hist = np.zeros((num_img,num_bins))
        cit_pla_hist = np.zeros((num_img,num_bins))
        flo_pla_hist = np.zeros((num_img,num_bins))
        ima_pla_hist = np.zeros((num_img,num_bins))
        kit_pla_hist = np.zeros((num_img,num_bins))
        lfw_pla_hist = np.zeros((num_img,num_bins))
        pla_pla_hist = np.zeros((num_img,num_bins))
        ipc_pla_hist = np.zeros((num_img,num_bins))
        
        
        for k in range(num_img):
            bed_bed_hist[k,:] = np.histogram(bed_bed[k,:],num_bins,(min_bed,max_bed))[0]
            cel_bed_hist[k,:] = np.histogram(cel_bed[k,:],num_bins,(min_bed,max_bed))[0]
            cit_bed_hist[k,:] = np.histogram(cit_bed[k,:],num_bins,(min_bed,max_bed))[0]
            flo_bed_hist[k,:] = np.histogram(flo_bed[k,:],num_bins,(min_bed,max_bed))[0]
            ima_bed_hist[k,:] = np.histogram(ima_bed[k,:],num_bins,(min_bed,max_bed))[0]
            kit_bed_hist[k,:] = np.histogram(kit_bed[k,:],num_bins,(min_bed,max_bed))[0]
            lfw_bed_hist[k,:] = np.histogram(lfw_bed[k,:],num_bins,(min_bed,max_bed))[0]
            pla_bed_hist[k,:] = np.histogram(pla_bed[k,:],num_bins,(min_bed,max_bed))[0]
            ipc_bed_hist[k,:] = np.histogram(ipc_bed[k,:],num_bins,(min_bed,max_bed))[0]
            
            
            bed_cel_hist[k,:] = np.histogram(bed_cel[k,:],num_bins,(min_cel,max_cel))[0]
            cel_cel_hist[k,:] = np.histogram(cel_cel[k,:],num_bins,(min_cel,max_cel))[0]
            cit_cel_hist[k,:] = np.histogram(cit_cel[k,:],num_bins,(min_cel,max_cel))[0]
            flo_cel_hist[k,:] = np.histogram(flo_cel[k,:],num_bins,(min_cel,max_cel))[0]
            ima_cel_hist[k,:] = np.histogram(ima_cel[k,:],num_bins,(min_cel,max_cel))[0]
            kit_cel_hist[k,:] = np.histogram(kit_cel[k,:],num_bins,(min_cel,max_cel))[0]
            lfw_cel_hist[k,:] = np.histogram(lfw_cel[k,:],num_bins,(min_cel,max_cel))[0]
            pla_cel_hist[k,:] = np.histogram(pla_cel[k,:],num_bins,(min_cel,max_cel))[0]
            ipc_cel_hist[k,:] = np.histogram(ipc_cel[k,:],num_bins,(min_cel,max_cel))[0]
            
            
            bed_ima_hist[k,:] = np.histogram(bed_ima[k,:],num_bins,(min_ima,max_ima))[0]
            cel_ima_hist[k,:] = np.histogram(cel_ima[k,:],num_bins,(min_ima,max_ima))[0]
            cit_ima_hist[k,:] = np.histogram(cit_ima[k,:],num_bins,(min_ima,max_ima))[0]
            flo_ima_hist[k,:] = np.histogram(flo_ima[k,:],num_bins,(min_ima,max_ima))[0]
            ima_ima_hist[k,:] = np.histogram(ima_ima[k,:],num_bins,(min_ima,max_ima))[0]
            kit_ima_hist[k,:] = np.histogram(kit_ima[k,:],num_bins,(min_ima,max_ima))[0]
            lfw_ima_hist[k,:] = np.histogram(lfw_ima[k,:],num_bins,(min_ima,max_ima))[0]
            pla_ima_hist[k,:] = np.histogram(pla_ima[k,:],num_bins,(min_ima,max_ima))[0]
            ipc_ima_hist[k,:] = np.histogram(ipc_ima[k,:],num_bins,(min_ima,max_ima))[0]
            
            
            bed_pla_hist[k,:] = np.histogram(bed_pla[k,:],num_bins,(min_pla,max_pla))[0]
            cel_pla_hist[k,:] = np.histogram(cel_pla[k,:],num_bins,(min_pla,max_pla))[0]
            cit_pla_hist[k,:] = np.histogram(cit_pla[k,:],num_bins,(min_pla,max_pla))[0]
            flo_pla_hist[k,:] = np.histogram(flo_pla[k,:],num_bins,(min_pla,max_pla))[0]
            ima_pla_hist[k,:] = np.histogram(ima_pla[k,:],num_bins,(min_pla,max_pla))[0]
            kit_pla_hist[k,:] = np.histogram(kit_pla[k,:],num_bins,(min_pla,max_pla))[0]
            lfw_pla_hist[k,:] = np.histogram(lfw_pla[k,:],num_bins,(min_pla,max_pla))[0]
            pla_pla_hist[k,:] = np.histogram(pla_pla[k,:],num_bins,(min_pla,max_pla))[0]
            ipc_pla_hist[k,:] = np.histogram(ipc_pla[k,:],num_bins,(min_pla,max_pla))[0]
            
        
        bed_bed = bed_bed_hist
        cel_bed = cel_bed_hist
        cit_bed = cit_bed_hist
        flo_bed = flo_bed_hist
        ima_bed = ima_bed_hist
        kit_bed = kit_bed_hist
        lfw_bed = lfw_bed_hist
        pla_bed = pla_bed_hist
        ipc_bed = ipc_bed_hist
        
        bed_cel = bed_cel_hist
        cel_cel = cel_cel_hist
        cit_cel = cit_cel_hist
        flo_cel = flo_cel_hist
        ima_cel = ima_cel_hist
        kit_cel = kit_cel_hist
        lfw_cel = lfw_cel_hist
        pla_cel = pla_cel_hist
        ipc_cel = ipc_cel_hist
        
        bed_ima = bed_ima_hist
        cel_ima = cel_ima_hist
        cit_ima = cit_ima_hist
        flo_ima = flo_ima_hist
        ima_ima = ima_ima_hist
        kit_ima = kit_ima_hist
        lfw_ima = lfw_ima_hist
        pla_ima = pla_ima_hist
        ipc_ima = ipc_ima_hist
        
        bed_pla = bed_pla_hist
        cel_pla = cel_pla_hist
        cit_pla = cit_pla_hist
        flo_pla = flo_pla_hist
        ima_pla = ima_pla_hist
        kit_pla = kit_pla_hist
        lfw_pla = lfw_pla_hist
        pla_pla = pla_pla_hist
        ipc_pla = ipc_pla_hist
        
        del bed_bed_hist, bed_cel_hist, bed_ima_hist, bed_pla_hist
        del cel_bed_hist, cel_pla_hist, cel_ima_hist, cel_cel_hist
        del cit_bed_hist, cit_pla_hist, cit_ima_hist, cit_cel_hist
        del flo_bed_hist, flo_pla_hist, flo_ima_hist, flo_cel_hist
        del ima_bed_hist, ima_pla_hist, ima_ima_hist, ima_cel_hist
        del kit_bed_hist, kit_pla_hist, kit_ima_hist, kit_cel_hist
        del lfw_bed_hist, lfw_pla_hist, lfw_ima_hist, lfw_cel_hist    
        del pla_bed_hist, pla_pla_hist, pla_ima_hist, pla_cel_hist
        del ipc_bed_hist, ipc_pla_hist, ipc_ima_hist, ipc_cel_hist
        
        cit = np.concatenate((cit_bed, cit_cel, cit_ima, cit_pla),1)
        cel = np.concatenate((cel_bed, cel_cel, cel_ima, cel_pla),1)
        bed = np.concatenate((bed_bed, bed_cel, bed_ima, bed_pla),1)
        kit = np.concatenate((kit_bed, kit_cel, kit_ima, kit_pla),1)
        lfw = np.concatenate((lfw_bed, lfw_cel, lfw_ima, lfw_pla),1)
        pla = np.concatenate((pla_bed, pla_cel, pla_ima, pla_pla),1)
        flo = np.concatenate((flo_bed, flo_cel, flo_ima, flo_pla),1)
        ima = np.concatenate((ima_bed, ima_cel, ima_ima, ima_pla),1)
        ipc = np.concatenate((ipc_bed, ipc_cel, ipc_ima, ipc_pla),1)
        
            
    del bed_bed, bed_cel, bed_ima, bed_pla
    del cel_bed, cel_pla, cel_ima, cel_cel
    del cit_bed, cit_pla, cit_ima, cit_cel
    del flo_bed, flo_pla, flo_ima, flo_cel
    del ima_bed, ima_pla, ima_ima, ima_cel
    del kit_bed, kit_pla, kit_ima, kit_cel
    del lfw_bed, lfw_pla, lfw_ima, lfw_cel    
    del pla_bed, pla_pla, pla_ima, pla_cel
    del ipc_bed, ipc_pla, ipc_ima, ipc_cel
     

    
    reducer = umap.UMAP()
    
    if supervised:
        num_train = int(5/6 * num_img)
        num_test = int(1/6 * num_img)
        bed_train = bed[0:num_train,:]
        cel_train = cel[0:num_train,:]
        cit_train = cit[0:num_train,:]
        flo_train = flo[0:num_train,:]
        ima_train = ima[0:num_train,:]
        kit_train = kit[0:num_train,:]
        lfw_train = lfw[0:num_train,:]
        pla_train = pla[0:num_train,:]
        comb_train = np.concatenate((bed_train,cel_train,cit_train,flo_train,ima_train,kit_train,lfw_train,pla_train),0)
        
        bed_test = bed[num_train:,:]
        cel_test = cel[num_train:,:]
        cit_test = cit[num_train:,:]
        flo_test = flo[num_train:,:]
        ima_test = ima[num_train:,:]
        kit_test = kit[num_train:,:]
        lfw_test = lfw[num_train:,:]
        pla_test = pla[num_train:,:]
        comb_test = np.concatenate((bed_test,cel_test,cit_test,flo_test,ima_test,kit_test,lfw_test,pla_test),0)
        
        
        y = []
        for k in range(0,len(comb_train)):
            color = k//num_train
            y.append(color)
        y = np.array(y)
        
        reducer.fit(comb_train,y)
        
        embedding = reducer.transform(comb_test)
        
        bed_emb = embedding[0:num_test,:]    
        cel_emb = embedding[num_test:2*num_test,:]
        cit_emb = embedding[2*num_test:3*num_test,:]
        flo_emb = embedding[3*num_test:4*num_test,:]
        ima_emb = embedding[4*num_test:5*num_test,:]
        kit_emb = embedding[5*num_test:6*num_test,:]
        lfw_emb = embedding[6*num_test:7*num_test,:]
        pla_emb = embedding[7*num_test:8*num_test,:]

    else:
        comb = np.concatenate((bed,cel,cit,flo,ima,kit,lfw,pla, ipc),0)
        embedding = reducer.fit_transform(comb)    
        bed_emb = embedding[0:num_img,:]
        cel_emb = embedding[num_img:2*num_img,:]
        cit_emb = embedding[2*num_img:3*num_img,:]
        flo_emb = embedding[3*num_img:4*num_img,:]
        ima_emb = embedding[4*num_img:5*num_img,:]
        kit_emb = embedding[5*num_img:6*num_img,:]
        lfw_emb = embedding[6*num_img:7*num_img,:]
        pla_emb = embedding[7*num_img:8*num_img,:]
        ipc_emb = embedding[8*num_img:9*num_img,:]
    

    
    colors = ['b', 'c', 'y', 'm', 'r', 'p', 'k', 'g', '']
    plt.figure()
    
    
    sc1 = plt.scatter(bed_emb[:,0], bed_emb[:,1], s=14, marker='o', color=sns.color_palette()[0])
    sc2 = plt.scatter(cel_emb[:,0], cel_emb[:,1], s=14, marker='o', color=sns.color_palette()[1])
    sc3 = plt.scatter(cit_emb[:,0], cit_emb[:,1], s=14, marker='o', color=sns.color_palette()[2])
    sc4 = plt.scatter(flo_emb[:,0], flo_emb[:,1], s=14, marker='o', color=sns.color_palette()[3])
    sc5 = plt.scatter(ima_emb[:,0], ima_emb[:,1], s=14, marker='o', color=sns.color_palette()[4])
    sc6 = plt.scatter(kit_emb[:,0], kit_emb[:,1], s=14, marker='o', color=sns.color_palette()[5])
    sc7 = plt.scatter(lfw_emb[:,0], lfw_emb[:,1], s=14, marker='o', color=sns.color_palette()[6])
    sc8 = plt.scatter(pla_emb[:,0], pla_emb[:,1], s=14, marker='o', color=sns.color_palette()[7])
    if not supervised:
        sc9 = plt.scatter(ipc_emb[:,0], ipc_emb[:,1], s=14, marker='o', color=sns.color_palette()[8])
    
    
    if not supervised:    
        plt.legend((sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9),
                   ('Bedrooms', 'Celeba', 'Cityscapes', 'Flowers', 'Imagenet', 'Kitchens', 'LFW', 'Places', 'IPCV'),
                   scatterpoints=3,
                   loc='best',
                   ncol=3,
                   fontsize=10)
    else:
        plt.legend((sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8),
                   ('Bedrooms', 'Celeba', 'Cityscapes', 'Flowers', 'Imagenet', 'Kitchens', 'LFW', 'Places'),
                   scatterpoints=3,
                   loc='best',
                   ncol=3,
                   fontsize=10)
    plt.title('UMAP for Discriminator Layer ' + str(5))
    plt.show()
    
    
    bed_cen = np.mean(bed_emb,0)
    cel_cen = np.mean(cel_emb,0)
    cit_cen = np.mean(cit_emb,0)
    flo_cen = np.mean(flo_emb,0)
    ima_cen = np.mean(ima_emb,0)
    kit_cen = np.mean(kit_emb,0)
    lfw_cen = np.mean(lfw_emb,0)
    pla_cen = np.mean(pla_emb,0)
    if not supervised:
        ipc_cen = np.mean(ipc_emb,0)
    
    
    bed_cov = np.cov(np.transpose(bed_emb))
    cel_cov = np.cov(np.transpose(cel_emb))
    cit_cov = np.cov(np.transpose(cit_emb))
    flo_cov = np.cov(np.transpose(flo_emb))
    ima_cov = np.cov(np.transpose(ima_emb))
    kit_cov = np.cov(np.transpose(kit_emb))
    lfw_cov = np.cov(np.transpose(lfw_emb))
    pla_cov = np.cov(np.transpose(pla_emb))
    if not supervised: 
        ipc_cov = np.cov(np.transpose(ipc_emb))
    
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
    
    if not supervised:
        ipc_bed = np.linalg.norm(ipc_cen-bed_cen)
        ipc_cel = np.linalg.norm(ipc_cen-cel_cen)
        ipc_ima = np.linalg.norm(ipc_cen-ima_cen)
        ipc_pla = np.linalg.norm(ipc_cen-pla_cen)
        
    cit_bed_mle = mle_estimation(cit_emb,bed_cen,bed_cov)
    cit_cel_mle = mle_estimation(cit_emb,cel_cen,cel_cov)
    cit_ima_mle = mle_estimation(cit_emb,ima_cen,ima_cov)
    cit_pla_mle = mle_estimation(cit_emb,pla_cen,pla_cov)
    
    flo_bed_mle = mle_estimation(flo_emb,bed_cen,bed_cov)
    flo_cel_mle = mle_estimation(flo_emb,cel_cen,cel_cov)
    flo_ima_mle = mle_estimation(flo_emb,ima_cen,ima_cov)
    flo_pla_mle = mle_estimation(flo_emb,pla_cen,pla_cov)
    
    
    lfw_bed_mle = mle_estimation(lfw_emb,bed_cen,bed_cov)
    lfw_cel_mle = mle_estimation(lfw_emb,cel_cen,cel_cov)
    lfw_ima_mle = mle_estimation(lfw_emb,ima_cen,ima_cov)
    lfw_pla_mle = mle_estimation(lfw_emb,pla_cen,pla_cov)
    
    kit_bed_mle = mle_estimation(kit_emb,bed_cen,bed_cov)
    kit_cel_mle = mle_estimation(kit_emb,cel_cen,cel_cov)
    kit_ima_mle = mle_estimation(kit_emb,ima_cen,ima_cov)
    kit_pla_mle = mle_estimation(kit_emb,pla_cen,pla_cov)
    
    source = np.concatenate((ima_emb,pla_emb,bed_emb,cel_emb),0)
    
    flo_knn = knn_estimation(flo_emb, source, 20)
    lfw_knn = knn_estimation(lfw_emb, source, 20)
    kit_knn = knn_estimation(kit_emb, source, 20)
    cit_knn = knn_estimation(cit_emb, source, 20)