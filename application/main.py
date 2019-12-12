# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:56:20 2019

@author: Nadja
"""

# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns




    
class MatplotlibWidget(QDialog):
    
    def __init__(self):
        
        QDialog.__init__(self)
        loadUi("qt_with_plot.ui",self)
        self.pushButton.clicked.connect(self.update_graph)
       
    
        
    def update_graph(self):
        i=int(self.comboBox.currentIndex())
        if i==0:
            from layer5_umapping import bed_emb, cel_emb, cit_emb, flo_emb, ima_emb, kit_emb, lfw_emb, pla_emb
            self.MplWidget.canvas.axes.set_title('UMAP for Discriminator Layer ' + str(5))
        else:
            from umapping import calc_umap
            bed_emb, cel_emb, cit_emb, flo_emb, ima_emb, kit_emb, lfw_emb, pla_emb = calc_umap(i-1)
            self.MplWidget.canvas.axes.set_title('UMAP for Discriminator Layer ' + str(i))
        



        sc1 = self.MplWidget.canvas.axes.scatter(bed_emb[:,0], bed_emb[:,1], s=14, marker='o', color=sns.color_palette()[0])
        sc2 = self.MplWidget.canvas.axes.scatter(cel_emb[:,0], cel_emb[:,1], s=14, marker='o', color=sns.color_palette()[1])
        sc3 = self.MplWidget.canvas.axes.scatter(cit_emb[:,0], cit_emb[:,1], s=14, marker='o', color=sns.color_palette()[2])
        sc4 = self.MplWidget.canvas.axes.scatter(flo_emb[:,0], flo_emb[:,1], s=14, marker='o', color=sns.color_palette()[3])
        sc5 = self.MplWidget.canvas.axes.scatter(ima_emb[:,0], ima_emb[:,1], s=14, marker='o', color=sns.color_palette()[4])
        sc6 = self.MplWidget.canvas.axes.scatter(kit_emb[:,0], kit_emb[:,1], s=14, marker='o', color=sns.color_palette()[5])
        sc7 = self.MplWidget.canvas.axes.scatter(lfw_emb[:,0], lfw_emb[:,1], s=14, marker='o', color=sns.color_palette()[6])
        sc8 = self.MplWidget.canvas.axes.scatter(pla_emb[:,0], pla_emb[:,1], s=14, marker='o', color=sns.color_palette()[7])
        
            
        self.MplWidget.canvas.axes.legend((sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8),
                   ('Bedrooms', 'Celeba', 'Cityscapes', 'Flowers', 'Imagenet', 'Kitchens', 'LFW', 'Places'),
                   scatterpoints=3,
                   loc='best',
                   ncol=2,
                   fontsize=10)
        
        self.MplWidget.canvas.draw()



app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()