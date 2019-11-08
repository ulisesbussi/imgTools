#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:51:17 2019

@author: ulises
"""
import numpy as np


def GexCalc(img):
	#Excess Green[1]
	ch_r,ch_g,ch_b = img.T # r, g, b
	greenIndex  = 2.0*ch_g- ch_r-ch_b
	return greenIndex.T

def RexCalc(img):
	#Excess Red[1]
	ch_r,ch_g,ch_b = img.T # r, g, b
	redIndex  = 1.4*ch_r-1.0*ch_g
	return redIndex.T

def CiveCalc(img):
	#Color Index of Vegetation[1]

	ch_r,ch_g,ch_b = img.T
	cive = 0.881*ch_g - 0.441*ch_r - 0.385* ch_b - 18.78745
	return cive.T

def NdiCalc(img):
	#Normalized Difference Index[1]
	ch_r,ch_g,ch_b = img.T
	den = (1.0*ch_g + ch_r )
	den[den==0]=100
	ndi = (1.0*ch_g - ch_r )/den
	return ndi.T


def VariCalc(img):
	#Visible Atmospheric Resistant Index[2]
	ch_r,ch_g,ch_b = img.T
	gamma=.1
	q = 1.0*ch_g + ch_r -ch_b
	q[q==0] = gamma
	vari = (1.0*ch_g - ch_r )/(q)
	return vari.T


def TgiCalc(img):
	#Triangular Greenness Index[2]
	ch_r,ch_g,ch_b = img.T
	tgi = 1.0*ch_g - 0.39*ch_r -0.61*ch_b
	return tgi.T

""" 
[1]  Milioto, A., Lottes, P., & Stachniss, C. (2018, May). 
	Real-time semantic segmentation of crop and weed for 
	precision agriculture robots leveraging background knowledge in CNNs. 
	In 2018 IEEE International Conference on 
	Robotics and Automation (ICRA) (pp. 2229-2235). IEEE.
[2]  Mckinnon, T., & Hoff, P. (2017). 
	Comparing RGB-based vegetation indices with NDVI for drone based agricultural sensing. Agribotix. Com, 1-8.
"""


#%%

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	imgT = plt.imread('foto.png')
	plt.figure();plt.title('Gex');plt.imshow(GexCalc(imgT));
	plt.figure();plt.title('Rex');plt.imshow(RexCalc(imgT));
	plt.figure();plt.title('Cive');plt.imshow(CiveCalc(imgT));
	plt.figure();plt.title('Ndi');plt.imshow(NdiCalc(imgT));
	plt.figure();plt.title('Vari');plt.imshow(VariCalc(imgT));
	plt.figure();plt.title('Tgi');plt.imshow(TgiCalc(imgT));



