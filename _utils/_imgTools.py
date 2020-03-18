#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:43:56 2019

@author: ulises
"""

import numpy as np
import pandas as pd


def LLtoMeters(oldla,oldlo,la,lo):
	"""Ellipsoidal Earth projected to a plane
	The FCC prescribes the following formulae for distances not exceeding 475 kilometres (295 mi):[2]"""
	deltaLat = la-oldla
	mlat = (la+oldla)/2
	deltaLon = lo-oldlo

	mlaR = np.deg2rad(mlat)
	k1 = 	111.13209- 0.56605*np.cos(2*mlaR)+\
					 0.00012*np.cos(4*mlaR)
	k2 = 	111.41513*np.cos(mlaR)-\
				0.09455*np.cos(3*mlaR)+\
				0.00012*np.cos(5*mlaR)
	dx =  k2 *deltaLon  *1000 #from km to m
	dy = k1 *deltaLat  *1000
	return dx,dy


def mapTompipi(nums):
	if hasattr(nums,'__len__'):
		for i,num in enumerate(nums):
			if num<-np.pi:
				num = num - 2*np.pi*np.ceil(num/(np.pi))
			elif num>np.pi:
				num = num - 2*np.pi*np.floor(num/(np.pi))
			nums[i] = num
	else:
		if nums<-np.pi:
			nums = nums - 2*np.pi*np.ceil(nums/(np.pi))
		elif nums>np.pi:
			nums = nums - 2*np.pi*np.floor(nums/(np.pi))
	return nums



def get_sRT_fromAffine(affineMatrix):
	T = affineMatrix[:,2]
	sRot =affineMatrix[:2,:2]
	s = np.sqrt(np.linalg.det(sRot))
	R = sRot/s
	return s,R,T

def get_s_theta_T_fromAffine(affineMatrix):
	affineMatrix = np.array(affineMatrix)
	if affineMatrix.ndim == 2:
		s,R,T = get_sRT_fromAffine(affineMatrix)
		theta = np.arctan2(*R[::-1,0])
		return s, theta, T[:-1]
	elif affineMatrix.ndim == 3:
		s 		=[]
		T 		=[]
		theta 	=[]
		for afi in affineMatrix:
			_s,_R,_T = get_sRT_fromAffine(afi)
			_theta = np.arctan2(*_R[::-1,0])
			s.append(_s)
			T.append(_T)
			theta.append(_theta)
		s 		= np.array(s)
		T 		= np.array(T)
		theta 	= np.array(theta)
		return s, theta, T[:,:-1]

	else:
		raise IOError('la entrada debe ser una matriz de 3x3, o una lista de matrices,o un array de Nx3x3')




def get_Affine_From_s_theta_T(s,theta,T):
	#theta = theta.item()
	T = T.reshape(2,1)
	R = np.array([[np.cos(theta),-np.sin(theta)],
				  [np.sin(theta), np.cos(theta)] ])
	sR = np.dot(s,R)
	sRT = np.concatenate([sR,T],axis=1)
	return sRT







def createDataFramesubt(subt):
	subt = subt.split('\n\n')
	df_subt = pd.DataFrame(columns = ['frameNumber','dateTime','lat','lon'])
	for i in range(len(subt)):
		l 			= subt[i].split('[')
		if l[0] != '':
			l0sp 		= l[0].split('\n')
			datetime 	= ','.join(l0sp[-2].split(',')[:-1])
			frNum 		= np.int32(l0sp[2].split(':')[1].split(',')[0])
			la 			= l[-2].replace(']','').replace(' ','').split(':')[-1]
			lo 			= l[-1].replace(']','').replace(' ','').split(':')[-1].replace('</font>','')
			df_subt = df_subt.append({'frameNumber':frNum,
							 'dateTime':pd.to_datetime(datetime),
							 'lat':np.float(la),
							 'lon':np.float(lo)},ignore_index=True)
	return df_subt












def cumdot(a):
	aux = [np.eye(len(a[0]))]
	for el in a:
		aux.append(aux[-1].dot(el))
	return np.array(aux)