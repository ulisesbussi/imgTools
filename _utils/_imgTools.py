#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:43:56 2019

@author: ulises
"""

import numpy as np



def get_sRT_fromAffine(affineMatrix):
	T = affineMatrix[:,-1]
	sRot =affineMatrix[:2,:2]
	s = np.sqrt(np.linalg.det(sRot))
	R = sRot/s
	return s,R,T

def get_s_theta_T_fromAffine(affineMatrix):
	affineMatrix = np.array(affineMatrix)
	if affineMatrix.ndim == 2:
		s,R,T = get_sRT_fromAffine(affineMatrix)
		theta = np.arctan2(*R[::-1,0])
	elif affineMatrix.ndim == 3:
		s 		=[]
		R 		=[]
		theta 	=[]
		for afi in affineMatrix:
			_s,_R,T = get_sRT_fromAffine(afi)
			_theta = np.arctan2(*_R[::-1,0])
			s.append(_s)
			R.append(_R)
			theta.append(_theta)
		s 		= np.array(s)
		R 		= np.array(R)
		theta 	= np.array(theta)
	else:
		raise IOError('la entrada debe ser una matriz de 3x3, o una lista de matrices,o un array de Nx3x3')
	return s, theta, T[:-1]



def get_Affine_From_s_theta_T(s,theta,T):
	theta = theta.item()
	T = T.reshape(2,1)
	R = np.array([[np.cos(theta),-np.sin(theta)],
				  [np.sin(theta), np.cos(theta)] ])
	sR = np.dot(s,R)
	sRT = np.concatenate([sR,T],axis=1)
	return sRT





def cumdot(a):
	ou = np.eye(len(a[0]))
	aux = []
	for el in a:
		ou = ou.dot(el)
		aux.append(ou)
	return np.array(aux)