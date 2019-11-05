#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:24:07 2019
DeprecatedVersion KalmanFilter
@author: ulises
"""

import numpy as np





"""oldVErsion without comprehensions"""


class KalmanFilter(object):
	def __init__(self,x0=[0,0,0],dt=.1,Cx=np.eye(3),
				  Cu=np.eye(3),Cmed=np.eye(3), model=None):

		self.xPrior 		= [x0]
		self.xPost 		= [x0]
		self.CxPrior 	= [Cx]
		self.CxPost  	= [Cx]
		self.med 		= []
		self.u 			= []
		self.Cu 			= [Cu]
		self.Cmed 		= [Cmed]
		self.dt 			= dt


		if model is not None:
 			self.model = model
		else:
 			self.model = ModeloConSeba(x0)

		self.innov 		= []
		self.KalmanGain = []


 	def _prior(self):
		xp = self.xPost[-1]
		Cx = self.CxPost[-1]
		Cu = self.Cu[-1]
		u  = self.u[-1]
		#dt = self.dt

		xprior = self.model.applyModel(xp,u)
		Fx = self.model.getFx(xp,u)
		Fu = self.model.getFu(xp,u)

		Cx = np.dot(np.dot(Fx,Cx),Fx.T) +\
				Cu

		self.xPrior.append(xprior)
		self.CxPrior.append(Cx)

 	def _post(self):
		xp = self.xPrior[-1]
		Cx = self.CxPrior[-1]

		z 		= self.med[-1]
		Cmed 	= self.Cmed[-1]
		Fm 		= self.model.getMx()
		innov 	= z - self.model.getMed()

		#innov[-1] = (innov[-1]+np.pi)%(2*np.pi)-np.pi
		S = np.dot( np.dot(Fm,Cx) ,Fm.T) + Cmed
		if len(S)>1:
 			invS = np.linalg.inv(S)
		else:
 			invS = 1/S

		K = np.dot(np.dot(Cx,Fm.T),invS)
		self.innov.append(innov)
		self.KalmanGain.append(K)
		xp += np.dot(K,innov)
		kh  = np.dot(K,Fm)
		if hasattr(kh,'__len__'):
 			Cx = np.dot((np.eye(*kh.shape)-kh) , Cx)
		else:
 			Cx = np.dot(1-kh , Cx)

		self.xPost.append(xp)
		self.CxPost.append(Cx)

 	def _postSinMed(self):
		self. xPost.append(self.xPrior[-1])
		self.CxPost.append(self.CxPrior[-1])


 	def set_U_Med(self,u,z=None):
		self.u.append(u)
		if z is not None:
 			self.med.append(z)

 	def runOneIt(self,u,z=None):
		self.set_U_Med(u,z)
		self._prior()
		if z is not None:
 			self._post()
		else:
 			self._postSinMed()
