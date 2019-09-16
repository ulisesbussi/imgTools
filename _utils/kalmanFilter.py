#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:20:19 2019

@author: ulises
"""


import numpy as np



class ModeloTh(object):
	def __init__(self):
		pass

	def __call__(self,x,u,dt):
		return self.applyModel(x,u,dt)

	def applyModel(self,x,u,dt):
		xn = x +u #+ w*dt
		return xn

	def getFx(self,x,u,dt):
		return np.array([1])

	def getFu(self,x,u,dt):
		return np.array([dt])

	def getMed(self,x,dt):
		#medicion
		return x

	def getMx(self,x,dt):
		#jacobiano de Med respecto x
		return np.array([1])

	def getR(self):
		#ruido de medicion
		return np.array([2])

class KalmanFilter(object):
	def __init__(self,x0=0,dt=.1,Cx=1,Cu=0,
				  model=None):

		self.xPrior = [x0]
		self.xPost = [x0]

		self.CxPrior = [np.array([Cx])]
		self.CxPost = [np.array([Cx])]

		self.med 	= []
		self.u 		= []
		self.Cu 		= [Cu]
		self.dt 		= dt
		if model is not None:
			self.model = model
		else:
			self.model = ModeloTh()

		self.innov 		= []
		self.KalmanGain = []


	def _prior(self):
		xp = self.xPost[-1]
		Cx = self.CxPost[-1]
		Cu = self.Cu[-1]
		u  = self.u[-1]
		dt = self.dt

		xprior = self.model(xp,u,dt)
		Fx = self.model.getFx(xp,u,dt)
		Fu = self.model.getFu(xp,u,dt)
		Cx = np.dot(np.dot(Fx,Cx),Fx.T) +\
				np.dot(np.dot(Fu,Cu),Fu.T)

		self.xPrior.append(xprior)
		self.CxPrior.append(Cx)

	def _post(self):
		xp = self.xPrior[-1]
		Cx = self.CxPrior[-1]
		dt = self.dt
		z  = self.med[-1]
		R  = self.model.getR()
		Fm = self.model.getMx(xp,dt)

		innov = z - self.model.getMed(xp,dt)
		S = np.dot( np.dot(Fm,Cx) ,Fm.T) + R
		if len(S)>1:
			K = np.dot(np.dot(Cx,Fm.T),np.linalg.inv(S))
		else:
			K = np.dot(np.dot(Cx,Fm.T), 1/S)
		self.innov.append(innov)
		self.KalmanGain.append(K)
		xp = xp + np.dot(K,innov)
		kh = np.dot(K,Fm)
		if hasattr(kh,'__len__'):
			Cx = np.dot((np.eye(kh.shape)-kh) , Cx)
		else:
			Cx = np.dot(1-kh , Cx)

		self.xPost.append(xp)
		self.CxPost.append(Cx)

	def set_U_Med(self,u,z):
		self.u.append(u)
		self.med.append(z)

	def runOneIt(self,u,z):
		self.set_U_Med(u,z)
		self._prior()
		self._post()

