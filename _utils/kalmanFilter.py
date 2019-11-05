#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:20:19 2019

@author: ulises
"""


import numpy as np

from ._imgTools import mapTompipi
from .ModelsSinTH_models import ModeloCoordenadas, Gps_aXYZ



class ParticleFilter(object):
	eye = np.eye(3)
	def __init__(self,x0=[0,0,0,0],dt=.1,Cx=eye,
						Cu=eye, Cmed=eye, model=None,cpars=[None], ll0 = None):
		self.nParts = 300
		x0 = np.array(self.nParts*[x0]).T

		sVar = {'xPrior':[x0] ,'xPost':[x0],
				'Cx':[Cx],'u':[],'Cu':[Cu],
				'med':[],'Cmed':[Cmed],'dt':dt,'innov':[]}

		[setattr(self,atr,val) for atr,val in sVar.items()]
		if ll0 is not None:
			self.medModel = Gps_aXYZ(ll0=ll0)
		else:
			self.medModel = Gps_aXYZ()
		self.model = model if model is not None else ModeloCoordenadas(x0,cpars)
		self.L = [np.linalg.inv(Cmed)]
		self.z = []

	def _prior(self):
		cvars = ['xPost','Cx','Cu','u']
		xp,Cx,Cu,u = [getattr(self,thisVar)[-1] for thisVar in cvars]
		n = self.nParts

		modelCalls = ['applyModel','getFx','getFu']
		xprior,Fx,Fu = [getattr(self.model,call)(xp,u) for call in modelCalls]

		su,sx = map(lambda C: np.sqrt(np.diag(C))[:,np.newaxis] , (Cu,Cx))

		inpNoise,modNoise = map(lambda F,s: F.dot(s*np.random.randn(len(s),n)),
								 (Fu,Fx),(su,sx))

		xprior = xprior + inpNoise +modNoise
		self.xPrior.append(xprior)


	def _post(self):
		cvars = ['xPrior','Cx','med','Cmed','L']
		xp,Cx,z,Cmed,L = [getattr(self,thisVar)[-1] for thisVar in cvars]
		n = self.nParts

		modelCalls = ['getMx','getMed']
		Fm,medPrior = [getattr(self.model,call)(xp) for  call in modelCalls]

		z[-1] = mapTompipi(z[-1])
		self.z.append(z)
		#medPrior[-1] = mapTompipi(medPrior[-1])
		innov 	= z[:,np.newaxis] - medPrior


		normL = np.array([(x.dot(L)).dot(x) for x in  innov.T ])


		w   = np.e**(-normL) +1e-17

		wn = w/w.sum()
		Wcum = np.cumsum(wn)
		r = np.random.rand(n)
		d = np.abs(Wcum[np.newaxis,:]-r[:,np.newaxis])

		partSel = np.argmin(d,axis=1)
		xPost  = xp[:,partSel] #+ 0.1*(Cx[:,np.newaxis,:]*\
					 #np.random.randn(3,n,1)).sum(-1)
		#xPost[-1] = mapTompipi(xPost[-1])

		Cx = np.cov(xPost)

		self.xPost.append(xPost)

		self.innov.append(innov)

	def _postSinMed(self):
		self.xPost.append(self.xPrior[-1])
		self.Cx.append(self.Cx[-1])


	def set_U_Med(self,u,z=None):
		self.u.append(u)

		if z is not None:
			oldz,newz = z
			med = self.medModel(self.xPost[-1].mean(-1),oldz,newz)
			self.med.append(med)

	def runOneIt(self,u,z=None):
		self.set_U_Med(u,z)
		self._prior()
		if z is not None:
			self._post()
		else:
			self._postSinMed()



















class KalmanFilter(object):
	eye = np.eye(3)
	def __init__(self,x0=[0,0,0],dt=.1,Cx=eye,
						Cu=eye, Cmed=eye, model=None):

		sVar = {'xPrior':[x0] ,'xPost':[x0],
				'CxPrior':[Cx],'CxPost':[Cx],
				'u':[],'Cu':[Cu],'med':[],'Cmed':[Cmed],
				'dt':dt,'innov':[],'KalmanGain':[]}

		[setattr(self,atr,val) for atr,val in sVar.items()]
		self.model = model if model is not None else ModeloCoordenadas(x0)


	def _prior(self):
		cvars = ['xPost','CxPost','Cu','u']
		xp,Cx,Cu,u = [getattr(self,thisVar)[-1] for thisVar in cvars]

		modelCalls = ['applyModel','getFx','getFu']
		xprior,Fx,Fu = [getattr(self.model,call)(xp,u) for
							  call in modelCalls]

		Cx = np.dot(np.dot(Fx,Cx),Fx.T) +\
				Cu

		self.xPrior.append(xprior)
		self.CxPrior.append(Cx)

	def _post(self):
		xp = self.xPrior[-1]
		Cx = self.CxPrior[-1]

		z 		= self.med[-1]
		Cmed 	= self.Cmed[-1]
		Fm 		= self.model.getMx(xp)
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


