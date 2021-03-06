#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:20:19 2019

@author: ulises
"""


import numpy as np

#from ._imgTools import mapTompipi
from .ModelsenWorld import ModeloCoordenadas, Gps_aXYZ,ModeloTh




# class ParticleFilter():
# 	eye = np.eye(3)
# 	def __init__(self,x0=[0,0,0,0],dt=.1,Cx=eye,
# 			  Cu=eye,Cmed=eye,model=None,cpars=None,
# 			  ll0 = None)):
# 		self.nParts=300
# 		x0 = np.array(self.nParts*[x0])
# 		self.xPrior = [x0]
# 		self.xPost = [x0]

# 		if ll0 is not None:
# 			self.medModel = Gps_aXYZ(ll0=ll0)
# 		else:
# 			self.medModel = Gps_aXYZ()
# 			
# 		self.model = ModeloCoordenadas(x0,pars=cpars)
# 		
# 		self.L = [np.linalg.inv(Cmed)]
# 		self.z = []
# 		self.w_c_T = [] #guardo la transformacion de la cam al mundo


# 	def _prior(self):
# 		xp = self.xPrior[-1]
# 		u  = self.u[-1]
# 		
# 	def _post(self):











class ParticleFilter(): 
	eye = np.eye(3)
	def __init__(self,x0=[0,0,0,0],dt=.1,Cx=eye,
						Cu=eye, Cmed=eye, model=None,cpars=[None], ll0 = None):
		self.nParts = 300
		x0 = np.array(self.nParts*[x0])
		#'xPrior':[x0] ,'xPost':[x0],
		sVar = {'Cx':[Cx],'u':[],'Cu':[Cu],
				'med':[],'Cmed':[Cmed],'dt':dt,'innov':[]}

		[setattr(self,atr,val) for atr,val in sVar.items()]
		self.xPrior = [x0]
		self.xPost = [x0]
		if ll0 is not None:
			self.medModel = Gps_aXYZ(ll0=ll0)
		else:
			self.medModel = Gps_aXYZ()
		self.model = ModeloCoordenadas(x0,pars=cpars)
		self.L = [np.linalg.inv(Cmed)]
		self.z = []
		self.w_c_T = [] #guardo la transformacion de la cam al mundo

	def _prior(self):
		cvars = ['xPost','Cx','Cu','u']
		xp,Cx,Cu,u = [getattr(self,thisVar)[-1] for thisVar in cvars]
# 		xp = self.xPost[-1]
# 		Cx = self.Cx[-1]
# 		Cu = self.Cu[-1]
# 		xp = self.xPrior[-1]
# 		u  = self.u[-1]
		n = self.nParts

		modelCalls = ['applyModel','getFx','getFu']
		xprior,Fx,Fu = [getattr(self.model,call)(xp,u) for call in modelCalls]
		#xprior 	= self.model.applyModel(xp,u)
		#Fx 		= self.model.getFx(xp,u)
		#Fu 		= self.model.getFu(xp,u)

		su,sx = map(lambda C: np.sqrt(np.diag(C))[:,np.newaxis] , (Cu,Cx))
		
		Fx = Fx.mean(0)
		
		Fx = np.vstack([Fx,Fx[-2:-1,:]])
		Fu = np.vstack([Fu,Fu[-2:-1,:]])
		
		inpNoise,modNoise = map(lambda F,s: F.dot(s*np.random.randn(len(s),n)),
								 (Fu,Fx),(su,sx))

		xprior = xprior + inpNoise.T +modNoise.T
		self.xPrior.append(xprior)


	def _post(self):
		#self.xPost.append(self.xPrior[-1])
		#self.Cx.append(self.Cx[-1])
		cvars = ['xPrior','Cx','med','Cmed','L']
		xp,Cx,z,Cmed,L = [getattr(self,thisVar)[-1] for thisVar in cvars]
		n = self.nParts

		modelCalls = ['getMx','getMed']
		Fm,medPrior = [getattr(self.model,call)(xp) for  call in modelCalls]


		self.z.append(z)

		z = np.concatenate([z,np.zeros((1))])
		z[-1] = np.sin(z[-2])
		z[-2] = np.cos(z[-2])
		innov 	= z[np.newaxis,:] - medPrior
		Ln = np.eye(5,5)
		Ln[:4,:4] = L
		Ln[-1,-1] = L[-1,-1]
		
		normL = np.array([(x.dot(Ln)).dot(x) for x in  innov ])

		w   = np.e**(-normL) +1e-17
		wn = w/w.sum()
		Wcum = np.cumsum(wn)
		r = np.random.rand(n)
		d = np.abs(Wcum[np.newaxis,:]-r[:,np.newaxis])

		partSel = np.argmin(d,axis=1)
		xPost  = xp[partSel,:] 
		self.xPost.append(xPost)

		self.innov.append(innov)

	def setModel(self,model):
		self.model = model
	
	def _postSinMed(self):
		self.xPost.append(self.xPrior[-1])
		self.Cx.append(self.Cx[-1])


	def set_U_Med(self,u,z=None):
		self.u.append(u)

		if z is not None:

			med = self.medModel(self.xPost[-1].mean(-1),z)
			self.med.append(med)

	def runOneIt(self,u,z=None):
		self.set_U_Med(u,z)
		self._prior()
		if z is not None:
			self._post()
		else:
			self._postSinMed()




class KF_theta(object):
	def __init__(self,x0=0,F=1,Cx=1,Cu=1,H=1,Cmed=1):
		sVar = {'xPrior':[x0] ,'xPost':[x0],
				'Fx':F, 'H':H,
				'CxPrior':[Cx],'CxPost':[Cx],
				'u':[],'Cu':[Cu],'med':[],'Cmed':[Cmed],
				'innov':[],'KalmanGain':[]}
		[setattr(self,atr,val) for atr,val in sVar.items()]
	
	def Prior(self,u):
		xp =self.xPost[-1] + u
		cx = self.Fx * self.CxPost[-1]*self.Fx + self.Cu[-1]
		
		self.xPrior.append(xp) 
		self.CxPrior.append(cx) 
	
	def Post(self,med=None):
		if med is not None:
			nu = med -self.xPrior[-1]
			S = self.H * self.CxPrior[-1] * self.H + self.Cmed[-1]
			K = self.CxPrior[-1]/S
			xp = self.xPrior[-1] + K*nu
			cx = (1-K)*self.CxPrior[-1]
			self.innov.append(nu)
			self.KalmanGain.append(K)
			self.xPost.append(xp)
			self.CxPost.append(cx)
		else:
			self.innov.append(self.xPrior[-1])
			self.KalmanGain.append(self.KalmanGain[-1])
			self.xPost.append(self.xPrior[-1])
			self.CxPost.append(self.CxPrior[-1])
	
	def __call__(self,u,med=None):
		self.Prior(u)
		self.Post(med)












class KalmanFilter():
	def __init__(self,model=None,x0=0):
		raise NotImplementedError('Modelo no definido')



# class KalmanFilter():
# 	eye = np.eye(3)
# 	def __init__(self,x0=[0,0,0],dt=.1,Cx=eye,
# 						Cu=eye, Cmed=eye, model=None):

# 		sVar = {'xPrior':[x0] ,'xPost':[x0],
# 				'CxPrior':[Cx],'CxPost':[Cx],
# 				'u':[],'Cu':[Cu],'med':[],'Cmed':[Cmed],
# 				'dt':dt,'innov':[],'KalmanGain':[]}

# 		[setattr(self,atr,val) for atr,val in sVar.items()]
# 		self.model = model if model is not None else ModeloCoordenadas(x0)


# 	def _prior(self):
# 		cvars = ['xPost','CxPost','Cu','u']
# 		xp,Cx,Cu,u = [getattr(self,thisVar)[-1] for thisVar in cvars]

# 		modelCalls = ['applyModel','getFx','getFu']
# 		xprior,Fx,Fu = [getattr(self.model,call)(xp,u) for
# 							  call in modelCalls]

# 		Cx = np.dot(np.dot(Fx,Cx),Fx.T) +\
# 				Cu

# 		self.xPrior.append(xprior)
# 		self.CxPrior.append(Cx)

# 	def _post(self):
# 		xp = self.xPrior[-1]
# 		Cx = self.CxPrior[-1]

# 		z 		= self.med[-1]
# 		Cmed 	= self.Cmed[-1]
# 		Fm 		= self.model.getMx(xp)
# 		innov 	= z - self.model.getMed()

# 		#innov[-1] = (innov[-1]+np.pi)%(2*np.pi)-np.pi
# 		S = np.dot( np.dot(Fm,Cx) ,Fm.T) + Cmed
# 		if len(S)>1:
# 			invS = np.linalg.inv(S)
# 		else:
# 			invS = 1/S

# 		K = np.dot(np.dot(Cx,Fm.T),invS)
# 		self.innov.append(innov)
# 		self.KalmanGain.append(K)
# 		xp += np.dot(K,innov)
# 		kh  = np.dot(K,Fm)
# 		if hasattr(kh,'__len__'):
# 			Cx = np.dot((np.eye(*kh.shape)-kh) , Cx)
# 		else:
# 			Cx = np.dot(1-kh , Cx)

# 		self.xPost.append(xp)
# 		self.CxPost.append(Cx)

# 	def _postSinMed(self):
# 		self. xPost.append(self.xPrior[-1])
# 		self.CxPost.append(self.CxPrior[-1])


# 	def set_U_Med(self,u,z=None):
# 		self.u.append(u)
# 		if z is not None:
# 			self.med.append(z)




# 	def runOneIt(self,u,z=None):
# 		self.set_U_Med(u,z)
# 		self._prior()
# 		if z is not None:
# 			self._post()
# 		else:
# 			self._postSinMed()


