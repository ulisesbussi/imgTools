#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:20:19 2019

@author: ulises
"""


import numpy as np
from numpy import cos,sin
from numpy import arctan2 as atan2

# class ModeloTh(object):
#  	def __init__(self):
# 		pass

#  	def __call__(self,x,u,dt):
# 		return self.applyModel(x,u,dt)

#  	def applyModel(self,x,u,dt):
# 		xn = x +u #+ w*dt
# 		return xn

#  	def getFx(self,x,u,dt):
# 		return np.array([1])

#  	def getFu(self,x,u,dt):
# 		return np.array([dt])

#  	def getMed(self,x,dt):
# 		#medicion
# 		return x

#  	def getMx(self,x,dt):
# 		#jacobiano de Med respecto x
# 		return np.array([1])

#  	def getR(self):
# 		#ruido de medicion
# 		return np.array([2])


# class ModeloEnteroVcte(object):
#  	"""Modelo matematico a velocidad constante, suponiendo altura fija,
#  	desplazamiento en el plano libre, el estado sera interno del modelo"""
#  	def __init__(self,x=None):
# 		self.x = x

# 		pass

#  	def __call__(self,u): #en principio no tengo en cuenta dt, despues vemos
# 		self.x = self.applyModel(self.x,u)
# 		return(self.x)

#  	def applyModel(self,x,u):
# 		dx_TramaMundo = self._rot(x[-1]).dot(u)
# 		xn = x + dx_TramaMundo
# 		return xn

#  	def _rot(self,th):
# 		return np.array([[np.cos(th),-np.sin(th),0],
#  						 [np.sin(th), np.cos(th),0],
#  						 [	0 		, 	0 		,1]])

#  	def getFx(self):
# 		return np.eye(3)

#  	def getFu(self):
# 		return self._rot(self.x[-1][-1])

#  	def getMed(self,x):
# 		#medicion
# 		return self.x[-1]

#  	def getMx(self):
# 		#jacobiano de Med respecto x
# 		return np.eye(3)

#  	def getR(self):
# 		#ruido de medicion
# 		return np.array([3,3,.1])





class ModeloConSeba(object):
	"""Modelo matematico a velocidad constante, suponiendo altura fija,
	desplazamiento en el plano libre, el estado sera interno del modelo"""
	def __init__(self,x=[0,0,0]):
		self.x = x
		pass

	def __call__(self,u): #en principio no tengo en cuenta dt, despues vemos
		self.x = self.applyModel(self.x,u)
		return self.x

	def applyModel(self,X,u):
		x,y,th 		= X
		dx,dy,dth 	= u

		x0 =  cos(dth)
		x1 =  sin(dth)
		x2 =  sin(th)
		x3 =  cos(th)
		out=[]
		out.append( dx + x*x0 + x1*y )
		out.append( dy - x*x1 + x0*y )
		out.append( atan2(x0*x2 - x1*x3, x0*x3 + x1*x2) )

		return np.array(out)



	def getFx(self,X,u):
		if np.ndim(X)>1:
			X= X.mean(1)
		if np.ndim(u)>1:
			u= u.mean(1)
		x,y,th 		= X
		dx,dy,dth 	= u
		x0 =  sin(dth)
		x1 =  sin(th)
		x2 =  cos(dth)
		x3 =  cos(th)
		x4 =  (x0*x1 + x2*x3)**2
		x5 =  x1*x2
		x6 =  x0*x3
		x7 =  1/(x4 + (x5 - x6)**2)
		out=[]
		out.append( 0 )
		out.append( 0 )
		out.append( x4*x7 + x7*(-x5 + x6)**2 )
		#return np.eye(3)
		return np.array([[x2, -x0, 0],
 						 [-x0, x2, 0],
 						 out])





	def getFu(self,X,u):
		if np.ndim(X)>1:
			X= X.mean(1)
		if np.ndim(u)>1:
			u= u.mean(1)
		x,y,th 		= X
		dx,dy,dth 	= u
		x0 =  cos(dth)
		x1 =  sin(dth)
		x2 =  sin(th)
		x3 =  x1*x2
		x4 =  cos(th)
		x5 =  x0*x4
		x6 =  x3 + x5
		x7 =  x0*x2
		x8 =  x1*x4
		x9 =  x7 - x8
		x10 =  1/(x6**2 + x9**2)
		out=[]
		out.append( -x*x1 + x0*y )
		out.append( -x*x0 - x1*y )
		out.append( x10*x6*(-x3 - x5) + x10*x9*(-x7 + x8) )
		returnVal = np.array([[1, 0, 0],	[0, 1, 0],out])

		return returnVal

	def getMed(self,X=None,u=None):
		if X is None:
			return self.x[-1]
		else:
			return X

	def getMx(self,X=None,u=None):
		#jacobiano de Med respecto x
		return np.eye(3)




class ParticleFilter(object):
	eye = np.eye(3)
	def __init__(self,x0=[0,0,0],dt=.1,Cx=eye,
						Cu=eye, Cmed=eye, model=None):
		self.nParts = 300
		x0 = np.array(self.nParts*[x0]).T


		sVar = {'xPrior':[x0] ,'xPost':[x0],
				'Cx':[Cx],'u':[],'Cu':[Cu],
				'med':[],'Cmed':[Cmed],'dt':dt,'innov':[]}

		[setattr(self,atr,val) for atr,val in sVar.items()]

		self.model = model if model is not None else ModeloConSeba(x0)
		self.L = [np.linalg.inv(Cmed)]


	def _prior(self):
		cvars = ['xPost','Cx','Cu','u']
		xp,Cx,Cu,u = [getattr(self,thisVar)[-1] for thisVar in cvars]
		n = self.nParts

		modelCalls = ['applyModel','getFx','getFu']
		xprior,Fx,Fu = [getattr(self.model,call)(xp,u) for
							  call in modelCalls]
		su = np.sqrt(np.diag(Cu))[:,np.newaxis]
		sx = np.sqrt(np.diag(Cx))[:,np.newaxis]
		inpNoise = Fu.dot(su*np.random.randn(3,n))
		modNoise = Fx.dot(sx*np.random.randn(3,n))
		xprior = xprior + inpNoise +modNoise

		self.xPrior.append(xprior)


	def _post(self):
		cvars = ['xPrior','Cx','med','Cmed','L']
		xp,Cx,z,Cmed,L = [getattr(self,thisVar)[-1] for thisVar in cvars]
		n = self.nParts

		modelCalls = ['getMx','getMed']
		Fm,medPrior = [getattr(self.model,call)(xp) for  call in modelCalls]

		innov 	= z[:,np.newaxis] - medPrior

	#	w = np.exp(- innov.T.dot(L).dot(innov).sum(1)) +1e-15
		w   = np.exp(-(innov.reshape(n,3,1)*L.reshape(1,3,3)*\
						innov.reshape(n,1,3)).sum(axis=(1,2))) +1e-17

		wn = w/w.sum()
		Wcum = np.cumsum(wn)
		r = np.random.rand(n)
		d = np.abs(Wcum[np.newaxis,:]-r[:,np.newaxis])

		partSel = np.argmin(d,axis=1)
		xPost  = xp[:,partSel] + 0.1*(Cx[:,np.newaxis,:]*np.random.randn(3,n,1)).sum(-1)
		#xPost = xp
		Cx = np.cov(xPost)

		self.xPost.append(xPost)
		#self.Cx.append(Cx)
		self.innov.append(innov)

	def _postSinMed(self):
		self. xPost.append(self.xPrior[-1])
		self.Cx.append(self.Cx[-1])


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




class KalmanFilter(object):
	eye = np.eye(3)
	def __init__(self,x0=[0,0,0],dt=.1,Cx=eye,
						Cu=eye, Cmed=eye, model=None):

		sVar = {'xPrior':[x0] ,'xPost':[x0],
				'CxPrior':[Cx],'CxPost':[Cx],
				'u':[],'Cu':[Cu],'med':[],'Cmed':[Cmed],
				'dt':dt,'innov':[],'KalmanGain':[]}

		[setattr(self,atr,val) for atr,val in sVar.items()]

		self.model = model if model is not None else ModeloConSeba(x0)


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





"""oldVErsion without comprehensions"""


# class KalmanFilter(object):
# 	def __init__(self,x0=[0,0,0],dt=.1,Cx=np.eye(3),
# 				  Cu=np.eye(3),Cmed=np.eye(3), model=None):

# 		self.xPrior 		= [x0]
# 		self.xPost 		= [x0]
# 		self.CxPrior 	= [Cx]
# 		self.CxPost  	= [Cx]
# 		self.med 		= []
# 		self.u 			= []
# 		self.Cu 			= [Cu]
# 		self.Cmed 		= [Cmed]
# 		self.dt 			= dt


# 		if model is not None:
# 			self.model = model
# 		else:
# 			self.model = ModeloConSeba(x0)

# 		self.innov 		= []
# 		self.KalmanGain = []


# 	def _prior(self):
# 		xp = self.xPost[-1]
# 		Cx = self.CxPost[-1]
# 		Cu = self.Cu[-1]
# 		u  = self.u[-1]
# 		#dt = self.dt

# 		xprior = self.model.applyModel(xp,u)
# 		Fx = self.model.getFx(xp,u)
# 		Fu = self.model.getFu(xp,u)

# 		Cx = np.dot(np.dot(Fx,Cx),Fx.T) +\
# 				Cu

# 		self.xPrior.append(xprior)
# 		self.CxPrior.append(Cx)

# 	def _post(self):
# 		xp = self.xPrior[-1]
# 		Cx = self.CxPrior[-1]

# 		z 		= self.med[-1]
# 		Cmed 	= self.Cmed[-1]
# 		Fm 		= self.model.getMx()
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
