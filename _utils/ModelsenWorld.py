#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:21:34 2019
Modelos matematicos para los filtros
@author: ulises
"""

import numpy as np

from ._imgTools import get_s_theta_T_fromAffine


class Modelo(object):
	"""Skeleton de unn modelo matematico, debe contener las clases 
	applyModel getFx getFu, getMed y getMx"""
	def __init__(self,x=[0,0,0]):
		self.x = x
		pass
	def __call__(self,u): #en principio no tengo en cuenta dt, despues vemos
		self.x = self.applyModel(self.x,u)
		return self.x

	def applyModel(self,X,u):
		raise NotImplementedError

	def getFx(self,X,u):
		raise NotImplementedError

	def getFu(self,X,u):
		raise NotImplementedError

	def getMed(self,X=None,u=None):
		raise NotImplementedError

	def getMx(self,X=None,u=None):
		raise NotImplementedError





class ModeloCoordenadas(Modelo):
	"""Modelo de la imagen para pasar de x_i a x_{i+1} tomando como entrada 
	del sistema u= H obtenidos de la matriz afin.
	on create: 
		x=[0,0,0] is the initial state of the Model
		pars=None CameraParameters list: [f_x, f_y,c_x,c_i]

		methods:
			applyModel(X_i,u_i): calculates the output X_{i+1}
				X_i = [x,y,th]
				u_i = [dx,dy,dth] 
			getFx(X_i,u_i): returns Jacobian of model on it's states X 
			getFu(X_i,u_i): returns Jacobian of model on it's input  
			getMx(X_i,u_i): returns Jacobian of meassurement 
			getMed(X_i,u_i): returns a meassurement predicted by the model  """

	def __init__(self,x=[0,0,3,1,0],pars=None):
		super(ModeloCoordenadas,self).__init__(x)
		if pars is None:
			self.pars = 4*[None]
		else:
			self.pars= pars
			self.createCameraMat(pars)

		self.cdT = np.array([[1, 0, 0,  8],  #trama que transforma desde el dron a camera
							 [0,-1, 0,  .3],
							 [0, 0, -1,   0],
							 [0, 0, 0,   1]  ])

	def createCameraMat(self,pars):
		self.C = np.array([	[pars[0],  0     , 0  , pars[2]], #aca no estoy seguro de los signos
							[  0    ,pars[1] , 0  , pars[3]],
							[  0    ,  0     , 1  ,   0    ],
							[  0    ,  0     , 0  ,   1    ]  ])
		
		self.iC = np.linalg.inv(self.C)


	def zeta(self,Z):
		if hasattr(Z,'ndim'):
			if Z.ndim>=1:
				_zeta = np.array([self.zeta(z) for z in Z])
			else:
				_zeta = self.zeta(Z.item())
		else:
			_zeta =np.array([[1/Z,  0   ,  0  ,  0],
							[0   , 1/Z ,  0  ,  0],
							[0   ,  0  , 1/Z ,  0],
							[0   ,  0  ,  0  ,  1]])
		return _zeta
	
	
	def izeta(self,z):
		return self.zeta(1/z)

	def applyModel(self,X,u):

		Ti 	= np.linalg.inv ( self._ensambleT(X))
		phi = self.getFx(X,u)
		if Ti.ndim>2:
			Ti1 = np.array([np.linalg.inv (  np.dot(p,t)) for t,p in zip(Ti,phi)])
		else:
			Ti1 = np.linalg.inv (  np.dot(phi,Ti))
		newX = self._unsambleT(Ti1)

		return newX


	def getFx(self,X=None,u=None):
		"""Este no es fielmente el jacobiano, ya que zeta depende de los estados
		es una aproximación considerarlo constante para evitar mas quilombo
		Ojo que es lo que uso para propagar modelo tambien, si lo corrijo hay que coregir eso"""
		T = self._ensambleT(X)

		h = self._ensambleH(u)
		if T.ndim>2:
			zeta = self.zeta(T[:,2,-1])
			cZetaT = np.array([self.C.dot(z).dot(self.cdT) for z in zeta])
			icZetaT = np.linalg.inv(cZetaT)
			jac = np.array([iczt.dot(h).dot(czt) for iczt,czt in zip(icZetaT,cZetaT)])
		else:
			zeta = self.zeta(T[2,-1])
			cZetaT = self.C.dot(zeta).dot(self.cdT)
			icZetaT = np.linalg.inv(cZetaT)
			jac = icZetaT.dot(h).dot(cZetaT)
		return jac


	def getFu(self,X=None,u=None):
		#TODO
		"""No estoy seguro que esto este bien del todo Checkear"""
		T = self._ensambleT(X)
		#h = self._ensambleH(u) """not Used"""

		zeta = self.zeta(T[2,-1].mean())
		cZetaT = self.C.dot(zeta.dot(self.cdT))
		icZetaT = np.linalg.inv(cZetaT)

		return 2*icZetaT


	def _ensambleT(self,X):
		if X.ndim>1:
			T=np.array([self._ensambleT(x) for x in X])
		else:
			x,y,z,ct,st = X
			#ct,st = (np.cos(th),np.sin(th))
			T = np.array([	[ct, -st, 0, x],
							[st,  ct, 0, y],
							[0 ,  0 , 1, z],
							[0 ,  0 , 0, 1] ])
		return T


	def _unsambleT(self,T):
		if T.ndim>2:
			X=np.array([self._unsambleT(t) for t in T])
		else:
			x,y,z = T[:-1,-1]
			ct,st = T[:2,0]
			X = np.array([x,y,z,ct,st]).T
		return X





	def _ensambleH(self,U):
		if U.ndim>1:
			H = np.array([self._ensambleH(u) for u in U])
		else:		
			 dx,dy,dth = U
			 ct,st = (np.cos(dth),np.sin(dth))
			 H = np.array([	[ct, -st, 0, dx],
							[st,  ct, 0, dy],
							[0 ,  0 , 1, 0],
							[0 ,  0 , 0, 1] ])
		return H



	def _unsambleH(self,H):
		if H.ndim>2:
			u = np.array([self.unsambleH(h)for h in H])
		else :
			_,th,t = get_s_theta_T_fromAffine(H)
			u = np.hstack([t[:-1],th])
		return u


	def getMed(self,X=None,u=None):
		if X is None:
			X=self.x
		return X

	def getMx(self,X=None,u=None):
		return np.eye(3)

class ModeloCoordenadas_old(Modelo):
	"""Modelo de la imagen para pasar de x_i a x_{i+1} tomando como entrada 
	del sistema u= H obtenidos de la matriz afin.
	on create: 
		x=[0,0,0] is the initial state of the Model
		pars=None CameraParameters list: [f_x, f_y,c_x,c_i]

		methods:
			applyModel(X_i,u_i): calculates the output X_{i+1}
				X_i = [x,y,th]
				u_i = [dx,dy,dth] 
			getFx(X_i,u_i): returns Jacobian of model on it's states X 
			getFu(X_i,u_i): returns Jacobian of model on it's input  
			getMx(X_i,u_i): returns Jacobian of meassurement 
			getMed(X_i,u_i): returns a meassurement predicted by the model  """

	def __init__(self,x=[0,0,3,0],pars=None):
		super(ModeloCoordenadas_old,self).__init__(x)
		if pars is None:
			self.pars = 4*[None]
		else:
			self.pars= pars
			self.createCameraMat(pars)

		self.cdT = np.array([[1, 0, 0,   0],  #trama que transforma desde el dron a camera
							 [0,-1, 0, -.3],
							 [0, 0,-1,   0],
							 [0, 0, 0,   1]  ])

	def createCameraMat(self,pars):
		self.C = np.array([	[pars[0],  0     , 0  , pars[2]], #aca no estoy seguro de los signos
							[  0    ,pars[1] , 0  , pars[3]],
							[  0    ,  0     , 1  ,   0    ],
							[  0    ,  0     , 0  ,   1    ]  ])
		
		self.iC = np.linalg.inv(self.C)


	def zeta(self,Z):
		if hasattr(Z,'ndim'):
			if Z.ndim>=1:
				_zeta = np.array([self.zeta(z) for z in Z])
			else:
				_zeta = self.zeta(Z.item())
		else:
			_zeta =np.array([[1/Z,  0   ,  0  ,  0],
							[0   , 1/Z ,  0  ,  0],
							[0   ,  0  , 1/Z ,  0],
							[0   ,  0  ,  0  ,  1]])
		return _zeta
	
	
	def izeta(self,z):
		return self.zeta(1/z)


# 	def getH(self,u):
# 		c0 = cos(u[-1])
# 		s0 = sin(u[-1])
# 		H = np.array([[c0,-s0,0 ,u[0]],
# 						[s0, c0,0 ,u[1]],
# 						[ 0, 0 ,1 ,u[2]],
# 						[ 0, 0 ,0 ,  1  ]])
# 		return H

	def applyModel(self,X,u):

		Ti 	= np.linalg.inv ( self._ensambleT(X))
		phi = self.getFx(X,u)
		if Ti.ndim>2:
			Ti1 = np.array([np.linalg.inv (  np.dot(p,t)) for t,p in zip(Ti,phi)])
		else:
			Ti1 = np.linalg.inv (  np.dot(phi,Ti))
		newX = self._unsambleT(Ti1)

		return newX

	def getFx(self,X=None,u=None):
		"""Este no es fielmente el jacobiano, ya que zeta depende de los estados
		es una aproximación considerarlo constante para evitar mas quilombo
		Ojo que es lo que uso para propagar modelo tambien, si lo corrijo hay que coregir eso"""
		T = self._ensambleT(X)

		h = self._ensambleH(u)
		if T.ndim>2:
			zeta = self.zeta(T[:,2,-1])
			cZetaT = np.array([self.C.dot(z).dot(self.cdT) for z in zeta])
			icZetaT = np.linalg.inv(cZetaT)
			jac = np.array([iczt.dot(h).dot(czt) for iczt,czt in zip(icZetaT,cZetaT)])
		else:
			zeta = self.zeta(T[2,-1])
			cZetaT = self.C.dot(zeta).dot(self.cdT)
			icZetaT = np.linalg.inv(cZetaT)
			jac = icZetaT.dot(h).dot(cZetaT)
		return jac


	def getFu(self,X=None,u=None):
		#TODO
		"""No estoy seguro que esto este bien del todo Checkear"""
		T = self._ensambleT(X)
		#h = self._ensambleH(u) """not Used"""

		zeta = self.zeta(T[2,-1].mean())
		cZetaT = self.C.dot(zeta.dot(self.cdT))
		icZetaT = np.linalg.inv(cZetaT)

		return 2*icZetaT


	def _ensambleT(self,X):
		if X.ndim>1:
			T=np.array([self._ensambleT(x) for x in X])
		else:
			x,y,z,th = X
			ct,st = (np.cos(th),np.sin(th))
			T = np.array([	[ct, -st, 0, x],
							[st,  ct, 0, y],
							[0 ,  0 , 1, z],
							[0 ,  0 , 0, 1] ])
		return T


	def _unsambleT(self,T):
		if T.ndim>2:
			X=np.array([self._unsambleT(t) for t in T])
		else:
			x,y,z = T[:-1,-1]
			ct,st = T[:2,0]
			th = np.arctan2(st,ct)
			X = np.array([x,y,z,th]).T
		return X





	def _ensambleH(self,U):
		if U.ndim>1:
			H = np.array([self._ensambleH(u) for u in U])
		else:		
			 dx,dy,dth = U
			 ct,st = (np.cos(dth),np.sin(dth))
			 H = np.array([	[ct, -st, 0, dx],
							[st,  ct, 0, dy],
							[0 ,  0 , 1, 0],
							[0 ,  0 , 0, 1] ])
		return H



	def _unsambleH(self,H):
		if H.ndim>2:
			u = np.array([self.unsambleH(h)for h in H])
		else :
			_,th,t = get_s_theta_T_fromAffine(H)
			u = np.hstack([t[:-1],th])
		return u


	def getMed(self,X=None,u=None):
		if X is None:
			X=self.x
		return X

	def getMx(self,X=None,u=None):
		return np.eye(3)










class Gps_aXYZ(Modelo):
	"""Mapping meassurement med =[lon,lat,z,th] to [x,y,z] coordinates"""
	def __init__(self,x=[0,0,0,0],ll0=[-34,-58]):
		super(Gps_aXYZ,self).__init__(x)
		self.ll0 = ll0
	def __call__(self,X=None,med=None):
		return self.get_med_meters(X,med)

	def latlonTodeltas(self,med=None,ref=None):
		"""Ellipsoidal Earth projected to a plane
		The FCC prescribes the following formulae for distances 
		not exceeding 475 kilometres (295 mi):[2]"""

		ref = self.ll0 if ref is None else ref
		delta = med[0:2]-ref[0:2]
		deltaLon,deltaLat = delta#esto lo cambieee
		mlat = (med[0:2]+ref[0:2])[1]/2
	
		mlaR = np.deg2rad(mlat)
		k1 = 	111.13209- 0.56605*np.cos(2*mlaR)+\
						 0.00012*np.cos(4*mlaR)
		k2 = 	111.41513*np.cos(mlaR)-\
					0.09455*np.cos(3*mlaR)+\
					0.00012*np.cos(5*mlaR)
		dx =  k2 *deltaLon  *1000 #from km to m
		dy =  k1 *deltaLat  *1000
		return dx,dy


	def get_med_meters(self,X=None,med=None,ref=None):
		if X is None:
			Warning('estas usando el x interno del modelo')
			X=self.x
		x, y  = self.latlonTodeltas(med,ref=ref)
		z, th = med[2:]

		med= np.array([x,y,z,th])

		return med



