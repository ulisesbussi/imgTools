#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:21:34 2019
Modelos matematicos para los filtros
@author: ulises
"""

import numpy as np
from numpy import cos, sin
from numpy import arctan2 as atan2



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
	del sistema u= [dx,dy,dth] obtenidos de la matriz afin.
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
		super(ModeloCoordenadas,self).__init__(x)
		if pars is None:
			self.pars = 4*[None]
		else:
			self.pars= pars
			self.createCameraMat(pars)


	def createCameraMat(self,pars):
		self.M = np.array([	[pars[0],  0    , 0 , -pars[2]], #aca no estoy seguro de los signos
							[  0    ,pars[1], 0 , -pars[3]],
							[  0    ,  0    , 1 , 0      ],
							[  0    ,  0    , 0 ,  1     ]])
		self.iM = np.linalg.inv(self.M)

	def getH(self,u):
		c0 = cos(u[-1])
		s0 = sin(u[-1])
		H = np.array([	[c0,-s0,0 ,u[0]],
						[s0, c0,0 ,u[1]],
						[ 0, 0 ,1 ,u[2]],
						[ 0, 0 ,0 ,  1  ]])
		return H

	def applyModel(self,X,u):
		Ti  = self._ensamble(X)
		phi = self.getFx(X,u)
		Ti1 = np.dot(phi,Ti)
		newX = self._unsamble(Ti1)
		return newX

	def getFx(self,X=None,u=None):
		H = self.getH(u)
		return np.dot(self.iM,H).dot(self.M)



	def _ensamble(self,X):
		x,y,z,th = X
		ct,st = (np.cos(th),np.sin(th))
		T = np.array([	[ct, -st, 0, x],
						[st,  ct, 0, y],
						[0 ,  0 , 1, z],
						[0 ,  0 , 0, 1] ])
		return T


	def _unsamble(self,T):
		x,y,z = T[:-1,-1]
		ct,st = T[:2,0]
		th = np.arctan2(st,ct)
		X = np.array([x,y,z,th])
		return X



	def getFu(self,X=None,u=None):
		#Los Jacobianos HAy que calcularlos hoy no me da el cerebro

		if X is None:
			X = self.x
		X = X.mean(-1) if X.ndim>1 else X
		x,y,z,th 		= X
		dx,dy,dz,dth 	= u
		fx,fy,cx,cy 	= self.pars

		x0,x2 = (sin(dth) , cos(dth))
		x1,x4 = (  1/fx   ,   1/fy  )
		x3 = x1*x2
		x5 = x2*x4
		u00 = x1
		u11 = x4
		u22 = 1
		u02 = -cx*x0*x1 - cy*x3 - fy*x3*y-x*x0
		u12 =  cx*x5 - cy*x0*x4 + fx*x*x5 - x0*y

		Fu = np.array( [ [u00, 0 , 0 ,u02],
						 [ 0 ,u11, 0 ,u12],
						 [ 0 , 0 ,u22, 0 ],
						 [ 0 , 0 , 0 , 0 ]])
		return Fu


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
	def __call__(self,X=None,oldMed=None,med=None):
		return self.get_xyz(X,oldMed,med)

	def latlonTodeltas(self,med=None,ref=None):
		"""Ellipsoidal Earth projected to a plane
		The FCC prescribes the following formulae for distances 
		not exceeding 475 kilometres (295 mi):[2]"""

		ref = self.ll0 if ref is None else ref
		delta = med[0:2]-ref[0:2]
		deltaLat,deltaLon = delta#esto lo cambieee
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


	def get_xyz(self,X=None,oldMed=None,med=None):
		if oldMed is None or med is None:
			raise TypeError('Es necesario definir la medicion actual y anterior')
		if X is None:
			Warning('estas usando el x interno del modelo')
			X=self.x
		x,y =self.latlonTodeltas(med)
		ox,oy =self.latlonTodeltas(oldMed)
		(dx,dy) = (x-ox,y-oy)
		dz, dth = (med-oldMed)[2:]
		th =med[-1]
		z =med[2]
		(cd,sd),(c,s) =map(lambda r: (cos(r),sin(r)), (dth,th))

		xn  =  cd * X[0] + sd *X[1] - c * dx - s * dy
		yn  = -sd * X[0] + cd *X[1] + s * dx - c * dy
		zn  = X[2] + dz
		thn =  X[3]  + dth
		Xnew = np.array([x,y,z,th])
		return Xnew

	def get_w_c_T(self,X=None,oldMed=None,med=None):
		if X is None:
			Warning('estas usando el x interno del modelo')
			X=self.x
		x,y =self.latlonTodeltas(med)
		ox,oy =self.latlonTodeltas(oldMed)
		(dx,dy) = (x-ox,y-oy)
		dz, dth = (med-oldMed)[2:]
		z = med[2]
		th =med[-1]
		(cd,sd),(c,s) =map(lambda r: (cos(r),sin(r)), (dth,th))
#		(c,s) = (cos(th),sin(th))

		T= np.array([x,y,z,th])
# 		T = np.array([  [cd, -sd, 0, dx],
# 						[sd,  cd, 0, dy],
# 						[0,  0, 1, dz],
# 						[0,  0, 0, 1]] )
		return T #ojo por ahora ignoro la transformacion del drone a la camara



