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
		z= X[2]
		X = X / z
		T = self.getFx(X,u)

		newX = np.dot(T, X)
		newX*=z
		return newX

	def getFx(self,X=None,u=None):
		H = self.getH(u)
		return np.dot(self.iM,H).dot(self.M)


	def getFu(self,X=None,u=None):
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
		(cd,sd),(c,s) =map(lambda r: (cos(r),sin(r)), (dth,th))

		xn  =  cd * X[0] + sd *X[1] - c * dx - s * dy
		yn  = -sd * X[0] + cd *X[1] + s * dx - c * dy
		zn  = X[2] + dz
		thn =  X[3]  + dth
		Xnew = np.array([xn,yn,zn,thn])
		return Xnew






class Gps_aXYZ2(Modelo):
	"""Mapping meassurement med =[lon,lat,z,th] to [x,y,z] coordinates"""
	def __init__(self,x=[0,0,0]):
		super(Gps_aXYZ,self).__init__(x)

	def __call__(self,X=None,oldMed=None,med=None):
		return self.get_xyz(X,oldMed,med)

	def latlonTodeltas(self,oldMed,med):
		"""Ellipsoidal Earth projected to a plane
		The FCC prescribes the following formulae for distances 
		not exceeding 475 kilometres (295 mi):[2]"""
		delta = med-oldMed
		deltaLon,deltaLat = delta[0:2]
		mlat = (med+oldMed)[1]/2
	
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
		dx,dy =self.latlonTodeltas(oldMed,med)
		dz,dth = (med-oldMed)[2:]
		th =med[-1]
		(cd,sd),(c,s) =map(lambda r: (cos(r),sin(r)), (dth,th))

		xn =  cd * X[0] + sd *X[1] + c * dx - s * dy
		yn = -sd * X[0] + cd *X[1] + s * dx - c * dy
		zn =  X[2]  + dz
		Xnew = np.array([xn,yn,zn])
		return Xnew





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
			X = self.x[-1]#return self.x[-1]

		c0 = np.cos(X[-1].mean(-1))
		s0 = np.sin(X[-1].mean(-1))
		T_gps_I = np.array([[  c0, -s0, 0 ],
							[ -s0, -c0, 0 ],
							[  0 ,  0 , 1 ] ],dtype=np.float32)
		return T_gps_I.dot(X)

	def getMx(self,X=None,u=None):
		#jacobiano de Med respecto x
		if X is None:
			X = self.x[-1]#return self.x[-1]

		c0 = np.cos(X[-1].mean(-1))
		s0 = np.sin(X[-1].mean(-1))
		T_gps_I = np.array([[  c0, -s0, 0 ],
							[ s0, c0, 0 ],
							[  0 ,  0 , 1 ] ])
		return T_gps_I
