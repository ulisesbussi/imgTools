#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:57:32 2019

testear la idea de stitching. see it running in runningStitcher.mp4

brief description
===
open video stream. and define initial values for parameters.

read first frame and find it's corners

start loop.

read new frame and calculate the corners of the old freame in this new one.

apply PID control to the quality level of the corner detector. this is
important to keep detecting corners despite the change of image structure.

calculate the affine transformation that maps from new frame to previous frame.

calculate affine that maps from new frame to the stitched image (accumulated
affine transform)

transform the 4 points that describe the image rectangle (esquinas) to the
stitched image. check if they are out of bounds and pad appropiately.

paste the new frame into stitched image.

define the new frame as the ond frame and calculate its corners for next loop.

end of loop


things to improve
===
control de parameters for the corners detectors and optical flow so that it's
robust and doesn't run out of corners. in other words, tune the PID or make
something better

@author: sebalander
@email: sebastian.arroyo@unq.edu.ar
"""
"""
Created on Mon Sep  9 17:07:52 2019
STITCHING From seba Code
@author: ulises
"""

import cv2
import numpy as np
#from .kalmanFilter import KalmanFilter
from ._PID import PID
from ._imgTools import get_s_theta_T_fromAffine, get_Affine_From_s_theta_T


class StitchingGround(object):
	"""This object perform the stitching of consecutive Image in video
	using feature matching  it takes at least 1 input argument: 
		vid : a cv2 videoCapture Object. 
		start=None <- frame number where to start the stitch 
						if None. then it takes 0 as start.
		end=None <- frame number where to stop the stitch 
						if None. then it takes the last frame of the video.
		pars=None <- A dictionary that contains at least the keys 'cameraMatrix' 
						and 'distCoeffs'. If None, Stitching._undistort(frame) 
						will raise a warning and return the original frame."""
		#TODO Infinitas cosas, documentar todas las funciones, implementar las faltantes
		#TODO comprar un conejo, ver si aprende a pilotear el drone.
	def __init__(self, vid, start=None, end=None, pars=None, \
					sub=None, log=None, fixaffine=False, **kwargs):
		#super(Stitching,self).__init__()

		self.downsample = True # i worked with the imges at half size for simplicity
		# params for ShiTomasi corner detection
		self.qLvl = 0.7 # cond inicial de umbral de calidad
		self.nCorRef 	= 100 # desired number of corners
		self.pid = PID(3e-4,0,1e-4)
		self.corDetected 	= list()
		self.qLvlList 		= list()

		self.ndth =0
		self.T = np.array([0,0])

		self.feature_params = dict( maxCorners = 1000, qualityLevel = self.qLvl,
								minDistance = 7, blockSize = 7 )
		
		# Parameters for lucas kanade optical flow
		self.lk_params = dict( winSize  = (15,15),maxLevel = 6,
								criteria = (cv2.TERM_CRITERIA_EPS |\
										cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		self.vid = vid

		self.start = start if start is not None else 0
		aux = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
		self.end = end if end is not None else aux

		self.vid.set(cv2.CAP_PROP_POS_FRAMES,self.start)

		self.cam_pars = pars
		self.sub_df = sub
		self.log_df = log
		self.tester =[]
		self.oth = 0
		self.theta = 0
		self.esquinasyShift = []


		self.fixaffine = fixaffine
		if fixaffine:
			Warning('Not Working Yet! still experimental')
		self._ParseKwargs(kwargs)
		
		self.detector = cv2.xfeatures2d.SURF_create()
		#self.detector = cv2.ORB_create()
		
		FLANN_INDEX_KDTREE = 1
		index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		self.matcher  = cv2.FlannBasedMatcher(index_params, search_params)
		


	def _ParseKwargs(self,kwargs):
		self.__dict__.update(kwargs)


	def _getFrame(self,undistort,step = 1):
		for i in range(step):
			ret, frame = self.vid.read()
		
		if not ret:
			return None
		if frame is None:
			raise ValueError
		if undistort:
			frame = self._undistort(frame)
		if self.downsample:#TODO aca el primero undistort después pyrdown
			frame = cv2.pyrDown(frame)
		return frame


	def _bboxing(self,affi,esquinas,sTiT):
			# === MAP ESQUINAS TO STITCH TO PAD IF NECESSARY
		esqAff = affi.dot(esquinas)
		# saco el bbx donde caeria warped
		bbx = np.int64([np.floor(min(esqAff[0])), np.floor(min(esqAff[1])),
						np.ceil(max(esqAff[0])), np.ceil(max(esqAff[1]))])
		# rellenar con negro para que haya lugar
		hS, wS = sTiT.shape[:2] # tamaño actual
		# chequeo los cuatro lados para ver si hay que agregar filas, columnas
		# actualizo el cero en la tranf affine si corresponde
		delta1 = 0 #esto es si estoy corriendo el 0 en x
		delta2 = 0 #esto es si estoy corriendo el 0 en y
		if bbx[0] < 0: # rellenar con columnas hacia izquierda
			delta1 = -bbx[0]
			relleno = np.zeros((hS, delta1, 3), dtype=np.uint8)
			sTiT = np.concatenate([relleno, sTiT], axis=1)
			hS, wS = sTiT.shape[:2] # tamaño actual
			affi[0, 2] += delta1 # actualizo desplazamiento afin
		
		if bbx[1] < 0: # rellenar con filas hacia arriba
			delta2 = -bbx[1]
			relleno = np.zeros((delta2, wS, 3), dtype=np.uint8)
			sTiT = np.concatenate([relleno, sTiT], axis=0)
			hS, wS = sTiT.shape[:2] # tamaño actual
			affi[1, 2] += delta2 # actualizo desplazamiento afin
	
		if wS < bbx[2]: # rellenar con columnas hacia derecha
			delta = bbx[2] - wS + 1
			relleno = np.zeros((hS, delta, 3), dtype=np.uint8)
			sTiT = np.concatenate([sTiT, relleno], axis=1)
			hS, wS = sTiT.shape[:2] # tamaño actual
	
		if hS < bbx[3]: # rellenar con filas hacia abajo
			delta = bbx[3] - hS + 1
			relleno = np.zeros((delta, wS, 3), dtype=np.uint8)
			sTiT = np.concatenate([sTiT, relleno], axis=0)
			hS, wS = sTiT.shape[:2] # tamaño actual
	
		self.esquinasyShift.append([esqAff,[delta1,delta2]])
		return affi,sTiT


	def _undistort(self,frame):

		if self.cam_pars is None:
			Warning('No hay parametros de camara definidos!')
			return frame
		else:
			frame = cv2.undistort(frame,self.cam_pars['cameraMatrix'],
										self.cam_pars['distCoeffs'])
			return frame

	def calcMask(self,frame,thresh=10):#20):
		gi = 2.0*frame[:,:,1]-frame[:,:,0]-frame[:,:,-1]
		return np.uint8(255*(gi<thresh))
		
	def _getAffine(self,old_frame,frame):
		# === calculate optical flow
		mask_old,mask_new = [self.calcMask(f) for f in [old_frame,frame]]
		kp1,des1 = self.detector.detectAndCompute(old_frame,mask_old)
		kp2,des2 = self.detector.detectAndCompute(frame,mask_new)

		matches = self.matcher.knnMatch(des1,des2,k=2)
		good = []
	
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append(m)
		
		if len(good)>5:
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			#affi, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			retval, inliers = cv2.estimateAffinePartial2D(dst_pts,src_pts)

		else:
			raise('Error pocos puntos')

		return retval#affi


	def _getIndexInLogDf(self,frNum):
		#frNum = start + frCount
		if self.sub_df is not None and self.log_df is not None:
			dateTimeInSub = self.sub_df.query('frameNumber=='+\
									 str(eval('frNum')))['dateTime']
			indexInLog = np.abs(self.log_df['CUSTOM.updateTime']-\
					   dateTimeInSub.values[0]).idxmin()
			self.indexInLog =  indexInLog
		else:
			self.indexInLog =  None


	def stitching(self,undistort=False):
		stp=30
		self.vid.set(cv2.CAP_PROP_POS_FRAMES,self.start)
		self._getIndexInLogDf(self.start)

		#self._initKf()

		frame = self._getFrame(undistort)
# 		frame = self._getFrame()
# 		if frame is None:
# 			raise ValueError
# 		if undistort:
# 			frame = self._undistort(frame)

		frCount = 0
		frDelta = self.end-self.start

		h, w = frame.shape[:2]
		old_frame = frame
#		old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#		p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **self.feature_params)

		sTiT = frame.copy()  # donde ir haciendo el stitch
		affi = np.eye(3)[:2] # donde ir acumulando las trans afin

		self.affines 	= [affi]
		self.afCor 		= [affi]
		esquinas = np.array([[0, 0, 1] , [0, w, 1],
							 [h, w, 1] , [h, 0, 1] ]).T[[1,0,2]]

		while frCount <= frDelta :
			frame = self._getFrame(undistort,stp)

#			cv2.imshow('frame', frame)

			#frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			retval =self._getAffine(old_frame,frame)
			if retval is None:
				print('No se encontró la transformación.. saliendo')
				break
			self.affines.append(retval)

			if self.fixaffine:
				self.frCount=frCount
				#self._getIndexInLogDf(self.start+frCount)
				retval = self._FixAffine(retval)

			# ==== ACCUMULATE AFFINE: GET MAP FROM NEW FRAME TO STITCH
			affi[:,2] += affi[:,:2].dot(retval[:,2])
			affi[:,:2] = affi[:,:2].dot(retval[:,:2])
			affi,sTiT = self._bboxing(affi,esquinas,sTiT)

			self.afCor.append(affi) #affine Corregida?

			# === WARP FRAME TO THE STICHED IMAGE
			hS, wS = sTiT.shape[:2]
			sTiT = cv2.warpAffine(frame, affi, (wS, hS), dst=sTiT,
								  flags 			= cv2.INTER_LINEAR,
								  borderMode 	= cv2.BORDER_TRANSPARENT)

			esqAff = affi.dot(esquinas)
			rectangulo = [np.int32(esqAff.T)]
			sTiTshow = cv2.polylines(sTiT.copy(), rectangulo, True, (0,0,255))
			cv2.imshow('stitched', cv2.pyrDown(cv2.pyrDown(sTiTshow)))

			# === LEAVE VARIABLE READY FOR NEXT LOOP
			old_frame = frame.copy()
			#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **self.feature_params)
			
			c = cv2.waitKey(1)
			print('{:d} de {:d} frames procesados'.format(frCount,frDelta) )
			if  c == ord('q'): #Salir
				break
			frCount += stp
		cv2.namedWindow('stitched')
		cv2.destroyWindow('stitched')
		cv2.namedWindow('frame')
		cv2.destroyWindow('frame')
		return [sTiT,np.array(self.affines)]


	def _initKf(self):
		Warning('ESTOY ENTRANDO EN _initKf, no debería')
		pass
# 		#if self.sub_df is not None and self.log_df is not None:
# 		#idx = self.indexInLog
# 		if self.fixaffine:
# 			th0 = self.yaw[0]
# 			self._kf = KalmanFilter(th0,0.04,.04,.01)
# 			self.nTheta = th0
# 			self.tester = []
# 		else:
# 			self._kf = None


	def _FixAffine(self,affine):
		#TODO Not WorkingYet

		af = np.concatenate([affine,
							np.array([0,0,1]).reshape(1,3)],axis=0)

		#dthMed = self.dyaw[self.frCount]
		s,d_theta,T = get_s_theta_T_fromAffine(af)

		 
		dth = 0.5*(d_theta +self.oth)
		self.oth = d_theta
		#self.theta = dth
		s = (s+1)/2
		#self.tester.append([dthMed,d_theta,dth])

		newAff = get_Affine_From_s_theta_T(s,dth,T)

		return newAff


	def pid_qLvl(self):

		eNcorNew = self.nCorRef - self.corDetected[-1]
		self.pid.updatePID(eNcorNew)
		self.qLvl = self.pid.getCor()
		if self.qLvl < 0 or 1 < self.qLvl:
			self.qLvl = np.random.random()*0.5 + 0.1

		self.feature_params["qualityLevel"] = np.float32(self.qLvl)

