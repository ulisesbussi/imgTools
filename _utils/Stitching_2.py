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
from .kalmanFilter import KalmanFilter
from ._PID import PID
from ._imgTools import (get_s_theta_T_fromAffine,
						get_Affine_From_s_theta_T,
						LLtoMeters)


class Stitching(object):
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
		self.pid = PID(1e-4,0,1e-4)
		self.corDetected 	= list()
		self.qLvlList 		= list()

		self.feature_params = dict( 	maxCorners = 1000,
									qualityLevel = self.qLvl,
									minDistance = 7, blockSize = 7 )

		# Parameters for lucas kanade optical flow
		self.lk_params = dict( 	winSize  = (15,15),maxLevel = 6,
								criteria = (cv2.TERM_CRITERIA_EPS |\
								cv2.TERM_CRITERIA_COUNT, 10, 0.03))



		self.new_dth   = 0
		self.new_trasl = np.array([0,0])



		self.vid = vid

		self.start = start if start is not None else 0
		aux = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
		self.end = end if end is not None else aux

		self.vid.set(cv2.CAP_PROP_POS_FRAMES,self.start)

		self.cam_pars = pars
		self.sub_df = sub
		self.log_df = log

		self.fixaffine = fixaffine
		if fixaffine:
			Warning('Still experimental')
		self._ParseKwargs(kwargs)

	def _ParseKwargs(self,kwargs):
		self.__dict__.update(kwargs)

	def _getFrame(self):
		ret, frame = self.vid.read()
		if not ret:
			return None
		if self.downsample:
			frame = cv2.pyrDown(frame)
		return frame

	def _bboxing(self,affi,esquinas,sTiT):
			# === MAP ESQUINAS TO STITCH TO PAD IF NECESSARY
		esqAff = affi.dot(esquinas)
		# saco el bbx donde caeria warped
# 		bbx = np.int64([np.floor(min(esqAff[0])), np.floor(min(esqAff[1])),
# 						np.ceil(max(esqAff[0])), np.ceil(max(esqAff[1]))])

		bbx = np.int64([np.floor(esqAff.min(1)),
						np.ceil( esqAff.max(1))]).reshape(-1)

		# rellenar con negro para que haya lugar
		hS, wS = sTiT.shape[:2] # tamaño actual

		# chequeo los cuatro lados para ver si hay que agregar filas, columnas
		# actualizo el cero en la tranf affine si corresponde
		if bbx[0] < 0: # rellenar con columnas hacia izquierda
			delta = -bbx[0]
			relleno = np.zeros((hS, delta, 3), dtype=np.uint8)
			sTiT = np.concatenate([relleno, sTiT], axis=1)
			hS, wS = sTiT.shape[:2] # tamaño actual
			affi[0, 2] += delta # actualizo desplazamiento afin
		
		if bbx[1] < 0: # rellenar con filas hacia arriba
			delta = -bbx[1]
			relleno = np.zeros((delta, wS, 3), dtype=np.uint8)
			sTiT = np.concatenate([relleno, sTiT], axis=0)
			hS, wS = sTiT.shape[:2] # tamaño actual
			affi[1, 2] += delta # actualizo desplazamiento afin
	
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
	
		return affi,sTiT


	def _undistort(self,frame):
		if self.cam_pars is None:
			Warning('No hay parametros de camara definidos!')
			return frame
		else:
			frame = cv2.undistort(frame,self.cam_pars['cameraMatrix'],
										self.cam_pars['distCoeffs'])
			return frame


	def _getAffine(self,old_gray,frame_gray,p0):
		# === calculate optical flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0,
										 None, **self.lk_params)
		
		# Select good points
		good_old = p0[st==1].reshape((-1, 1, 2))
		good_new = p1[st==1].reshape((-1, 1, 2))
		nCor = len(p1)
		self.corDetected.append(nCor)
		self.qLvlList.append(self.qLvl)
		self.pid_qLvl() #pid
		retval, inliers = cv2.estimateAffinePartial2D(good_new, good_old)
		return retval


	def _getIndexInLogDf(self,frNum):
		if self.sub_df is not None and self.log_df is not None:

			dateTimeInSub = self.sub_df.query('frameNumber=='+\
									 str(eval('frNum')))['dateTime']
			indexInLog = np.abs(self.log_df['CUSTOM.updateTime']-\
					   dateTimeInSub.values[0]).idxmin()
			self.indexInLog =  indexInLog
		else:
			self.indexInLog =  None


	def stitching(self,undistort=False):
		self.vid.set(cv2.CAP_PROP_POS_FRAMES,self.start)
		self._getIndexInLogDf(self.start)

		self._initKf()

		frame = self._getFrame()
		if frame is None:
			raise ValueError
		if undistort:
			frame = self._undistort(frame)

		frCount = 0
		frDelta = self.end-self.start

		h, w = frame.shape[:2]

		old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
								   **self.feature_params)

		sTiT = frame.copy()  # donde ir haciendo el stitch
		affi = np.eye(3)[:2] # donde ir acumulando las trans afin

		self.affines 	= [affi]
		self.afCor 		= [affi]
		esquinas = np.array([[0, 0, 1] , [0, w, 1],
							 [h, w, 1] , [h, 0, 1] ]).T[[1,0,2]]

		while frCount <= frDelta :
			frame = self._getFrame()
			if frame is None:
				break
			cv2.imshow('frame', frame)

			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			retval = self._getAffine(old_gray,frame_gray,p0)
			if retval is None:
				print('No se encontró la transformación.. saliendo')
				break
			self.affines.append(retval)

			if self.fixaffine:
				self.frCount=frCount
				retval = self._FixAffine(retval)

			# ==== ACCUMULATE AFFINE: GET MAP FROM NEW FRAME TO STITCH
			affi[:,2] += affi[:,:2].dot(retval[:,2])
			affi[:,:2] = affi[:,:2].dot(retval[:,:2])
			affi,sTiT = self._bboxing(affi,esquinas,sTiT)

			self.afCor.append(affi) #affine Corregida

			# === WARP FRAME TO THE STICHED IMAGE
			hS, wS = sTiT.shape[:2]
			sTiT = cv2.warpAffine(frame, affi, (wS, hS), dst=sTiT,
								  flags 			= cv2.INTER_LINEAR,
								  borderMode 	= cv2.BORDER_TRANSPARENT)

			esqAff = affi.dot(esquinas)
			rectangulo = [np.int32(esqAff.T)]
			sTiTshow = cv2.polylines(sTiT.copy(), rectangulo, True, (0,0,255))
			cv2.imshow('stitched', cv2.pyrDown(sTiTshow))

			# === LEAVE VARIABLE READY FOR NEXT LOOP
			old_gray = frame_gray.copy()
			p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
											**self.feature_params)
			
			c = cv2.waitKey(1)
			print('{:d} de {:d} frames procesados'.format(frCount,frDelta) )
			if  c == ord('q'): #Salir
				break
			frCount += 1
		cv2.namedWindow('stitched')
		cv2.destroyWindow('stitched')
		cv2.namedWindow('frame')
		cv2.destroyWindow('frame')
		return [sTiT,np.array(self.affines)]


	def _initKf(self):
		if self.fixaffine:
			x0 		= (0,0,0)
			grado = np.pi/180
			Cx 		= np.array([[.005,0,0],[0,.005,0],[0,0,.5*grado]])
			Cu 		= np.array([[.01 ,0,0],[0, .01,0],[0,0, 1*grado]])
			Cmed 	= np.array([[  5 ,0,0],[0,  5 ,0],[0,0, 3*grado]])
			self._kf = KalmanFilter(x0,Cx=Cx,
									   Cu=Cu,Cmed=Cmed)
			self.nTheta = self.yaw[0]
			self.tester = []
			c0 = np.cos(self.yaw[0])
			s0 = np.sin(self.yaw[0])
			self._rotWorldToImg =np.array([ [ c0,  -s0, 0 ],
											[ -s0, -c0, 0 ],
											[ 0 ,   0 , 1 ]])
		else:
			self._kf = None


	def _FixAffine(self,affine):
		"""This is suposed To be a Kalman Filter aplied in the model"""
		#TODO A Lot
		###KALMANFILTER
		af = np.concatenate([affine,
							np.array([[0,0,1]])],
							axis=0)
		fn = self.frCount
		s,d_theta,T = get_s_theta_T_fromAffine(af)

		_pix2Meters = 1/60  #6/2000 #el 1/60 es a manopla
		t = T*_pix2Meters

		inpt  	 = [*(t[::-1]),d_theta]

		thMed  =   self.yaw[self.frCount]
		dmed   = LLtoMeters(self.lat[0] , self.lon[0],
						 self.lat[fn], self.lon[fn])

		medition = [*dmed,thMed]
		medition = np.dot(self._rotWorldToImg,medition)
		self._kf.runOneIt(inpt,medition)

		###ADITIONALFILTER
		deltaAux 	= 	self._kf.xPost[-1] -\
						self._kf.xPost[-2]

		d_est_traslation 	= 	deltaAux[:-1]
		d_est_theta 			= 	deltaAux[-1]


		self.new_dth = d_theta
		#self.new_dth = d_theta*.75 ++self.new_dth/4
		#self.new_dth   = d_theta/2 +self.new_dth/4 + d_est_theta/4
		if self.frCount==0:
			self.new_dth=thMed

		self.new_trasl = T
		#self.new_trasl =  T/2 + self.new_trasl/4 +\
		#				 (1/4)*(d_est_traslation/_pix2Meters)


		newAff = get_Affine_From_s_theta_T(s,self.new_dth,self.new_trasl)
		if self.frCount==0:
			self.new_dth=0
		return newAff


	def pid_qLvl(self):

		eNcorNew = self.nCorRef - self.corDetected[-1]
		self.pid.updatePID(eNcorNew)
		self.qLvl = self.pid.getCor()
		if self.qLvl < 0 or 1 < self.qLvl:
			self.qLvl = np.random.random()*0.5 + 0.1

		self.feature_params["qualityLevel"] = np.float32(self.qLvl)

