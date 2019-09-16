#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:17:29 2019

@author: ulises
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time


# path 	= '/home/ulises/00_Doctorado/FAUBA_190904/003/'
# name = 'DJI_0191'
# extVid 	= '.MP4'
# extSRT 	= '.SRT'

class Visualizer(object):
	def __init__(self,path,name,extVid,extSRT):
		self.VidFile 	= path+name+extVid
		self.subtLoc 	= path+name+extSRT
		self.video 		= cv2.VideoCapture(self.VidFile)
		self.nFrames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
		self.dw 			= self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.dh 			= self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
		self.wait 		= 10
		self.state ='play'


	def CheckAndOpenVid(self):
		if not self.video.isOpened():
			self.video 		= cv2.VideoCapture(self.VidFile)


	def startVideo(self):
		self.CheckAndOpenVid()
		self._oneFrame()
		cv2.waitKey(10)
		self._idle()

	def _oneFrame(self):
		ret,self.frame = self.video.read()
		if not ret:
			raise cv2.error('No hay más Frames')
		w = int(self.dw/3)
		h = int(self.dh/3)
		cv2.imshow('Video',self.frame [h:-h,w:-w])
		#cv2.imshow('Video',cv2.resize(frame,(self.dw,self.dh)))


	# States
	def Play(self):
		self.state ='play'
		while self.state=='play' :
			self._oneFrame()
			if cv2.waitKey(self.wait) & 0xFF == ord('s'):
				return self._idle()
				#return
			#self.Play()

	def PlayFromTo(self, startFrame,stopFrame):
		self.CheckAndOpenVid()
		self.state = 'playFromTo'
		self.video.set(cv2.CAP_PROP_POS_FRAMES,startFrame)
		delta 	= stopFrame-startFrame
		counter = 0
		while counter<delta and self.state == 'playFromTo':
			self._oneFrame()
			if cv2.waitKey(self.wait) & 0xFF == ord('s'):
				self.aux = [startFrame+counter,stopFrame]
				return self._idle()

			counter +=1

	def _idle(self):
		self.oldState= self.state
		self.state = 'idle'
		while self.state == 'idle':
			key = (cv2.waitKey(self.wait) & 0xFF)
			if key == ord('s'):
				if self.oldState=='play':
					return self.Play()
				elif self.oldState=='PlayFromTo':
					return self.PlayFromTo(self.aux[0],self.aux[1])
			elif key == ord('a'):
				pos = self.video.get(cv2.CAP_PROP_POS_FRAMES)
				if pos != 0 :
					self.video.set(cv2.CAP_PROP_POS_FRAMES,pos-2)
					self._oneFrame()
				else:
					print('Video en Frame 0!')
			elif key == ord('d'):
				self._oneFrame()
			elif key == ord('q'):
				cv2.destroyWindow('Video')
				print('cerrando video.')
				self.state = 'closed'
				return




# vid = Visualizer(path,name,extVid,extSRT)
# vid.startVideo()



# #%%

# import pandas as pd

# fl = open(path+'DJIFlightRecord_2019-09-04_[10-18-25]-TxtLogToCsv.csv',encoding='latin-1')
# df = pd.read_csv(fl)
# df=df[df['CUSTOM.isVideo']=='Recording']
# #%%
# #vx = df[df['CUSTOM.isVideo']=='Recording']['OSD.xSpeed [m/s]']
# #vy = df[df['CUSTOM.isVideo']=='Recording']['OSD.ySpeed [m/s]']

# vx = df['OSD.xSpeed [m/s]']
# vy = df['OSD.ySpeed [m/s]']

# plt.figure()
# plt.plot(vy.cumsum(),vx.cumsum())
# #%%

# lat = df['OSD.latitude']
# lon = df['OSD.longitude']
# plt.figure()
# plt.plot(lon,lat)

# #%%




