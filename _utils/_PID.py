#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:28:57 2019

@author: ulises
"""



class PID(object):
	def __init__(self,kp=1,ki=0,kd=0,**kwargs):
		self.kp=kp
		self.ki=ki
		self.kd=kd

		self.dv = 0
		self.iv = 0
		self.pv = 0
		self.oldMed = 0
	def updatePID(self,med):
		self.pv = med
		self.iv += med
		self.dv = med- self.oldMed
		self.corr =  self.pv*self.kp +\
				self.iv*self.ki +\
				self.dv*self.kd
		self.oldMed = med

	def getCor(self):
		return self.corr