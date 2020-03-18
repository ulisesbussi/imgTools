import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from scipy.signal import find_peaks


# %%


def _getLineImg(length, direction='h', lineWidth=5):
	"""Crea una imagen con una linea, para realizar la convolución
	es más una función interna del script que otra cosa.
	input args:
	length   : entero, largo de la imagen en la dirección que se extiende la linea
	direction: char 'v' o 'h' dirección vertical u horizontal respectivamente.
	linewidth: ancho de la linea que creará"""
	cover = 3  # ancho alrededor de linea
	cen = np.uint0(cover + lineWidth / 2)
	if direction == 'h':
		imageShape = [lineWidth + 2 * cover, length]
		args = [(0, cen), (imageShape[1], cen)]

	if direction == 'v':
		imageShape = [length, lineWidth + 2 * cover]
		args = [(cen, 0), (cen, imageShape[0])]

	blackImage = np.zeros(imageShape)

	return cv2.line(blackImage, *args, 255, lineWidth)


def _verticalConv(img, fil):
	"""convolucion vertical, para aplicar con una linea horizontal,
	img: imagen de entrada
	fil: filtro con el que se realiza la convolucion"""
	imShape = img.shape
	filShape = fil.shape
	convRes = []
	for i in range(imShape[0] - filShape[0] - 1):
		subImg = img[i:i + filShape[0]]
		prod = np.sum(subImg * fil)
		convRes.append(prod)
	convRes = np.array(convRes)
	return convRes


def _horizontalConv(img, fil):
	"""convolucion horizontal, para aplicar con una linea vertical,
	img: imagen de entrada
	fil: filtro con el que se realiza la convolucion"""
	imShape = img.shape
	filShape = fil.shape
	convRes = []
	for i in range(imShape[1] - filShape[1] - 1):
		subImg = img[:, i:i + filShape[1]]
		prod = np.sum(subImg * fil)
		convRes.append(prod)
	convRes = np.array(convRes)
	return convRes


def findNPeaks(data, n):
	peakIdx, _ = find_peaks(data)
	peakVals = data[peakIdx]
	orderedIdx = np.argsort(peakVals)[-n:]
	minDist = np.floor(0.7 * np.median(np.diff(np.sort(peakIdx[orderedIdx]))))
	bestPeakIdx, _ = find_peaks(data, distance=minDist)
	return bestPeakIdx


# imgs = [imutils.rotate(lp,ang) for ang in rotaciones]
# vertC = [verticalConv(im,horizontalLine) for im in imgs]
def findBestDirection(img, convFunc,
					  line, peaksNumber,
					  thetaList=np.arange(-5, 5)):
	"""
	:param img : imagen de entrada
	:param convFunc: tipo de función de convolución _verticalConv o _horizontalConv
	:param line: imagen de linea con la cual se buscará
	:param peaksNumber: prior del numero de picos a encontrar
	:param thetaList: 	lista de ángulos en los cuales se buscará la rotación
						por defecto [-5,-4,..,4,5]
	:return: diccionario conteniendo
				'img': imagen rotada de mejor resultado
				'peaks':  picos para esa imagen
				'convRes': valor de la convolucion
				'rotVal': valor de la rotacion en grados
				'maxIdx': indice de mayor valor en thetaList
	1) Roto la imagen con un vector de rotaciones
	2) Realizo la convolucion con las imagenes rotadas y les resto la media
	3) Encuentro los n picos, busco la que tiene mejor intensidad de picos en promedio
	4) Me quedo con ese indice
	esta funcion devuelve un diccionario de resultado, si solo quiero los valores
	hago fun(...).values()
	"""
	# 1)
	imgs = [imutils.rotate(img, ang) for ang in thetaList]
	# 2)
	convRes = [convFunc(im, line) for im in imgs]
	convRes = [cr - cr.mean() for cr in convRes]
	# 3)
	peaks = [findNPeaks(cr, peaksNumber) for cr in convRes]
	meanPeakIntensity = [np.mean(cr[p]) for cr, p in zip(convRes, peaks)]
	maxIntensityIdx = np.argmax(meanPeakIntensity)
	resDict = dict(img=imgs[maxIntensityIdx],
					peaks=peaks[maxIntensityIdx],
					convRes=convRes[maxIntensityIdx],
					rotVal=thetaList[maxIntensityIdx],
					maxIdx=maxIntensityIdx)
	return resDict


def crearImagenDeLineas(imShape, verticalPeaks, verticalRotationValue,
						horizontalPeaks, horizontalRotationValue,
						lineWidth=2):
	"""
		:param imShape : forma de la imagen de salida
		:param verticalPeaks: posición de los picos verticales
		:param verticalRotationValue: angulo de rotación de las rectas verticales
		:param horizontalPeaks: posición de los picos horizontales
		:param horizontalRotationValue: angulo de rotación de las rectas horizontales
		:param lineWidth: ancho de la linea a dibujar
		:return: imagen con lineas
	"""

	im = np.zeros(imShape)
	size = lambda th, iS: np.int0(np.sin(th * np.pi / 180) * iS / 2)
	sv, sh = map(size, [verticalRotationValue, horizontalRotationValue], imShape[:2])

	[cv2.line(im, (l + sv, 0), (l - sv, imShape[0]), 255, lineWidth) for l in verticalPeaks]
	[cv2.line(im, (0, l - sh), (imShape[1], l + sh), 255, lineWidth) for l in horizontalPeaks]
	return im



def findLines(img,nHorizontal=10,nVertical=10, rots = None):
	"""
	Encuentra las lineas en la imagen, tanto horizontales como verticales,
	devuelve dos diccionarios
	"""
	hLine = _getLineImg(img.shape[1],'h')
	vLine = _getLineImg(img.shape[0],'v')
	if rots is None:
		horizontalDict = findBestDirection(img,_verticalConv, hLine, nHorizontal)
		verticalDict   = findBestDirection(img,_horizontalConv, vLine,nVertical)
	else:
		horizontalDict = findBestDirection(img, _verticalConv, hLine, nHorizontal,rots)
		verticalDict   = findBestDirection(img, _horizontalConv, vLine, nVertical,rots)
	return (horizontalDict,verticalDict)
