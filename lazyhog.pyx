# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np
'''
Based on _hog.py from https://github.com/scikit-image/scikit-image

Copyright (C) 2011, the scikit-image team
(C) 2013 Tim Sheerman-Chase
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in
	the documentation and/or other materials provided with the
	distribution.
 3. Neither the name of skimage nor the names of its contributors may be
	used to endorse or promote products derived from this software without
	specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin

cdef float CellHog(np.ndarray[np.float64_t, ndim=2] magnitude, 
	np.ndarray[np.float64_t, ndim=2] orientation,
	float ori1, float ori2):

	cdef int cx1, cy1

	cdef float total = 0.
	cdef int i
	for y in range(magnitude.shape[0]):
		for x in range(magnitude.shape[1]):
			if orientation[y, x] >= ori1: continue
			if orientation[y, x] < ori2: continue
	
			total += magnitude[y, x]

	return total

cdef HogThirdStage(magPatch, oriPatch,
	int visualise, int orientations, 
	np.ndarray[np.float64_t, ndim=2] orientation_histogram):

	"""
	The third stage aims to produce an encoding that is sensitive to
	local image content while remaining resistant to small changes in
	pose or appearance. The adopted method pools gradient orientation
	information locally in the same way as the SIFT [Lowe 2004]
	feature. The image window is divided into small spatial regions,
	called "cells". For each cell we accumulate a local 1-D histogram
	of gradient or edge orientations over all the pixels in the
	cell. This combined cell-level 1-D histogram forms the basic
	"orientation histogram" representation. Each orientation histogram
	divides the gradient angle range into a fixed number of
	predetermined bins. The gradient magnitudes of the pixels in the
	cell are used to vote into the orientation histogram.
	"""

	cdef int i, x, y, o, yi, xi, cy1, cy2, cx1, cx2, count, cellNum, centX, centY
	cdef float ori1, ori2

	# compute orientations integral images
	for i in range(orientations):
		# isolate orientations in this range

		ori1 = 180. / orientations * (i + 1)
		ori2 = 180. / orientations * i

		for patchNum in range(magPatch.shape[0]):
				
			orientation_histogram[patchNum, i] = \
				CellHog(magPatch[patchNum, :, :], oriPatch[patchNum, :, :], ori1, ori2)


def hog(magPatch, 
		oriPatch,
		int orientations=9, 
		int visualise=0):
	"""Extract Histogram of Oriented Gradients (HOG) for a given image.

	Compute a Histogram of Oriented Gradients (HOG) by

		1. (optional) global image normalisation
		2. computing the gradient image in x and y
		3. computing gradient histograms
		4. normalising block
		5. flattening into a feature vector

	Parameters
	----------
	image : (M, N) ndarray
		Input image (greyscale).
	cellOffsets:

	orientations : int
		Number of orientation bins.
	pixels_per_cell : 2 tuple (int, int)
		Size (in pixels) of a cell.
	visualise : bool, optional
		Also return an image of the HOG.
	normalise : bool, optional
		Apply power law compression to normalise the image before
		processing.

	Returns
	-------
	newarr : ndarray
		HOG for the image as a 1D (flattened) array.
	hog_image : ndarray (if visualise=True)
		A visualisation of the HOG image.

	References
	----------
	* http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

	* Dalal, N and Triggs, B, Histograms of Oriented Gradients for
	  Human Detection, IEEE Computer Society Conference on Computer
	  Vision and Pattern Recognition 2005 San Diego, CA, USA

	"""

	"""
	The first stage applies an optional global image normalisation
	equalisation that is designed to reduce the influence of illumination
	effects. In practice we use gamma (power law) compression, either
	computing the square root or the log of each colour channel.
	Image texture strength is typically proportional to the local surface
	illumination so this compression helps to reduce the effects of local
	shadowing and illumination variations.
	"""

	#if normalise:
	#	image = sqrt(image)

	"""
	The second stage computes first order image gradients. These capture
	contour, silhouette and some texture information, while providing
	further resistance to illumination variations. The locally dominant
	colour channel is used, which provides colour invariance to a large
	extent. Variant methods may also include second order image derivatives,
	which act as primitive bar detectors - a useful feature for capturing,
	e.g. bar like structures in bicycles and limbs in humans.
	"""

	cdef np.ndarray[np.float64_t, ndim=2] orientation_histogram = np.zeros((magPatch.shape[0], orientations))

	HogThirdStage(magPatch, oriPatch,
		visualise, orientations, orientation_histogram)
	
	#cdef np.ndarray[np.float64_t, ndim=2] hog_image
	#if visualise:
	#	hog_image = np.zeros((sy, sx), dtype=float)
	#	VisualiseHistograms(cx, cy, n_cellsx, n_cellsy, 
	#		orientations, orientation_histogram, hog_image)

	"""
	The fourth stage computes normalisation, which takes local groups of
	cells and contrast normalises their overall responses before passing
	to next stage. Normalisation introduces better invariance to illumination,
	shadowing, and edge contrast. It is performed by accumulating a measure
	of local histogram "energy" over local groups of cells that we call
	"blocks". The result is used to normalise each cell in the block.
	Typically each individual cell is shared between several blocks, but
	its normalisations are block dependent and thus different. The cell
	thus appears several times in the final output vector with different
	normalisations. This may seem redundant but it improves the performance.
	We refer to the normalised block descriptors as Histogram of Oriented
	Gradient (HOG) descriptors.
	"""

	eps = 1e-5
	normalised_block = orientation_histogram / sqrt(orientation_histogram.sum()**2 + eps)

	"""
	The final step collects the HOG descriptors from all blocks of a dense
	overlapping grid of blocks covering the detection window into a combined
	feature vector for use in the window classifier.
	"""

	#if visualise:
	#	return normalised_block.ravel(), hog_image
	#else:
	return normalised_block.ravel()

def GenerateCellPatchCentres():
	#Calculate cell centre positions
	cdef int cx = 8
	cdef int cy = 8
	y = cy / 2
	x = cx / 2
	cy2 = cy * 3
	cx2 = cx * 3
	cellOffsetsLi = []

	while y < cy2:
		x = cx / 2
		while x < cx2:
			cellOffsetsLi.append((x, y))
			x += cx
		y += cy
		
	cellOffsets = np.array(cellOffsetsLi, dtype=np.int32)
	return cellOffsets

def ExtractPatches(img, cellCentres, cx, cy):

	normalise = 0
	if normalise:
		img = sqrt(img)

	cdef int sy = img.shape[0]
	cdef int sx = img.shape[1]
	cdef np.ndarray[np.float64_t, ndim=2] gx = np.zeros((sy,sx))
	cdef np.ndarray[np.float64_t, ndim=2] gy = np.zeros((sy,sx))
	gx[:, :-1] = np.diff(img, n=1, axis=1)
	gy[:-1, :] = np.diff(img, n=1, axis=0)

	cdef np.ndarray[np.float64_t, ndim=2] magnitude = (gx**2 + gy**2) ** 0.5
	cdef np.ndarray[np.float64_t, ndim=2] orientation = arctan2(gy, gx) * (180 / 3.14159265359) % 180
	numCells = cellCentres.shape[0]
	magPatch = np.empty((numCells, cy, cx), dtype=np.float64)
	oriPatch = np.empty((numCells, cy, cx), dtype=np.float64)
	count = 0

	for cellNum in range(cellCentres.shape[0]):

		centX = cellCentres[cellNum, 0]
		centY = cellCentres[cellNum, 1]

		for y in range(cy):
			for x in range(cx):
				magPatch[count, y, x] = magnitude[y + centY - cy/2, x + centX - cx/2]
				oriPatch[count, y, x] = orientation[y + centY - cy/2, x + centX - cx/2]

		count += 1

	return magPatch, oriPatch

