
import sys, time, cv, cv2, multiprocessing, pickle, supra, normalisedImage, yappi
from PyQt4 import QtGui, QtCore
import numpy as np
from PIL import Image
import skimage.io as io

def detect(img, cascade):
	rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10), flags = cv.CV_HAAR_SCALE_IMAGE)
	if len(rects) == 0:
		return np.array([])
	rects[:,2:] += rects[:,:2]
	return rects

def DrawPoint(scene,x,y):
	pen = QtGui.QPen(QtGui.QColor(255,0,0))
	scene.addLine(x-10,y,x+10,y,pen)
	scene.addLine(x,y-10,x,y+10,pen)

class CamWorker(multiprocessing.Process): 

	def __init__(self, childConnIn): 
		super(CamWorker, self).__init__()
		self.cap = cv2.VideoCapture(-1)
		self.childConn = childConnIn
		self.count = 0

	def __del__(self):
		self.cap.release()
		print "CamWorker stopping"

	def run(self):
		running = True
		while running:
			time.sleep(0.01)
			_,frame = self.cap.read()
			#Reorder channels
			frame = frame[:,:,[2,1,0]]
			self.childConn.send(["frame", frame, self.count])
			self.count += 1

			if self.childConn.poll(0):
				ev = self.childConn.recv()
				if ev[0] == "quit":
					running = False

		self.childConn.send(["done",1])

class DetectorWorker(multiprocessing.Process): 
	def __init__(self, childConnIn): 
		super(DetectorWorker, self).__init__()
		self.cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
		self.childConn = childConnIn

	def __del__(self):
		print "DetectorWorker stopping"

	def run(self):
		running = True
		while running:
			time.sleep(0.01)

			if self.childConn.poll(0):
				ev = self.childConn.recv()
				if ev[0] == "quit":
					running = False
				if ev[0] == "frame":
					gray = cv2.cvtColor(ev[1], cv.CV_RGB2GRAY)
					gray = cv2.equalizeHist(gray)
					rects = detect(gray, self.cascade)
					self.childConn.send(["faces",rects])

		self.childConn.send(["done",1])

class TrackingWorker(multiprocessing.Process): 
	def __init__(self, childConnIn): 
		super(TrackingWorker, self).__init__()
		self.childConn = childConnIn
		self.detectPtsPos = [(0.32, 0.38), (1.-0.32,0.38), (0.5,0.6), (0.35, 0.77), (1.-0.35, 0.77)]
		self.meanFace = pickle.load(open("meanFace.dat", "rb"))
		self.currentFrame = None
		self.currentFrameNum = None
		self.normIm = None
		self.tracker = None
		self.currentModel = None
		self.prevFrameFeatures = None
		self.trackingPending = False
		self.count = 0

	def __del__(self):
		print "TrackingWorker stopping"

	def run(self):
		running = True
		#self.tracker = pickle.load(open("tracker-5pt-halfdata.dat", "rb"))
		self.tracker = pickle.load(open("tracker-mirror.dat", "rb"))
		yappi.start(True)

		while running:
			time.sleep(0.01)

			while self.childConn.poll(0):
				ev = self.childConn.recv()
				if ev[0] == "quit":
					running = False
				if ev[0] == "frame":
					self.currentFrame = ev[1]
					self.trackingPending = True
					self.currentFrameNum = ev[2]

				if ev[0] == "faces":
					posModel = []
					if len(ev[1]) == 0: continue
					face = ev[1][0]
					w = face[2]-face[0]
					h = face[3]-face[1]
					for pt in self.detectPtsPos:
						posModel.append((face[0] + pt[0] * w, face[1] + pt[1] * h))
					if self.currentModel is None:
						self.currentModel = posModel
				time.sleep(0.01)

			if self.trackingPending and self.currentModel is not None:
				#Normalise input image using procrustes
				self.normIm = normalisedImage.NormalisedImage(self.currentFrame, self.currentModel, self.meanFace, {})
				#normalisedImage.SaveNormalisedImageToFile(self.normIm, "img.jpg")

				#Convert coordinates to normalised space
				normPosModel = [self.normIm.GetNormPos(*pt) for pt in self.currentModel]
				print "normPosModel",normPosModel

				#Initialise prev features if necessary
				if self.prevFrameFeatures is None:
					self.prevFrameFeatures = self.tracker.CalcPrevFrameFeatures(self.normIm, normPosModel)

				#Make prediction
				pred = self.tracker.Predict(self.normIm, normPosModel, self.prevFrameFeatures)

				#Convert prediction back to image space
				self.currentModel = [self.normIm.GetPixelPosImPos(*pt) for pt in pred]

				print self.currentFrameNum, self.currentModel
				#io.imsave("currentFrame.jpg", self.currentFrame)
				self.childConn.send(["tracking", self.currentModel])
				self.trackingPending = False
				self.prevFrameFeatures = self.tracker.CalcPrevFrameFeatures(self.normIm, pred)
				
				self.count += 1
				if self.count > 10:	
					yappi.stop()
					stats = yappi.get_stats()
					pickle.dump(stats, open("prof.dat","wb"), protocol = -1)
					self.count = 0
					yappi.start()

			if self.trackingPending and self.currentModel is None:
				self.childConn.send(["tracking", None])
				self.trackingPending = False

		self.childConn.send(["done",1])


class MainWindow(QtGui.QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__() 
		self.resize(700, 550)
		self.move(300, 300)
		self.setWindowTitle('Qt Webcam Demo')
		self.faces = np.array([])
		self.cameraPipe = None
		self.detectorPipe = None
		self.detectionPending = False
		self.trackingPending = False
		self.detectPtsPos = [(0.32, 0.38), (1.-0.32,0.38), (0.5,0.6), (0.35, 0.77), (1.-0.35, 0.77)]
		self.tracking = []

		self.scene = QtGui.QGraphicsScene(self)
		self.view  = QtGui.QGraphicsView(self.scene)

		self.vbox = QtGui.QVBoxLayout()
		self.vbox.addWidget(self.view)

		centralWidget = QtGui.QWidget()
		centralWidget.setLayout(self.vbox)
		self.setCentralWidget(centralWidget)
		self.show()

		self.ctimer = QtCore.QTimer(self)
		self.ctimer.timeout.connect(self.CheckForEvents)
		self.ctimer.start(10)

	def __del__(self):
		pass

	def CheckPipeForEvents(self, pipe):
		try:
			eventWaiting = pipe.poll(0)
		except IOError:
			eventWaiting = 0
		if eventWaiting:
			try:
				ev = pipe.recv()
			except IOError:
				ev = [None, None]
			self.HandleEvent(ev)

	def HandleEvent(self,ev):
		if ev[0] == "frame" and ev[1] is not None:
			self.ProcessFrame(ev[1])
			if not self.detectionPending:
				self.detectorPipe.send(ev)
				self.detectionPending = True
			if not self.trackingPending:
				self.trackingPipe.send(ev)
				self.trackingPending = True

		if ev[0] == "faces":
			self.ProcessFaces(ev[1])
			self.detectionPending = False
			self.trackingPipe.send(ev)
	
		if ev[0] == "tracking":
			self.trackingPending = False
			self.tracking = ev[1]

	def CheckForEvents(self):
		self.CheckPipeForEvents(self.cameraPipe)
		self.CheckPipeForEvents(self.detectorPipe)
		self.CheckPipeForEvents(self.trackingPipe)

	def ProcessFrame(self, im):
		#print "Frame update", im.shape
		im = QtGui.QImage(im.tostring(), im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)
		self.pix = QtGui.QPixmap(im)
		self.RefreshDisplay()

	def ProcessFaces(self, faces):
		self.faces = faces

	def RefreshDisplay(self):
		self.scene.clear()
		self.scene.addPixmap(self.pix)
		for fnum in range(self.faces.shape[0]):
			face = self.faces[fnum,:]
			w = face[2]-face[0]
			h = face[3]-face[1]
			self.scene.addRect(face[0],face[1],w,h)
			#for pt in self.detectPtsPos:				
			#	DrawPoint(self.scene,face[0] + pt[0] * w,face[1] + pt[1] * h)
		if self.tracking is not None:
			for pt in self.tracking:
				DrawPoint(self.scene,pt[0], pt[1])

	def closeEvent(self, event):
		self.detectorPipe.send(["quit",1])
		self.cameraPipe.send(["quit",1])
		self.trackingPipe.send(["quit",1])
		try:
			self.detectorPipe.recv()
		except:
			pass
		try:
			self.cameraPipe.recv()
		except:
			pass
		try:
			self.trackingPipe.recv()
		except:
			pass

if __name__ == '__main__':

	app = QtGui.QApplication(sys.argv)

	mainWindow = MainWindow()

	parentConn, childConn = multiprocessing.Pipe()
	camWorker = CamWorker(childConn)
	mainWindow.cameraPipe = parentConn
	camWorker.start()

	parentConn, childConn = multiprocessing.Pipe()
	detectorWorker = DetectorWorker(childConn)
	mainWindow.detectorPipe = parentConn
	detectorWorker.start()

	parentConn, childConn = multiprocessing.Pipe()
	trackingWorker = TrackingWorker(childConn)
	mainWindow.trackingPipe = parentConn
	trackingWorker.start()

	ret = app.exec_()

	sys.exit(ret)

