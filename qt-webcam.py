
import sys, time, cv, cv2, multiprocessing
from PyQt4 import QtGui, QtCore
import numpy as np
	
def detect(img, cascade):
	rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10), flags = cv.CV_HAAR_SCALE_IMAGE)
	if len(rects) == 0:
		return np.array([])
	rects[:,2:] += rects[:,:2]
	return rects

class CamWorker(multiprocessing.Process): 

	#webcamFrameSignal = QtCore.pyqtSignal(np.ndarray, name='webcam_frame')
	#facesDetectedSignal = QtCore.pyqtSignal(np.ndarray, name='faces_detected')

	def __init__(self, childConnIn): 
		super(CamWorker, self).__init__()
		self.cap = cv2.VideoCapture(-1)
		self.cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
		self.childConn = childConnIn

	def __del__(self):
		print "Worker stopping"

	def run(self):
		running = True
		while running:
			time.sleep(0.01)
			_,frame = self.cap.read()
			self.childConn.send(["frame",frame])

			if self.childConn.poll(0):
				ev = self.childConn.recv()
				if ev == "quit":
					running = False
			#gray = cv2.cvtColor(frame, cv.CV_RGB2GRAY)
			#gray = cv2.equalizeHist(gray)
			#rects = detect(gray, self.cascade)
			#self.facesDetectedSignal.emit(rects)

		self.childConn.send(["done",1])

class MainWindow(QtGui.QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__() 
		self.resize(700, 550)
		self.move(300, 300)
		self.setWindowTitle('Qt Webcam Demo')
		self.faces = np.array([])
		self.parentConn = None

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
		#act = QtGui.QAction("timeout()", self.ctimer)
		#act.triggered.connect(self.Test)
		self.ctimer.start(10)

	def __del__(self):
		pass

	def CheckForEvents(self):
		if self.parentConn.poll(0):
			ev = self.parentConn.recv()
			if ev[0] == "frame" and ev[1] is not None:
				self.ProcessFrame(ev[1])

	def ProcessFrame(self, im):
		print "Frame update", im.shape
		im = QtGui.QImage(im.tostring(), im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)
		self.pix = QtGui.QPixmap(im.rgbSwapped())
		self.RefreshDisplay()

	def ProcessFaces(self, faces):
		self.faces = faces

	def RefreshDisplay(self):
		self.scene.clear()
		self.scene.addPixmap(self.pix)
		for fnum in range(self.faces.shape[0]):
			face = self.faces[fnum,:]
			self.scene.addRect(face[0],face[1],face[2]-face[0],face[3]-face[1])

	def closeEvent(self, event):
		self.parentConn.send("quit")
		self.parentConn.recv()

if __name__ == '__main__':

	app = QtGui.QApplication(sys.argv)

	mainWindow = MainWindow()

	parentConn, childConn = multiprocessing.Pipe()
	camWorker = CamWorker(childConn)
	mainWindow.parentConn = parentConn
	camWorker.start()

	ret = app.exec_()

	sys.exit(ret)

