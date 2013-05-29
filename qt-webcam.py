
import sys, time, cv, cv2
from PyQt4 import QtGui, QtCore
import numpy as np
	
def detect(img, cascade):
	rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10), flags = cv.CV_HAAR_SCALE_IMAGE)
	if len(rects) == 0:
		return np.array([])
	rects[:,2:] += rects[:,:2]
	return rects

class CamWorker(QtCore.QThread): 

	webcamFrameSignal = QtCore.pyqtSignal(np.ndarray, name='webcam_frame')
	facesDetectedSignal = QtCore.pyqtSignal(np.ndarray, name='faces_detected')

	def __init__(self): 
		super(CamWorker, self).__init__()
		self.cap = cv2.VideoCapture(-1)
		self.cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

	def run(self):
		while 1:
			time.sleep(0.01)
			_,frame = self.cap.read()
			self.webcamFrameSignal.emit(frame)

			gray = cv2.cvtColor(frame, cv.CV_RGB2GRAY)
			gray = cv2.equalizeHist(gray)
			rects = detect(gray, self.cascade)
			self.facesDetectedSignal.emit(rects)

class MainWindow(QtGui.QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__() 
		self.resize(700, 550)
		self.move(300, 300)
		self.setWindowTitle('Qt Webcam Demo')
		self.faces = np.array([])

		self.scene = QtGui.QGraphicsScene(self)
		self.view  = QtGui.QGraphicsView(self.scene)

		self.vbox = QtGui.QVBoxLayout()
		self.vbox.addWidget(self.view)

		centralWidget = QtGui.QWidget()
		centralWidget.setLayout(self.vbox)
		self.setCentralWidget(centralWidget)
		self.show()

	def ProcessFrame(self, im):
		print "Frame update", im.shape
		im = QtGui.QImage(im.tostring(), im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)
		self.pix = QtGui.QPixmap(im.rgbSwapped())


	def ProcessFaces(self, faces):
		self.faces = faces
		self.scene.clear()
		self.scene.addPixmap(self.pix)
		for fnum in range(self.faces.shape[0]):
			face = self.faces[fnum,:]
			self.scene.addRect(face[0],face[1],face[2]-face[0],face[3]-face[1])

if __name__ == '__main__':

	app = QtGui.QApplication(sys.argv)

	mainWindow = MainWindow()

	camWorker = CamWorker()
	camWorker.webcamFrameSignal.connect(mainWindow.ProcessFrame)
	camWorker.facesDetectedSignal.connect(mainWindow.ProcessFaces)
	camWorker.start()

	sys.exit(app.exec_())

