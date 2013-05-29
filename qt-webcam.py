
import sys, time, cv, cv2
from PyQt4 import QtGui, QtCore
import numpy as np
	
webcamFrameSignal = QtCore.pyqtSignal(np.ndarray, name='webcam_frame')

class CamWorker(QtCore.QThread): 

	webcamFrameSignal = QtCore.pyqtSignal(np.ndarray, name='webcam_frame')

	def __init__(self): 
		super(CamWorker, self).__init__()
		self.cap = cv2.VideoCapture(-1)
		cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

	def run(self):
		while 1:
			time.sleep(0.01)
			_,frame = self.cap.read()
			self.webcamFrameSignal.emit(frame)

			#gray = cv2.cvtColor(frame, cv.CV_RGB2GRAY)
			#gray = cv2.equalizeHist(gray)
			#rects = detect(gray, cascade)
			#print rects
			

class MainWindow(QtGui.QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__() 
		self.resize(700, 550)
		self.move(300, 300)
		self.setWindowTitle('Qt Webcam Demo')

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
		print im.shape, im.strides
		im = QtGui.QImage(im.tostring(), im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)
		print im, im.size().width(), im.size().height()
		pix = QtGui.QPixmap(im.rgbSwapped())
		print pix
		self.scene.clear()
		self.scene.addPixmap(pix)


if __name__ == '__main__':

	app = QtGui.QApplication(sys.argv)

	mainWindow = MainWindow()

	camWorker = CamWorker()
	camWorker.webcamFrameSignal.connect(mainWindow.ProcessFrame)
	camWorker.start()

	sys.exit(app.exec_())

