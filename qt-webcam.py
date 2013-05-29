
import sys, time, cv
from PyQt4 import QtGui, QtCore
	
class CamWorker(QtCore.QThread): 
    def __init__(self): 
		super(CamWorker, self).__init__() 
		self.cap = cv.CaptureFromCAM(-1)
		capture_size = (640,480)
		cv.SetCaptureProperty(self.cap, cv.CV_CAP_PROP_FRAME_WIDTH, capture_size[0])
		cv.SetCaptureProperty(self.cap, cv.CV_CAP_PROP_FRAME_HEIGHT, capture_size[1])

    def run(self):
		while 1:
			time.sleep(0.01)
			frame = cv.QueryFrame(self.cap)
			im = QtGui.QImage(frame.tostring(), frame.width, frame.height, QtGui.QImage.Format_RGB888).rgbSwapped()	
			self.emit(QtCore.SIGNAL('webcam_frame(QImage)'), im)





class MainWindow(QtGui.QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__() 
		self.resize(250, 150)
		self.move(300, 300)
		self.setWindowTitle('Simple')

		self.scene = QtGui.QGraphicsScene(self)
		self.view  = QtGui.QGraphicsView(self.scene)

		self.vbox = QtGui.QVBoxLayout()
		self.vbox.addWidget(self.view)

		centralWidget = QtGui.QWidget()
		centralWidget.setLayout(self.vbox)
		self.setCentralWidget(centralWidget)
		self.show()


	def SomeFunc(self, im):
		print "Frame update"
		pix = QtGui.QPixmap(im)
		self.scene.addPixmap(pix)


def main():
	app = QtGui.QApplication(sys.argv)

	w = QtGui.QWidget()
	w.resize(250, 150)
	w.move(300, 300)
	w.setWindowTitle('Simple')

	camWorker = CamWorker()
	QtCore.QObject.connect(camWorker, QtCore.SIGNAL("webcam_frame(QImage)"), SomeFunc)
	camWorker.start() 

	global scene
	scene = QtGui.QGraphicsScene(w)
	view  = QtGui.QGraphicsView(scene)
   
	#rect = QtGui.QGraphicsRectItem(10, 10, 50, 50)
	#scene.addItem(rect)

	vbox = QtGui.QVBoxLayout()
	vbox.addWidget(view)
	w.setLayout(vbox)
	w.show()
 
	sys.exit(app.exec_())


if __name__ == '__main__':
	#main()
	#CamWorker()
	app = QtGui.QApplication(sys.argv)

	mainWindow = MainWindow()

	camWorker = CamWorker()
	QtCore.QObject.connect(camWorker, QtCore.SIGNAL("webcam_frame(QImage)"), mainWindow.SomeFunc)
	camWorker.start() 

	sys.exit(app.exec_())

