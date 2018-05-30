import sys

from PyQt5 import uic, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from fast_gradient_method import FastGradientMethod

FormClass = uic.loadUiType("ui.ui")[0]


class Ui(QtWidgets.QMainWindow, FormClass):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.image_path = None
        self.model = FastGradientMethod()
        self.setup_events()

    def setup_events(self):
        self.loadButton.clicked.connect(self.load_original_image)
        self.modifyButton.clicked.connect(self.modify)

    def load_original_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, 'Choose A File')
        self.load_image(image_path, self.originalLabel)
        self.image_path = image_path
        original_class = self.model.predict_image(image_path)
        self.originalClassLabel.setText(original_class)

    def load_image(self, image_path, label):
        pixmap = QtGui.QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(label.size())
        label.setPixmap(scaled_pixmap)

    def modify(self):
        eps = self.epsSpinBox.value()
        original_class, adv_class, adv_path, perturbation_name = self.model.modify_image(self.image_path, eps)
        self.originalClassLabel.setText(original_class)
        self.advClassLabel.setText(adv_class)
        self.load_image(adv_path, self.advLabel)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui()
    ui.setWindowTitle("Seminar Demo")
    ui.show()
    app.exec_()
