from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")

        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setObjectName("label_20")
        self.label_20.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.horizontalLayout_6.addWidget(self.label_20)
        self.horizontalres_label = QtWidgets.QLabel(self.centralwidget)
        self.horizontalres_label.setObjectName(u"horizontalres_label")
        self.horizontalres_label.setMaximumSize(QtCore.QSize(150, 30)) 
        self.horizontalLayout_6.addWidget(self.horizontalres_label)


        self.horizontalres = QtWidgets.QLineEdit(self.centralwidget)
        self.horizontalres.setObjectName(u"horizontalres")
        self.horizontalres.setEnabled(True)
        self.horizontalres.setMaximumSize(QtCore.QSize(200, 16777215))
        self.horizontalres.setReadOnly(False)
        self.horizontalLayout_6.addWidget(self.horizontalres)
        
        self.verticalres_label = QtWidgets.QLabel(self.centralwidget)
        self.verticalres_label.setObjectName(u"verticalres_label")
        self.verticalres_label.setMaximumSize(QtCore.QSize(150, 30)) 
        self.horizontalLayout_6.addWidget(self.verticalres_label)

        self.verticalres = QtWidgets.QLineEdit(self.centralwidget)
        self.verticalres.setObjectName(u"verticalres")
        self.verticalres.setEnabled(True)
        self.verticalres.setMaximumSize(QtCore.QSize(200, 16777215))
        self.verticalres.setReadOnly(False)
        self.horizontalLayout_6.addWidget(self.verticalres)

        self.horizontalLayout_6.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_6.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("patreon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_3.setIcon(icon)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_6.addWidget(self.pushButton_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frameView = QtWidgets.QLabel(self.centralwidget)
        self.frameView.setText("")
        self.frameView.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.frameView.setObjectName("frameView")
        self.horizontalLayout.addWidget(self.frameView)
        self.frameViewRender = QtWidgets.QLabel(self.centralwidget)
        self.frameViewRender.setText("")
        self.frameViewRender.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.frameViewRender.setObjectName("frameViewRender")
        self.horizontalLayout.addWidget(self.frameViewRender)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.inputBtn = QtWidgets.QPushButton(self.centralwidget)
        self.inputBtn.setObjectName("inputBtn")
        self.horizontalLayout_7.addWidget(self.inputBtn)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_7.addWidget(self.line)
        self.frameRenderBtn = QtWidgets.QPushButton(self.centralwidget)
        self.frameRenderBtn.setObjectName("frameRenderBtn")
        self.horizontalLayout_7.addWidget(self.frameRenderBtn)
        self.autoCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.autoCheck.setChecked(True)
        self.autoCheck.setObjectName("autoCheck")
        self.horizontalLayout_7.addWidget(self.autoCheck)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_2.addWidget(self.label_6)
        self.outputMode = QtWidgets.QComboBox(self.centralwidget)
        self.outputMode.setObjectName("outputMode")
        self.outputMode.addItem("")
        self.outputMode.addItem("")
        self.outputMode.addItem("")
        self.horizontalLayout_2.addWidget(self.outputMode)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_2.addWidget(self.label_9)
        self.interpolationType = QtWidgets.QComboBox(self.centralwidget)
        self.interpolationType.setObjectName("interpolationType")
        self.interpolationType.addItem("")
        self.interpolationType.addItem("")
        self.interpolationType.addItem("")
        self.horizontalLayout_2.addWidget(self.interpolationType)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_8.addWidget(self.label_4)
        self.blurTresh = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.blurTresh.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.IsleOfMan))
        self.blurTresh.setProperty("value", 5.0)
        self.blurTresh.setObjectName("blurTresh")
        self.horizontalLayout_8.addWidget(self.blurTresh)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_8.addWidget(self.label_5)
        self.blurForce = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.blurForce.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.blurForce.setProperty("value", 10.0)
        self.blurForce.setObjectName("blurForce")
        self.horizontalLayout_8.addWidget(self.blurForce)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_8.addWidget(self.label_10)
        self.blurStrength = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.blurStrength.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.blurStrength.setProperty("value", 1.0)
        self.blurStrength.setObjectName("blurStrength")
        self.horizontalLayout_8.addWidget(self.blurStrength)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_8.addWidget(self.label_8)
        self.blurSmooth = QtWidgets.QSpinBox(self.centralwidget)
        self.blurSmooth.setProperty("value", 5)
        self.blurSmooth.setObjectName("blurSmooth")
        self.horizontalLayout_8.addWidget(self.blurSmooth)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.frameQtd = QtWidgets.QLabel(self.centralwidget)
        self.frameQtd.setObjectName("frameQtd")
        self.horizontalLayout_5.addWidget(self.frameQtd)
        self.frameSlider = QtWidgets.QSlider(self.centralwidget)
        self.frameSlider.setOrientation(QtCore.Qt.Horizontal)
        self.frameSlider.setObjectName("frameSlider")
        self.horizontalLayout_5.addWidget(self.frameSlider)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_4.addWidget(self.label_3)
        self.gpuBox = QtWidgets.QComboBox(self.centralwidget)
        self.gpuBox.setObjectName("gpuBox")
        self.horizontalLayout_4.addWidget(self.gpuBox)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_4.addWidget(self.label_7)
        self.formatBox = QtWidgets.QComboBox(self.centralwidget)
        self.formatBox.setObjectName("formatBox")
        self.formatBox.addItem("")
        self.formatBox.addItem("")
        self.horizontalLayout_4.addWidget(self.formatBox)
        self.renderBtn = QtWidgets.QPushButton(self.centralwidget)
        self.renderBtn.setObjectName("renderBtn")
        self.horizontalLayout_4.addWidget(self.renderBtn)
        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(4, 7)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout_2.setStretch(2, 10)
        self.verticalLayout_2.setStretch(4, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 854, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Credits"))
        self.pushButton_2.setText(_translate("MainWindow", "Patreons!"))
        self.pushButton_3.setText(_translate("MainWindow", "Support us on Patreon"))
        self.label_2.setText(_translate("MainWindow", "Input Footage:"))
        self.label.setText(_translate("MainWindow", "Output Footage:"))
        self.inputBtn.setText(_translate("MainWindow", "Select Input"))
        self.frameRenderBtn.setText(_translate("MainWindow", "Render OutputFrame"))
        self.autoCheck.setText(_translate("MainWindow", "Auto Update Output Frame On Change"))
        self.label_6.setText(_translate("MainWindow", "Output Type:"))
        self.outputMode.setItemText(0, _translate("MainWindow", "Motion Blur"))
        self.outputMode.setItemText(1, _translate("MainWindow", "Motion Vector"))
        self.outputMode.setItemText(2, _translate("MainWindow", "Normalized Motion Vector"))
        self.label_9.setText(_translate("MainWindow", "Blur Interpolation"))
        self.interpolationType.setItemText(0, _translate("MainWindow", "Easy In"))
        self.interpolationType.setItemText(1, _translate("MainWindow", "Easy Out"))
        self.interpolationType.setItemText(2, _translate("MainWindow", "Linear"))
        self.label_4.setText(_translate("MainWindow", "Blur Threshold:"))
        self.label_5.setText(_translate("MainWindow", "Blur Force:"))
        self.label_10.setText(_translate("MainWindow", "Blur Strength:"))
        self.label_8.setText(_translate("MainWindow", "Blur Smooth:"))
        self.frameQtd.setText(_translate("MainWindow", "Frame: 0"))
        self.label_3.setText(_translate("MainWindow", "GPU:"))
        self.label_7.setText(_translate("MainWindow", "Output Format:"))
        self.label_20.setText(_translate("MainWindow", "Resolution: "))
        self.horizontalres_label.setText(_translate("MainWindow", "Horizontal Res: "))
        self.verticalres_label.setText(_translate("MainWindow", "Vertical Res: "))
        self.formatBox.setItemText(0, _translate("MainWindow", "MP4"))
        self.formatBox.setItemText(1, _translate("MainWindow", "PNG Seq."))
        self.renderBtn.setText(_translate("MainWindow", "Render Video"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
