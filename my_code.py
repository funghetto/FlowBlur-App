import os
import subprocess
import re

from mainwindow import Ui_MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt, QSettings



import torch
import run
import webbrowser

import sys
import argparse

import cv2
import subprocess as sp
import PIL
import numpy as np

#import skvideo.io

from tqdm import tqdm

_DEBUG_MODE = False

class My_Ui_Dialog(Ui_MainWindow):
    
    def Init(self, Dialog, args):
        #frameViewRender

        #formatBox
        
        self.isFileSelected = False
        self.isSliderPressed = False
        self.model = None

        self.Dialog = Dialog

        self.totalFrames = 0
        self.selFrame = 1
        self.inputBtn.clicked.connect(self.OpenInputFile)
        self.frameSlider.sliderMoved.connect(self.OnSlider)

        self.frameRenderBtn.clicked.connect(self.ForceRender)
        

        self.frameSlider.sliderPressed.connect(self.OnSliderPressed)
        self.frameSlider.sliderReleased.connect(self.OnSliderReleased)
        self.frameSlider.valueChanged.connect(self.OnSliderChanged)

        self.renderBtn.clicked.connect(self.OnRender)

        self.gpuBox.currentIndexChanged.connect(self.SetDevices)
        self.outputMode.currentIndexChanged.connect(self.SelectFrameToShow)
        self.interpolationType.currentIndexChanged.connect(self.RenderFrame)


        self.pushButton.clicked.connect(self.OpenCredits)
        self.pushButton_2.clicked.connect(self.OpenPatrons)
        self.pushButton_3.clicked.connect(self.OpenPatron)

        
        self.GetDevices()

        self.blurTresh.valueChanged.connect(self.RenderFrame)
        self.blurForce.valueChanged.connect(self.RenderFrame)
        self.blurStrength.valueChanged.connect(self.RenderFrame)
        self.blurSmooth.valueChanged.connect(self.RenderFrame)
        
        
        if not torch.cuda.is_available():
            self.MessageBox("Error", "This application can be real slow without a compatible Nvidia Card.")

    def OpenInputFile(self, Dialog):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.selectFiles, _ = QFileDialog.getOpenFileName()

        process = subprocess.Popen(
            "ffmpeg -i \"{}\" -map 0:v:0 -c copy -f null -".format(self.selectFiles)
            , stdout=subprocess.PIPE , stderr=subprocess.STDOUT,universal_newlines=True)
        for line in process.stdout:
            if "frame=" in line:
                print(line)
                print(line.split())
                numbers = [int(s) for s in line.replace("frame=", "").split() if s.isdigit()]
                self.totalFrames = numbers[0]
        if self.totalFrames < 2:
            self.isFileSelected = False
            print("Could not extract more than one frame from input, stopping.")
            return

        print("Loading model, this can take a little while.")

        del self.model
        self.model = run.load_model(args)
        self.SetDevices()

        self.isFileSelected = True
        self.frameSlider.setMinimum(1)
        self.frameSlider.setMaximum(self.totalFrames - 1)
        self.RenderFrame()
        
    def OnSlider(self, pos):
        self.selFrame = pos
        self.frameQtd.setText("Frame: " + str(pos))
    def OnSliderPressed(self):
        self.isSliderPressed = True
    def OnSliderReleased(self):
        self.isSliderPressed = False
        self.RenderFrame()
    def OnSliderChanged(self):
        if not self.isSliderPressed:
            self.RenderFrame()

    def ForceRender(self):
        self._RenderFrame(True)
    
    def RenderFrame(self):
        self._RenderFrame(False)

    def _RenderFrame(self, force = False):
        if not self.isFileSelected:
            return
        if not self.autoCheck.isChecked() and force == False:
            return

        if _DEBUG_MODE:
            self.ExtractFrame(self.selFrame, "out.png", ",scale=480*2:-1")
            self.ExtractFrame(self.selFrame-1, "out0.png", ",scale=480*2:-1")
        else:
            self.ExtractFrame(self.selFrame, "out.png", "")
            self.ExtractFrame(self.selFrame-1, "out0.png", "")
        self.ShowFrame()
        tresh = float(self.blurTresh.text())
        force = float(self.blurForce.text())
        smooth = int(self.blurSmooth.text())
        strength = float(self.blurStrength.text())
        intepolation = self.interpolationType.currentIndex()
        
        run.run_pair(self.model, "out.png", "out0.png", tresh, force, strength, smooth, intepolation)
        self.SelectFrameToShow()
        
    def SelectFrameToShow(self):
        if not self.isFileSelected:
            return

        outIndex = self.outputMode.currentIndex()
        if outIndex == 0:
            self.ShowFrame("out.png", "blur.png")
        if outIndex == 1:
            self.ShowFrame("out.png", "flow.png")
        if outIndex == 2:
            self.ShowFrame("out.png", "nflow.png")
    def ExtractFrame(self, frame, name="out.png", resize = ""):
        os.system('ffmpeg -y -i "{}" -vf "select=eq(n\,{}){}" -hide_banner -loglevel panic -pix_fmt rgb24 -vframes 1 "{}"'.format(self.selectFiles, frame, resize, name))
        

    #Should not need to save as a file, just showing the data directly at the GUI
    def ShowFrame(self, i1 = "out.png", i2 = "out.png"):
        if not self.isFileSelected:
            return
        image_profile = QtGui.QImage(i1) #QImage object
        image_profile = image_profile.scaled(500,500, aspectRatioMode=QtCore.Qt.KeepAspectRatio, transformMode=QtCore.Qt.SmoothTransformation) # To scale image for example and keep its Aspect Ration    
        self.frameView.setPixmap(QtGui.QPixmap.fromImage(image_profile))

        image_profile = QtGui.QImage(i2) #QImage object
        image_profile = image_profile.scaled(500,500, aspectRatioMode=QtCore.Qt.KeepAspectRatio, transformMode=QtCore.Qt.SmoothTransformation)
        self.frameViewRender.setPixmap(QtGui.QPixmap.fromImage(image_profile))
    def OnRender(self):
        if not self.isFileSelected:
            return

        
        renderIndex = self.formatBox.currentIndex()

        if renderIndex == 0:
            print("Rendering MP4")

        if renderIndex == 1:
            print("Rendering PNG Seq.")

        self.RenderVideo(self.selectFiles, renderIndex)

    def GetDevices(self):
        count = torch.cuda.device_count()

        for i in reversed(range(0, self.gpuBox.count())):
            self.gpuBox.removeItem(i)
            
        for i in range(0, count):
            name = torch.cuda.get_device_name(i)
            self.gpuBox.addItem(name)

        if self.gpuBox.count() > 0:
            self.gpuBox.setCurrentIndex(torch.cuda.current_device())

	
    def SetDevices(self):
        device_num = self.gpuBox.currentIndex()
        self._device = device_num
        if self.model != None:
            self.model = self.model.cuda(device_num)

    def MessageBox(self, title, message):
        win = QWidget(self.Dialog)
        QMessageBox.about(win, title, message)
        win.show()
    def ConfirmBox(self, title, message, callback):
        win = QWidget()
        buttonReply = QMessageBox.question(win, title, message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            callback(True)
        else:
            callback(False)
        win.show()
    def SetDarkMode(self):
        C1 = 22

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)

        self.Dialog.setPalette( palette )

    def OpenCredits(self, Dialog):
        #ui = QtWidgets.QApplication(self.Dialog)
        dia = QtWidgets.QDialog()
        #dia.ui = ui
        
        c = [
        "<b>MotionFlow-App:</b>",
        "",
        "&nbsp;&nbsp;RAFT: Recurrent All Pairs Field Transforms for Optical Flow:",
        "&nbsp;&nbsp;&nbsp;&nbsp;Zachary Teed and Jia Deng",
        "<br />",
        "&nbsp;&nbsp;FlowBlur-App:",
        "&nbsp;&nbsp;&nbsp;Gabriel Poetsch"
        ]

        dia.setObjectName("Dialog")
        dia.resize(500, 600)
        gridLayout = QtWidgets.QGridLayout(dia)
        gridLayout.setObjectName("gridLayout")

        
        verticalLayout = QtWidgets.QVBoxLayout()
        verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        verticalLayout.setObjectName("verticalLayout")
        label = QtWidgets.QLabel(dia)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(30)
        label.setFont(font)
        label.setScaledContents(False)
        label.setAlignment(QtCore.Qt.AlignLeft)
        label.setObjectName("label")
        label.setText("<br />".join(c))
        verticalLayout.addWidget(label)
        gridLayout.addLayout(verticalLayout, 0, 0, 1, 1)

        dia.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dia.exec_()

    def OpenPatrons(self, Dialog):
        dia = QtWidgets.QDialog()
        
        f = open("patrons.txt", "r", encoding="utf-8")
        c = f.read()

        dia.setObjectName("Dialog")
        dia.resize(500, 600)
        gridLayout = QtWidgets.QGridLayout(dia)
        gridLayout.setObjectName("gridLayout")


        verticalLayout = QtWidgets.QVBoxLayout()
        verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        verticalLayout.setObjectName("verticalLayout")

        label = QtWidgets.QTextEdit(dia)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False),  
        font.setUnderline(False)
        font.setWeight(30)
        label.setFont(font)
        label.setReadOnly(True)
        label.setAlignment(QtCore.Qt.AlignLeft)
        label.setObjectName("label")
        label.setText(c)
        verticalLayout.addWidget(label)
        gridLayout.addLayout(verticalLayout, 0, 0, 1, 1)

        dia.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dia.exec_()
    def OpenPatron(self):
        webbrowser.open('https://www.patreon.com/DAINAPP')

    def RenderVideo(self, input_file, type = 0):
        cap = cv2.VideoCapture(input_file, cv2.CAP_FFMPEG)
        ret, framePrev = cap.read()
        framePrev = cv2.resize(framePrev, dsize=(3840, 2160), interpolation=cv2.INTER_CUBIC)

        basename = os.path.basename(input_file)
        basefile = os.path.splitext(basename)[0]
        dirname = os.path.dirname(input_file)

        pathMp4 = os.path.join(dirname, "blur_" + basename)
        pathPNG = os.path.join(dirname, "blur_" + basefile )

        if type == 0:
            print("Saving video to: " + pathMp4)
        if type == 1:
            print("Saving png sequence into: " + pathPNG)

        if type == 1 and not os.path.isdir(pathPNG):
            os.mkdir(pathPNG)

        
        pathBlur = os.path.join(pathPNG, "blur")
        pathMotion = os.path.join(pathPNG, "motion")
        pathMotionNor = os.path.join(pathPNG, "motion_norm")

        if type == 1 and not os.path.isdir(pathBlur):
            os.mkdir(pathBlur)
        if type == 1 and not os.path.isdir(pathMotion):
            os.mkdir(pathMotion)
        if type == 1 and not os.path.isdir(pathMotionNor):
            os.mkdir(pathMotionNor)

        height, width, ch = framePrev.shape

        dimension = '{}x{}'.format(width, height)
        fps = str(cap.get(cv2.CAP_PROP_FPS))

        print("DIM: {}".format(dimension))

        tresh = float(self.blurTresh.text())
        force = float(self.blurForce.text())
        smooth = int(self.blurSmooth.text())
        strength = float(self.blurStrength.text())
        intepolation = self.interpolationType.currentIndex()
        outIndex = self.outputMode.currentIndex()


        if type == 0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(pathMp4, fourcc, float(fps), (width,height))
            video.set(cv2.VIDEOWRITER_PROP_QUALITY, 99)
        i = 0


        with tqdm(total=self.totalFrames) as pbar:
            while True:
                ret, frameNext = cap.read()
                if not ret:
                    break
                frameNext = cv2.resize(frameNext, dsize=(3840, 2160), interpolation=cv2.INTER_AREA)

                framePrevC = torch.from_numpy(framePrev).permute(2, 0, 1).unsqueeze(0).float().to("cuda")
                frameNextC = torch.from_numpy(frameNext).permute(2, 0, 1).unsqueeze(0).float().to("cuda")

                
                

                blur, flo1, flo2 = run.run_pair_tensor(self.model, frameNextC, framePrevC, tresh, force, strength, smooth, intepolation)

                if _DEBUG_MODE:
                    cv2.imwrite(os.path.join(pathBlur, "f_{}.png".format(str(i).zfill(7))), blur.astype(np.uint8))
                    cv2.imwrite(os.path.join(pathMotion, "f_{}.png".format(str(i).zfill(7))), flo1.astype(np.uint8))
                    cv2.imwrite(os.path.join(pathMotionNor, "f_{}.png".format(str(i).zfill(7))), flo2.astype(np.uint8))
                else:
                    if outIndex == 0:
                        sel = blur.astype(np.uint8)
                    if outIndex == 1:
                        sel = flo1.astype(np.uint8)
                    if outIndex == 2:
                        sel = flo2.astype(np.uint8)

                    if type == 0:
                        video.write(sel)
                    if type == 1:
                        cv2.imwrite(os.path.join(pathPNG, "f_{}.png".format(str(i).zfill(7))), sel)
                

                
                
                #print("Written frame {}".format(i))
                pbar.update(1)
                i += 1

                framePrev = frameNext

        cv2.destroyAllWindows()
        if type == 0:
            video.release()
        print("Render Finish")


if __name__ == "__main__":
    print("Starting...")

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_false', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    Dialog = QtWidgets.QMainWindow()
    ui = My_Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.setWindowTitle("FlowBlur-App 1.2")
    

    ui.Init(Dialog, args)
    ui.SetDarkMode()
    #ui.TestPipe()

    Dialog.setWindowFlags(Dialog.windowFlags() |
    QtCore.Qt.WindowMinimizeButtonHint |
    QtCore.Qt.WindowSystemMenuHint)


    Dialog.show()
    sys.exit(app.exec_())
