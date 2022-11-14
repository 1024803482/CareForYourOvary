def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script

import os
import sys

from PyQt5.QtWidgets import QStatusBar, QLabel, QWidget, QFileDialog, QTextEdit,\
    QPushButton, QSlider, QComboBox, QDesktopWidget, QApplication
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QIcon, QColor
from PyQt5.QtCore import Qt

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import imageio
from PIL import Image

from Grad_CAM_cpu import GradCAM
import recognition_models
import segmentation_display


""" mmOvary_V1 Related List: """
color_list = [[128, 128, 0],      # 0 : Chocolate Cyst 1
              [128, 128, 0],      # 1 : Serous Cystadenoma 1
              [128, 128, 0],      # 2 : Teratoma 1
              [128, 128, 0],      # 3 : Thecal cell tumor 1
              [128, 128, 0],      # 4 : Simple cyst 1
              [0, 192, 0],        # 5 : Normal ovary 0
              [128, 128, 0],      # 6 : Mucinous cystadenoma 1
              [192, 0, 0],        # 7 : Serous cystadenocarcinoma 2
              [0, 0, 192]]        # 8 : Others 2

label_list = ["Chocolate Cyst",
              "Serous Cystadenoma",
              "Teratoma",
              "Thecal Cell Tumor",
              "Simple Cyst",
              "Normal Ovary",
              "Mucinous Cystadenoma",
              "Serous Cystadenocarcinoma",
              "Others"]

health_list = [1,
               1,
               1,
               1,
               1,
               0,
               1,
               2,
               2]


class IUADOT_DEMO(QWidget):
    def __init__(self):
        super(IUADOT_DEMO, self).__init__()
        """ 1. Geometry """
        window_width = 1300
        window_height = 650
        self.setFixedSize(window_width, window_height)
        # Background
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("./Library/Pic/14.jpg")))
        self.setPalette(palette)
        # MainWindow Icon
        self.setWindowIcon(QIcon(QPixmap("./Library/Icon/hospital.svg")))
        # MainWindow Title
        self.setWindowTitle("Ovary CAD （Reasearch）")

        self.statusBar = QStatusBar(self)
        self.statusBar.setFixedSize(150, 20)
        self.statusBar.move(window_width - 150, window_height - 20)
        self.statusBar.setStyleSheet('QStatusBar{color:rgb(20, 20, 20,); font-size:12px}')
        self.statusBar.showMessage("Author: Linghan Cai", 0)

        """ 2. Imshow Window """
        show_width = 450
        show_height = 500
        self.show_width = show_width
        self.show_height = show_height
        # Input Image Window in_label
        self.in_label = QLabel(self)
        self.in_label.setFixedSize(show_width, show_height)
        self.in_label.move(350, 100)
        self.in_label.setAlignment(Qt.AlignCenter)
        canvas = QPixmap(show_width, show_height)
        canvas.fill(QColor(112, 112, 112, 80))
        self.in_label.setPixmap(canvas)
        # Input Image Title in_title
        self.in_title = QLabel(self)
        self.in_title.setFixedSize(show_width, 50)
        self.in_title.move(350, 50)
        self.in_title.setStyleSheet("QLabel{font-size:18px; font-weight:bold; font-family:Arial;"
                                    "color:rgb(224, 224, 224); background-color:rgba(56, 56, 56, 128);}")
        self.in_title.setText("Input Image")
        self.in_title.setAlignment(Qt.AlignCenter)

        # Output Image Window out_label
        self.out_label = QLabel(self)
        self.out_label.setFixedSize(show_width, show_height)
        self.out_label.move(825, 100)
        self.out_label.setAlignment(Qt.AlignCenter)
        canvas = QPixmap(show_width, show_height)
        canvas.fill(QColor(112, 112, 112, 80))
        self.out_label.setPixmap(canvas)
        # Output Image Title out_title
        self.out_title = QLabel(self)
        self.out_title.setFixedSize(show_width, 50)
        self.out_title.move(825, 50)
        self.out_title.setStyleSheet("QLabel{font-size:18px; font-weight:bold; font-family:Arial;"
                                     "color:rgb(224, 224, 224); background-color:rgba(56, 56, 56, 128);}")
        self.out_title.setText("Output Image")
        self.out_title.setAlignment(Qt.AlignCenter)

        ''' Control Bar '''
        control_width = 300
        control_height = 500
        self.widget = QWidget(self)
        self.widget.setFixedSize(control_width, control_height)
        self.widget.setStyleSheet("QWidget{background-color:rgba(112, 112, 112, 64);}")
        self.widget.move(25, 100)
        # Control Bar Title
        self.control_title = QLabel(self)
        self.control_title.setFixedSize(control_width, 50)
        self.control_title.move(25, 50)
        self.control_title.setStyleSheet("QLabel{font-size:18px; font-weight:bold; font-family:Arial;"
                                         "color:rgb(224, 224, 224); background-color:rgba(56, 56, 56, 128);}")
        self.control_title.setText("Control Bar")
        self.control_title.setAlignment(Qt.AlignCenter)

        ''' Widget Settings '''
        button_height = 45
        button_width = 135
        # Load Image Button load_file_btn
        self.load_file_btn = QPushButton(self.widget)
        self.load_file_btn.setFixedSize(button_width, button_height)
        self.load_file_btn.setStyleSheet("QPushButton{font-size:18px; font-family:Arial; font-weight:bold;"
                                         "color:rgb(12, 12, 12,); background-color:rgba(96, 96, 96, 64)}"
                                         "QPushButton:hover{font-size:20px; font-family:Arial; font-weight:bold;"
                                         "color:rgb(224, 224, 224,); background-color:rgba(96, 96, 96, 64)}"
                                         )
        self.load_file_btn.setText("Load Image")
        self.load_file_btn.setIcon(QIcon('./Library/Icon/Open.svg'))
        self.load_file_btn.move(10, 25)
        self.load_file_btn.clicked.connect(self.loadFile)
        # Save Image Button save_file_btn
        self.save_file_btn = QPushButton(self.widget)
        self.save_file_btn.setFixedSize(button_width, button_height)
        self.save_file_btn.setStyleSheet("QPushButton{font-size:18px; font-family:Arial; font-weight:bold;"
                                         "color:rgb(12, 12, 12,); background-color:rgba(96, 96, 96, 64)}"
                                         "QPushButton:hover{font-size:20px; font-family:Arial; font-weight:bold;"
                                         "color:rgb(224, 224, 224,); background-color:rgba(96, 96, 96, 64)}"
                                         )
        self.save_file_btn.setText("Save Image")
        self.save_file_btn.setIcon(QIcon('./Library/Icon/Save.svg'))
        self.save_file_btn.move(155, 25)
        self.save_file_btn.clicked.connect(self.saveImage)
        # Image Classification reg_btn
        self.reg_btn = QPushButton(self.widget)
        self.reg_btn.setFixedSize(button_width, button_height)
        self.reg_btn.setStyleSheet("QPushButton{font-size:18px; font-family:Arial; font-weight:bold;"
                                   "color:rgb(12, 12, 12,); background-color:rgba(96, 96, 96, 64)}"
                                   "QPushButton:hover{font-size:20px; font-family:Arial; font-weight:bold;"
                                   "color:rgb(224, 224, 224,); background-color:rgba(96, 96, 96, 64)}"
                                   )
        self.reg_btn.setText("Classify")
        self.reg_btn.setIcon(QIcon('./Library/Icon/category.svg'))
        self.reg_btn.move(10, 90)
        self.reg_btn.clicked.connect(self.CAM)
        # Choose Classification Models reg_combo
        self.reg_name = QLabel(self.widget)
        self.reg_name.setFixedSize(button_width, button_height//2)
        self.reg_name.setStyleSheet("QLabel{font-size:14px; font-family:Arial; font-weight:bold;"
                                    "color:rgb(24, 24, 24); background-color:rgba(56, 56, 56, 0);}")
        self.reg_name.setAlignment(Qt.AlignLeft)
        self.reg_name.move(155, 90)
        self.reg_name.setText("Classifier:")

        self.reg_combo = QComboBox(self.widget)
        self.reg_combo.addItem('ResNeXt50')
        self.reg_combo.addItem('ResNet50')
        self.reg_combo.addItem('DenseNet121')
        self.reg_combo.setFixedSize(button_width, button_height//2)
        self.reg_combo.setStyleSheet("QComboBox{background: rgba(192, 192, 192, 192),}")
        self.reg_combo.move(155, 110)
        self.reg_combo.currentTextChanged.connect(self.RegModelChoose)
        # Segmentation Button seg_btn
        self.seg_btn = QPushButton(self.widget)
        self.seg_btn.setFixedSize(button_width, button_height)
        self.seg_btn.setStyleSheet("QPushButton{font-size:18px; font-family:Arial; font-weight:bold;"
                                   "color:rgb(12, 12, 12,); background-color:rgba(96, 96, 96, 64)}"
                                   "QPushButton:hover{font-size:20px; font-family:Arial; font-weight:bold;"
                                   "color:rgb(224, 224, 224,); background-color:rgba(96, 96, 96, 64)}"
                                   )
        self.seg_btn.setText("Segment")
        self.seg_btn.setIcon(QIcon('./Library/Icon/razor.svg'))
        self.seg_btn.move(10, 155)
        self.seg_btn.clicked.connect(self.Seg)
        # Choose Segmentation Model seg_combo
        self.seg_name = QLabel(self.widget)
        self.seg_name.setFixedSize(button_width, button_height // 2)
        self.seg_name.setStyleSheet("QLabel{font-size:14px; font-family:Arial; font-weight:bold;"
                                    "color:rgb(24, 24, 24); background-color:rgba(56, 56, 56, 0);}")
        self.seg_name.setAlignment(Qt.AlignLeft)
        self.seg_name.move(155, 155)
        self.seg_name.setText("Segmenter:")

        self.seg_combo = QComboBox(self.widget)
        self.seg_combo.addItem('DeepLabV3+')
        self.seg_combo.addItem('U-Net')
        self.seg_combo.addItem('PSPNet')
        self.seg_combo.setFixedSize(button_width, button_height // 2)
        self.seg_combo.setStyleSheet("QComboBox{background: rgba(192, 192, 192, 192),}")
        self.seg_combo.move(155, 175)
        self.seg_combo.currentTextChanged.connect(self.SegModelChoose)

        ''' Threshold Bar '''
        self.thr_label = QLabel(self.widget)
        self.thr_label.setFixedSize(200, button_height)
        self.thr_label.setStyleSheet("QLabel{font-size:18px; font-family:Arial; font-weight:bold;"
                                     "color:rgb(192, 192, 192); background-color:rgba(56, 56, 56, 0);}")
        self.thr_label.setAlignment(Qt.AlignCenter)
        self.thr_label.move(50, 205)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setParent(self.widget)
        self.slider.setMinimum(1)
        self.slider.setMaximum(19)
        self.slider.setSingleStep(1)
        self.slider.setValue(10)
        self.slider.setTickPosition(QSlider.TicksLeft)
        self.slider.setTickInterval(3)
        self.slider.setFixedSize(200, 20)
        self.slider.setToolTip("Threshold")
        self.slider.setStyleSheet("QSlider{\n"
                                  "border-color: #bcbcbc;\n"
                                  "color:#d9d9d9;\n"
                                  "}\n"
                                  "QSlider::groove:horizontal {                                \n"
                                  "     border: 1px solid #999999;                             \n"
                                  "     height: 3px;                                           \n"
                                  "    margin: 0px 0;                                         \n"
                                  "     left: 5px; right: 5px; \n"
                                  " }\n"
                                  "QSlider::handle:horizontal {                               \n"
                                  "     border: 0px ; \n"
                                  "     border-image: url(./Library/Icon/circle1.png);\n"
                                  "     width:15px;\n"
                                  "     margin: -7px -7px -7px -7px;                  \n"
                                  "} \n"
                                  "QSlider::add-page:horizontal{\n"
                                  "background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #d9d9d9, stop:0.25 #d9d9d9, stop:0.5 #d9d9d9, stop:1 #d9d9d9); \n"
                                  "\n"
                                  "}\n"
                                  "QSlider::sub-page:horizontal{                               \n"
                                  " background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #373737, stop:0.25 #373737, stop:0.5 #373737, stop:1 #373737);                     \n"
                                  "}")
        self.slider.move(50, 245)
        self.thr_label.setText("Threshold: {:.2f}".format(self.slider.value() / 20.0))
        self.slider.valueChanged.connect(self.valueChange)

        ''' Status Window '''
        self.state_name = QLabel(self.widget)
        self.state_name.setFixedSize(280, 20)
        self.state_name.move(10, 290)
        self.state_name.setStyleSheet("QLabel{font-size:16px; font-family:Arial; font-weight:bold;"
                                      "color:rgb(192, 192, 192); background-color:rgba(24, 24, 24, 128);}")
        self.state_name.setAlignment(Qt.AlignCenter)
        self.state_name.setText("Status")

        self.state_label = QTextEdit(self.widget)
        self.state_label.setReadOnly(True)
        self.state_label.setFixedSize(280, 160)
        self.state_label.move(10, 310)
        self.state_label.setStyleSheet("QTextEdit{font-size:14px; font-family:Arial;"
                                       "color:rgb(224, 224, 224); background-color:rgba(56, 56, 56, 64);}")
        self.state_label.setLineWrapMode(1)
        self.state_label.setText("Welcome to the ovarian ultrasound image aided diagnosis system:\n"
                                 "Loading ultrasonic images for intelligent analysis")
        self.state_label.setAlignment(Qt.AlignLeft)

        ''' Related Parameters '''
        self.in_name = None
        self.in_image = None
        self.out_image = None
        self.image_path = None
        self.reg_pred = None  # index
        self.seg_pred = None  # after sigmoid
        self.show_flag = 1    # 1 indicates seg; 2 indicates reg
        self.last_path = "./"
        self.save_path = "./save/"

        self.classifier = recognition_models.ResNeXt50(in_channels=3,
                                                       num_classes=9,
                                                       pretrained=False)
        self.classifier.load_state_dict(torch.load("./weights/classification/resnext50.pth",
                                                   map_location=torch.device("cpu")))
        self.classifier.eval()
        self.grad_cam = GradCAM(self.classifier, name='ResNeXt50', size=(256, 256), num_cls=9, )

        self.segmenter = smp.DeepLabV3Plus(encoder_name = "resnet34",
                                           encoder_depth = 5,
                                           encoder_weights = None,
                                           encoder_output_stride = 16,
                                           decoder_channels = 256,
                                           decoder_atrous_rates = (12, 24, 36),
                                           in_channels = 1,
                                           classes = 1,
                                           activation = None,
                                           upsampling = 4, )
        self.segmenter.load_state_dict(torch.load("./weights/segmentation/deeplabv3p.pth",
                                                  map_location=torch.device("cpu")))
        self.segmenter.eval()

    def centralization(self):  # window centralization
        screen = QDesktopWidget().screenGeometry()
        window = self.geometry()
        w = (screen.width() - window.width()) // 2
        h = (screen.height() - window.height()) // 2 - 50
        self.move(w, h)

    def __img_transform(self, img_arr: np.ndarray, transform: transforms) -> torch.Tensor:
        img = img_arr.copy()  # [H, W, C]
        img = Image.fromarray(np.uint8(img))
        img = transform(img).unsqueeze(0)  # [N,C,H,W]
        return img

    def __classify_img_preprocess(self, img_arr: np.ndarray) -> torch.Tensor:
        img = img_arr.copy()
        img = cv2.resize(img, (256, 256))
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        img_tensor = self.__img_transform(img, transform)
        return img_tensor

    def CAM(self):
        if self.in_image is None:
            self.state_label.setText("Ultrasound images have not been loaded")
            pass
        else:
            cam = self.grad_cam.forward(self.in_image)
            size = self.in_image.shape
            h, w = size[0], size[1]
            bg = np.array([112, 112, 112, 80]) * np.ones((self.show_height, self.show_width, 4), dtype=np.uint8)
            top = self.show_height // 2 - h // 2
            down = top + h
            left = self.show_width // 2 - w // 2
            right = left + w
            bg[top:down, left:right, 0:3] = cam
            self.out_image = cam
            bg[top:down, left:right, 3] = 255
            imageio.imwrite("./cache/cam.png", bg)
            self.out_label.setPixmap(QPixmap("./cache/cam.png"))
            self.out_title.setText(label_list[self.reg_pred])
            self.show_flag = 2

    def valueChange(self):
        # sender = self.sender()
        self.thr_label.setText("Threshold: {:.2f}".format(self.slider.value()/20.0))
        if self.show_flag == 1 and self.out_image is not None:
            self.Seg()
        elif self.show_flag == 2 and self.out_image is not None:
            self.state_label.setText("The current mode is class activation map mode. Please select "
                                     "\"lesion segmentation\" to view the segmentation "
                                     "results under different thresholds")
        else:
            self.state_label.setText("Ultrasound images have not been loaded")

    def __segment_img_preprocess(self, img_arr: np.ndarray) -> torch.Tensor:
        img = img_arr.copy()
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean, std = [0.5], [0.225]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        img_tensor = self.__img_transform(img, transform)
        return img_tensor

    def Seg(self):
        if self.in_image is None:
            self.state_label.setText("Ultrasound images have not been loaded")
            pass
        else:
            size = self.in_image.shape
            h, w = size[0], size[1]
            img = self.__segment_img_preprocess(self.in_image)
            with torch.no_grad():
                pred = self.segmenter(img)
            pred = 255 * torch.sigmoid(pred).cpu().numpy()
            pred = pred.squeeze()
            pred = np.reshape(pred, (256, 256))
            pred = np.uint8(pred)
            pred = cv2.resize(pred, (w, h))
            self.seg_pred = pred
            threshold = 255 * self.slider.value() / 20
            image = segmentation_display.vis_segmentation2(self.in_image, np.uint8(pred > threshold),
                                                           color_list[self.reg_pred])
            bg = np.array([112, 112, 112, 80]) * np.ones((self.show_height, self.show_width, 4), dtype=np.uint8)
            top = self.show_height // 2 - h // 2
            down = top + h
            left = self.show_width // 2 - w // 2
            right = left + w
            bg[top:down, left:right, 0:3] = image
            bg[top:down, left:right, 3] = 255
            imageio.imwrite("./cache/seg.png", bg)
            self.out_image = image
            self.out_label.setPixmap(QPixmap("./cache/seg.png"))
            self.out_title.setText(label_list[self.reg_pred])
            self.show_flag = 1

    def loadFile(self):
        image_path, file_type = QFileDialog.getOpenFileName(self, "Load image", self.last_path, "*.JPG;;*.PNG")
        if image_path == '':
            self.state_label.setText("Failed to load image: cancel loading image")
            pass
        else:
            self.in_name = image_path.split('/')[-1]
            self.in_image = imageio.imread(image_path)
            self.image_path = image_path
            self.last_path = image_path[0:-len(self.in_name)]
            size = self.in_image.shape
            h, w = size[0], size[1]
            if max([h, w]) > self.show_width - 20:
                ratio_h = 1.0 * (self.show_width - 20) / h
                ratio_w = 1.0 * (self.show_width - 20) / w
                ratio = min([ratio_h, ratio_w])
                h = int(ratio * h)
                w = int(ratio * w)
                self.in_image = cv2.resize(self.in_image, (w, h))

            bg = np.array([112, 112, 112, 80]) * np.ones((self.show_height, self.show_width, 4), dtype=np.uint8)
            top = self.show_height // 2 - h // 2
            down = top + h
            left = self.show_width // 2 - w // 2
            right = left + w
            bg[top:down, left:right, 0:3] = self.in_image
            bg[top:down, left:right, 3] = 255
            imageio.imwrite("./cache/in.png", bg)
            self.in_label.setPixmap(QPixmap("./cache/in.png"))
            # Load Image
            canvas = QPixmap(self.show_width, self.show_height)
            canvas.fill(QColor(112, 112, 112, 80))
            self.out_label.setPixmap(canvas)
            img_tensor = self.__classify_img_preprocess(self.in_image)
            reg_pred = torch.softmax(self.classifier(img_tensor), dim=-1)
            reg_prob = 100 * reg_pred.detach().cpu().numpy()[0]
            reg_index = np.argsort(reg_prob)[::-1]
            self.reg_pred = reg_index[0]
            self.Seg()

            text = label_list[reg_index[0]] + " : " + "{:.2f}%".format(reg_prob[reg_index[0]]) + "\n" + \
                   label_list[reg_index[1]] + " : " + "{:.2f}%".format(reg_prob[reg_index[1]]) + "\n" + \
                   label_list[reg_index[2]] + " : " + "{:.2f}%".format(reg_prob[reg_index[2]]) + "\n"

            self.state_label.setText("Image path:" + image_path +
                                     '\nClassification results and probability:\n' + text)

    def saveImage(self):
        if self.in_name is None:
            self.state_label.setText("Save failed: the image has not been loaded")
            pass
        else:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Image", self.save_path + self.in_name, "All Files (*)",)
            if self.out_image is None:
                self.state_label.setText("Save failed: output image not loaded")
            elif filename == "":
                self.state_label.setText("Save failed: user cancels saving")
            else:
                if len(filename) < 4 or filename[-4:] not in ['.PNG', '.png', '.JPG', '.jpg']:
                    filename += ".PNG"
                if self.show_flag == 1:
                    imageio.imwrite(filename, self.out_image)
                    threshold = 255 * self.slider.value() / 20
                    imageio.imwrite(filename[:-4]+"_mask.PNG", 255 * np.uint8(self.seg_pred > threshold))
                    self.state_label.setText("Save successfully!\n" + "save segmentation result to:" + filename +
                                             "\nsave mask to:" + filename[:-4]+"_mask.PNG")
                elif self.show_flag == 2:
                    imageio.imwrite(filename, self.out_image)
                    self.state_label.setText("Save successfully!\n" + "save heatmap to:" + filename)

                name = filename.split('/')[-1]
                self.save_file_btn = filename[:-len(name)]

    def RegModelChoose(self):
        model_name = self.reg_combo.currentText()
        if model_name == "ResNeXt50":
            self.classifier = recognition_models.ResNeXt50(in_channels=3,
                                                           num_classes=9,
                                                           pretrained=False)
            self.classifier.load_state_dict(torch.load("./weights/classification/resnext50.pth",
                                                       map_location=torch.device("cpu")))
            self.grad_cam = GradCAM(self.classifier, name='ResNeXt50', size=(256, 256), num_cls=9, )
        elif model_name == "ResNet50":
            self.classifier = recognition_models.ResNet50(in_channels=3,
                                                          num_classes=9,
                                                          pretrained=False)
            self.classifier.load_state_dict(torch.load("./weights/classification/resnet50.pth",
                                                       map_location=torch.device("cpu")))
            self.grad_cam = GradCAM(self.classifier, name='ResNet50', size=(256, 256), num_cls=9, )
        elif model_name == "DenseNet121":
            self.classifier = recognition_models.DenseNet121(in_channels=3,
                                                             num_classes=9,
                                                             pretrained=False)
            self.classifier.load_state_dict(torch.load("./weights/classification/densenet121.pth",
                                                       map_location=torch.device("cpu")))
            self.grad_cam = GradCAM(self.classifier, name='DenseNet121', size=(256, 256), num_cls=9, )
        self.classifier.eval()
        if self.out_image is not None:
            if self.show_flag == 1:
                canvas = QPixmap(self.show_width, self.show_height)
                canvas.fill(QColor(112, 112, 112, 80))
                self.out_label.setPixmap(canvas)
                img_tensor = self.__classify_img_preprocess(self.in_image)
                reg_pred = torch.softmax(self.classifier(img_tensor), dim=-1)
                reg_prob = 100 * reg_pred.detach().cpu().numpy()[0]
                reg_index = np.argsort(reg_prob)[::-1]
                self.reg_pred = reg_index[0]
                self.Seg()

                text = label_list[reg_index[0]] + " : " + "{:.2f}%".format(reg_prob[reg_index[0]]) + "\n" + \
                       label_list[reg_index[1]] + " : " + "{:.2f}%".format(reg_prob[reg_index[1]]) + "\n" + \
                       label_list[reg_index[2]] + " : " + "{:.2f}%".format(reg_prob[reg_index[2]]) + "\n"

                self.state_label.setText("Classifier:" + model_name + "\nImage path:" +
                                         self.image_path + '\nClassification results and probability:\n' + text)
            elif self.show_flag == 2:
                canvas = QPixmap(self.show_width, self.show_height)
                canvas.fill(QColor(112, 112, 112, 80))
                self.out_label.setPixmap(canvas)
                img_tensor = self.__classify_img_preprocess(self.in_image)
                reg_pred = torch.softmax(self.classifier(img_tensor), dim=-1)
                reg_prob = 100 * reg_pred.detach().cpu().numpy()[0]
                reg_index = np.argsort(reg_prob)[::-1]
                self.reg_pred = reg_index[0]
                self.CAM()

                text = label_list[reg_index[0]] + " : " + "{:.2f}%".format(reg_prob[reg_index[0]]) + "\n" + \
                       label_list[reg_index[1]] + " : " + "{:.2f}%".format(reg_prob[reg_index[1]]) + "\n" + \
                       label_list[reg_index[2]] + " : " + "{:.2f}%".format(reg_prob[reg_index[2]]) + "\n"

                self.state_label.setText("Classifier:" + model_name + "\nImage path:" +
                                         self.image_path + '\nClassification results and probability:\n' + text)
        else:
            self.state_label.setText("Classifier:" + model_name)

    def SegModelChoose(self):
        model_name = self.seg_combo.currentText()
        if model_name == "DeepLabV3+":
            self.segmenter = smp.DeepLabV3Plus(encoder_name = "resnet34",
                                               encoder_depth = 5,
                                               encoder_weights = None,
                                               encoder_output_stride = 16,
                                               decoder_channels = 256,
                                               decoder_atrous_rates = (12, 24, 36),
                                               in_channels = 1,
                                               classes = 1,
                                               activation = None,
                                               upsampling = 4, )
            self.segmenter.load_state_dict(torch.load("./weights/segmentation/deeplabv3p.pth",
                                                      map_location=torch.device("cpu")))
        elif model_name == "U-Net":
            self.segmenter = smp.Unet(encoder_depth = 5,
                                      encoder_weights = None,
                                      decoder_channels = [256, 128, 64, 32, 16],
                                      in_channels = 1,
                                      classes = 1)
            self.segmenter.load_state_dict(torch.load("./weights/segmentation/unetmini.pth",
                                                      map_location=torch.device("cpu")))
        elif model_name == "PSPNet":
            self.segmenter = smp.PSPNet(encoder_name = "resnet34",
                                        encoder_weights = None,
                                        encoder_depth = 3,
                                        psp_out_channels = 512,
                                        psp_use_batchnorm = True,
                                        psp_dropout = 0.2,
                                        in_channels = 1,
                                        classes = 1,)
            self.segmenter.load_state_dict(torch.load("./weights/segmentation/pspnet.pth",
                                                      map_location=torch.device("cpu")))
        self.segmenter.eval()
        if self.out_image is not None:
            if self.show_flag == 1:
                canvas = QPixmap(self.show_width, self.show_height)
                canvas.fill(QColor(112, 112, 112, 80))
                self.out_label.setPixmap(canvas)
                img_tensor = self.__classify_img_preprocess(self.in_image)
                reg_pred = torch.softmax(self.classifier(img_tensor), dim=-1)
                reg_prob = 100 * reg_pred.detach().cpu().numpy()[0]
                reg_index = np.argsort(reg_prob)[::-1]
                self.reg_pred = reg_index[0]
                self.Seg()

                text = label_list[reg_index[0]] + " : " + "{:.2f}%".format(reg_prob[reg_index[0]]) + "\n" + \
                       label_list[reg_index[1]] + " : " + "{:.2f}%".format(reg_prob[reg_index[1]]) + "\n" + \
                       label_list[reg_index[2]] + " : " + "{:.2f}%".format(reg_prob[reg_index[2]]) + "\n"

                self.state_label.setText("Segmentor:" + model_name + "\nImage path:" +
                                         self.image_path + '\nClassification results and probability:\n' + text)
            elif self.show_flag == 2:
                canvas = QPixmap(self.show_width, self.show_height)
                canvas.fill(QColor(112, 112, 112, 80))
                self.out_label.setPixmap(canvas)
                img_tensor = self.__classify_img_preprocess(self.in_image)
                reg_pred = torch.softmax(self.classifier(img_tensor), dim=-1)
                reg_prob = 100 * reg_pred.detach().cpu().numpy()[0]
                reg_index = np.argsort(reg_prob)[::-1]
                self.reg_pred = reg_index[0]
                self.CAM()

                text = label_list[reg_index[0]] + " : " + "{:.2f}%".format(reg_prob[reg_index[0]]) + "\n" + \
                       label_list[reg_index[1]] + " : " + "{:.2f}%".format(reg_prob[reg_index[1]]) + "\n" + \
                       label_list[reg_index[2]] + " : " + "{:.2f}%".format(reg_prob[reg_index[2]]) + "\n"

                self.state_label.setText("Segmentor:" + model_name + "\nImage path:" +
                                         self.image_path + '\nClassification results and probability:\n' + text)
        else:
            self.state_label.setText("Segmentor:" + model_name)


def main():
    app = QApplication(sys.argv)
    window = IUADOT_DEMO()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
