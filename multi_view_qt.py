import collections
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List, OrderedDict
import colour_demosaicing
from dataclasses import dataclass
import cv2
import numpy as np
import pyqtgraph as pg
import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QVBoxLayout, \
    QHBoxLayout, QGroupBox, QLabel, QComboBox, QInputDialog, QFileDialog, QLineEdit, \
    QRadioButton, QButtonGroup, QScrollArea


@dataclass
class ImageTypeCfg:
    path: str
    pattern: str
    action: str
    pivot: bool = False


def crop_center(img, cropy, cropx):
    y, x = img.shape[:2]
    cropy, cropx = min(cropy, img.shape[0]), min(cropx, img.shape[1])

    sx = x//2-(cropx//2)
    sx -= sx % 2
    sy = y//2-(cropy//2)
    sy -= sy % 2
    return img[sy:sy+cropy, sx:sx+cropx]


def auto_gamma_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = np.log(mid * 255) / np.log(mean)
    # print(gamma)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    corrected = cv2.LUT(img, table)
    return cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)


def auto_gamma_gray(img):
    max_val = 2**16 - 1 if img.dtype == np.uint16 else 2**8 - 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mid = 0.45
    mean = np.mean(gray)
    gamma = np.log(mid * max_val) / np.log(mean)
    corrected = np.power(img, gamma).clip(0, max_val).astype(img.dtype)
    return corrected


def comb_view(name) -> (pg.ImageItem, pg.ViewBox, QGroupBox):
    vb = pg.ViewBox(lockAspect=True)
    gv = pg.GraphicsView(useOpenGL=False)
    gv.setCentralItem(vb)
    it = pg.ImageItem()
    it.setImage()
    vb.addItem(it)
    cam_group = QGroupBox(name)
    cam_layout = QVBoxLayout()
    cam_layout.addWidget(gv)
    cam_group.setLayout(cam_layout)
    return it, vb, cam_group


class ImageType:
    def __init__(self, name: str, radio_group: QButtonGroup, ext_layout, cfg: ImageTypeCfg):
        self.name = name
        self.src_dir = Path(cfg.path)
        self.group = QGroupBox(title=name)
        layout = QVBoxLayout()
        self.in_path = QLabel(self.src_dir.name)
        layout.addWidget(self.in_path)
        act_layout = QHBoxLayout()
        layout.addLayout(act_layout)
        self.group.setLayout(layout)
        self.pattern_line = QLineEdit(cfg.pattern)
        act_layout.addWidget(self.pattern_line)
        self.action_select = QComboBox()
        self.action_select.addItems(['rgb', 'rgb_gamma', 'mono', 'demosaic', 'mosaic', 'bgr', 'rggb_gamma', 'bgr_gamma'])
        act_layout.addWidget(self.action_select)
        self.radio_btn = QRadioButton('pivot')
        self.radio_btn.setChecked(cfg.pivot)
        act_layout.addWidget(self.radio_btn)
        radio_group.addButton(self.radio_btn)
        index = self.action_select.findText(cfg.action, Qt.MatchFixedString)
        if index >= 0:
            self.action_select.setCurrentIndex(index)
        else:
            self.action_select.addItem(cfg.action)
            index = self.action_select.findText(cfg.action, Qt.MatchFixedString)
            self.action_select.setCurrentIndex(index)
        ext_layout.addWidget(self.group)
        self.it, self.vb, self.view_group = comb_view(self.name)
        self.files: List[Path] = []
        self.convert = None

    def is_pivot(self) -> bool: return self.radio_btn.isChecked()

    def start(self):
        if self.src_dir.as_posix() == 'mock':
            self.files = None
        else:
            self.files = list(self.src_dir.glob(self.pattern_line.text()))
            # print(f'{self.pattern_line.text()} {len(self.files)}')
        action = self.action_select.currentText()
        self.convert = []
        if 'mono' in action:
            self.convert.append(lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        if 'rgb' in action:
            self.convert.append(lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if 'bgr' in action:
            self.convert.append(lambda im: im)
        if 'rggb' in action:
            # self.convert.append(lambda im: cv2.demosaicing(im, cv2.COLOR_BayerRG2RGB))
            self.convert.append(lambda im: cv2.demosaicing(im, cv2.COLOR_BAYER_BG2RGB))
        if 'gamma' in action:
            self.convert.append(auto_gamma_gray)
        if 'crop' in action:
            self.convert.append(lambda im: crop_center(im, 2048, 2048))

    def display_image(self, im_base_name) -> bool:
        if self.files is None:
            # mock mode
            self.it.setImage(np.zeros((500, 500, 3), dtype=np.uint8))
            self.vb.autoRange()
            return True
        for f in self.files:
            if im_base_name in f.stem:
            # if im_base_name == f.stem.split('_')[0]:
                im = cv2.imread(f.as_posix(), cv2.IMREAD_UNCHANGED)
                for conversion in self.convert:
                    im = conversion(im)
                if im.dtype == np.uint16:
                    im = im // 256
                    im = im.astype(np.uint8)
                self.it.setImage(np.rot90(im, 3), autoLevels=False, levelSamples=255)
                self.vb.autoRange()
                return True
        return False

    def exist(self, base_name):
        for f in self.files:
            if base_name in f.stem:
                return True
        return False


class MainWin(QWidget):
    def __init__(self, cfg_path):
        super().__init__()
        self.multi_view = QWidget(parent=self)
        self.multi_view.setMinimumWidth(500)
        self.multi_view.setMinimumHeight(500)
        self.multi_layout = QGridLayout()
        self.multi_view.setLayout(self.multi_layout)

        with cfg_path.open('r') as f:
            cfg = yaml.safe_load(f)
            self.rows = cfg['rows']
            self.input_cfg = cfg['inputs']
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)
        self.main_layout.addWidget(self.multi_view)
        # splitter = QSplitter()
        side_widget = QScrollArea()
        side_widget.setMaximumWidth(300)
        side_layout = QVBoxLayout()
        side_widget.setLayout(side_layout)
        side_layout.maximumSize()
        side_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(side_widget)
        btn_layout = QHBoxLayout()
        side_layout.addLayout(btn_layout)

        next_btn = QPushButton('next')
        # next_btn.setMaximumWidth(50)
        next_btn.setShortcut(QKeySequence(Qt.Key_Space))
        btn_layout.addWidget(next_btn)
        next_btn.clicked.connect(self.step_next)
        btn_layout.addSpacing(10)

        add_btn = QPushButton('+')
        # add_btn.setMaximumWidth(30)
        add_btn.clicked.connect(self.add_input)
        btn_layout.addWidget(add_btn)
        btn_layout.addSpacing(10)

        dec_btn = QPushButton('-')
        # dec_btn.setMaximumWidth(30)
        dec_btn.clicked.connect(self.dec_input)
        btn_layout.addWidget(dec_btn)
        btn_layout.addSpacing(10)

        init_btn = QPushButton('set')
        # init_btn.setMaximumWidth(50)
        init_btn.clicked.connect(self.init_input)
        btn_layout.addWidget(init_btn)

        self.inputs: OrderedDict[str, ImageType] = collections.OrderedDict()
        self.radio_group = QButtonGroup(parent=self)
        for im_name, im_params in self.input_cfg.items():
            cfg = ImageTypeCfg(**im_params)
            self.inputs[im_name] = ImageType(im_name, self.radio_group, side_layout, cfg)
            if len(self.inputs) == 1:
                list(self.inputs.values())[0].radio_btn.setChecked(True)
        self.pivot: ImageType = None
        self.im_gen = None
        self.init_input()
        self.show()

    def add_input(self):
        if len(self.inputs) > 0:
            last_path = list(self.inputs.values())[-1].in_path.text()
        else:
            last_path = None
        in_dir = Path(QFileDialog.getExistingDirectory(self, "Select Directory", last_path))
        im_name, pressed = QInputDialog.getText(self, 'input name', 'input name:', QLineEdit.Normal, in_dir.stem)
        if pressed and im_name != '':
            self.inputs[im_name] = ImageType(name=im_name, src_dir=in_dir.as_posix(), radio_group=self.radio_group,
                                             ext_layout=self.side_layout, action=None)
        self.input_cfg[im_name] = {'path': in_dir.as_posix(), 'pattern': '*.png', 'action': 'rgb'}

    def dec_input(self):
        if len(self.inputs) > 0:
            dec_name = next(reversed(self.inputs))
            self.multi_layout.removeWidget(self.inputs[dec_name].group)
            del self.inputs[dec_name]
            self.inputs[next(reversed(self.inputs))]
            self.init_side_layout()

    def init_input(self):
        cols = (len(self.inputs) // self.rows) + len(self.inputs) % self.rows
        in_iter = iter(self.inputs.values())
        for r in range(self.rows):
            for c in range(cols):
                try:
                    im_type: ImageType = next(in_iter)
                except StopIteration:
                    break
                self.multi_layout.addWidget(im_type.view_group, r, c)
                if im_type.is_pivot():
                    self.pivot = im_type
                    self.im_gen = self.pivot_iterator()
                im_type.start()
        for im_type in self.inputs.values():
            if not im_type.is_pivot():
                im_type.vb.setXLink(self.pivot.vb)
                im_type.vb.setYLink(self.pivot.vb)
            if im_type.files is not None and len(im_type.files) == 0:
                print(f'ERROR - files list is empty - path {im_type.src_dir} and pattern {im_type.pattern_line.text()}')
        self.step_next()

    def pivot_iterator(self):
        for im in self.pivot.files:
            suffix = self.pivot.pattern_line.text().replace('*', '')
            base_name = im.name.replace(suffix, '')
            # base_name = im.name.replace('.png', '')
            yield base_name

    def step_next(self):
        non_pivots = [i for i in self.inputs.values() if i is not self.pivot and i.files is not None]
        try:
            base_name = next(self.im_gen)
        except StopIteration:
            return
        while not all([it.exist(base_name) for it in non_pivots]):
            try:
                base_name = next(self.im_gen)
            except StopIteration:
                return
        for im_type in self.inputs.values():
            if not im_type.display_image(base_name):
                print(f'failed to find image in {im_type.name}')

    def closeEvent(self, event):
        out_cfg = {}
        for in_name, in_group in self.inputs.items():
            out_cfg[in_name] = {'path': in_group.in_path.text(),
                                'pattern': in_group.pattern_line.text(),
                                'action': in_group.action_select.currentText()}
        # with self.cfg_path.open('w') as f:
        #     yaml.dump(out_cfg, f, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg_file', help='data set parameters file path')
    args = parser.parse_args()

    app = QApplication([])
    cfg_path = Path(args.cfg_file)
    assert cfg_path.exists(), f'failed to find input file {args.cfg_file} '
    gui = MainWin(cfg_path)
    app.exec_()
