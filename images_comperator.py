import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob
import os.path
from threading import Thread
import matplotlib as mpl
import numpy as np
mpl.rcParams['keymap.back'].remove('left')
mpl.rcParams['keymap.forward'].remove('right')


def crop_center(img,cropy,cropx):
    y, x = img.shape[:2]
    sx = x//2-(cropx//2)
    sy = y//2-(cropy//2)
    return img[sy:sy+cropy, sx:sx+cropx]

class MultiViewer:
    def __init__(self, list_of_folders, factors):
        n_views = len(list_of_folders)
        self.factors = factors
        cols = 2
        if n_views > 4:
            cols = 3
        rows = int((n_views / cols) + 0.5)
        self.fig, axis = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(8,8))
        self.axis = axis.reshape(-1)[:len(list_of_folders)]
        for ax, folder in zip(self.axis, ['input'] + list_of_folders[1:]):
            ax.set_title(os.path.basename(folder))
        self.next_images = []
        self.im_name = ''
        self.generator = self.images_generator(list_of_folders)
        plt.gcf().canvas.mpl_connect("key_press_event", self.on_key_press)
        self.load_thread = Thread(target=self.get_next)
        self.load_thread.start()
        self.draw()
        plt.show()

    def on_key_press(self, event: mpl.backend_bases.KeyEvent):
        if event.key == "right":
            self.draw()

    def draw(self):
        self.load_thread.join()
        print('drawing', self.im_name)
        if len(self.next_images):
            images = self.next_images.copy()
            self.next_images.clear()
            for ax, im in zip(self.axis, images):
                ax.imshow(im)
            self.fig.suptitle(self.im_name)
            self.fig.canvas.draw()
            self.load_thread = Thread(target=self.generator.__next__)
            self.load_thread.start()

    def get_next(self):
        try:
            self.generator.__next__()
        except StopIteration:
            pass

    def images_generator(self, list_of_folders):
        # pre process
        f_map = {}
        for folder in list_of_folders[1:]:
            f_map[folder] = [os.path.basename(im_path) for im_path in glob(os.path.join(folder, '*.png'))]
        for im_path in glob(os.path.join(list_of_folders[0], '*.png')):
            print('getting', im_path)
            im_name = os.path.basename(im_path)
            if all(im_name in f_map[folder] for folder in list_of_folders[1:]):
                images = []
                for ind, folder in enumerate(list_of_folders):
                    im = plt.imread(os.path.join(folder, im_name))
                    im = crop_center(im, 2048, 2048)
                    if ind == 0 and self.factors is not None:
                        im = im * self.factors
                    images.append(im)
                self.next_images = images
                self.im_name = im_name
                yield 1
        self.next_images = []

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('root_folder')
    parser.add_argument('models', nargs='+')
    parser.add_argument('-f', '--factors', nargs='+', type=float, default=None)
    args = parser.parse_args()
    viewer = MultiViewer(args.folders, args.factors)




