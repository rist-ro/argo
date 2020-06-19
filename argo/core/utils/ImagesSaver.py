#from abc import ABCMeta, abstractmethod

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pdb

import numpy as np
#from .Logger import Logger

class ImagesSaver():

    def __init__(self, dirName, pm=True):
        #super(ImagesGenerator, self).__init__(dirName, fileName, period_log, verbose)

        #self._n_images_columns = n_images_columns
        #self._n_images_rows = n_images_rows
        #self._color = 0

        self._dirName = dirName

        self._image_width = -1
        self._image_height = -1
        self._channels = -1
        self._pm = pm
        if self._pm:
            self._vmin = -1
        else:
            self._vmin = 0

    '''
    def set_color(self, color):
        self._color = color
    '''

    @property
    def image_shape(self):
        return (self._image_width, self._image_height, self._channels)

    @image_shape.setter
    def image_shape(self, shape):
        self._image_width = shape[0]
        self._image_height = shape[1]
        self._channel = shape[2]

    def save_images(self, panel, fileName, title="", fontsize=9):
        '''
        if width is None or height is None:
            width = self._n_images_columns
            height = self._n_images_rows
        else:
            assert(width>1)
            assert(height>1)
            width = int(width)
            height = int(height)

        if self._image_width<0 or self._image_height<0:
            raise Exception("Image shape not set")
        '''

        height = len(panel)
        width = len(panel[0])

        fig = plt.figure(figsize=(width, height))
        plt.axis('off')
        #gs1 = gridspec.GridSpec(width, height)
        #gs1.update(wspace=0.05, hspace=0.05)

        plt.title(title, fontsize=fontsize, loc='center')

        k = 0
        for i in range(len(panel)):
            for j in range(len(panel[i])):
                ax0 = fig.add_subplot(height, width, k+1)

                # maybe this can be merged to avoid the if, in case shapes are correct
                # to be checked
                if panel[0][0].shape[2] == 1:

                    #squeeze does not do anything if dimension 2 is not 1, compatible with RGB
                    img = np.squeeze(panel[i][j], axis=2)

                    #from the doc: cmap="gray" is ignored for RGB data
                    ax0.imshow(img, cmap="gray", vmin=self._vmin, vmax=1)

                elif panel[0][0].shape[2] == 3:
                    #im_r = panel[i][j][:,:,0]
                    #im_g = panel[i][j][:,:,1]
                    #im_b = panel[i][j][:,:,2]
                    #img = np.dstack((im_r, im_g, im_b))

                    # vmin and vmax, seems to be ignored in the RGB case, for safety,
                    # we rescale between 0 and 1 which is imshow default and let it deal with it
                    img = (panel[i][j]+1)/2

                    # I clip inside 0 and 1 in case the algorithm is not clipping the noise
                    img = np.clip(img, 0, 1)

                    ax0.imshow(img)
                    # img = panel[i][j]
                    # ax0.imshow(img, vmin=-1, vmax=1)

                else:
                    raise Exception("Expected number of channels is 1 or 3")

                #ax0.set_axis_off()
                #ax0.set_adjustable('box-forced')
                #ax0.set_xticks([])
                #ax0.set_xticks([])
                ax0.get_xaxis().set_visible(False)
                ax0.get_yaxis().set_visible(False)
                k = k + 1
            k = k + (len(panel[0]) - len(panel[i]))

        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
        # eft=0, bottom=0, right=1, top=1,
        plt.subplots_adjust(wspace=0, hspace=0)
        #plt.tight_layout() #pad=-12, w_pad=0, h_pad=0.5)  # h_pad=Y

        plt.savefig(self._dirName + '/' + fileName + '.png') # , bbox_inches='tight'
        plt.close()

    #def log_epoch(self, epoch):
    #    return epoch % self._period == 0

    '''
    @property
    def number_images_columns(self):
        return self._n_images_columns

    @property
    def number_images_rows(self):
        return self._n_images_rows

    def init(self):
        pass

    def release(self):
        pass
    '''
