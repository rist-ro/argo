import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .AbstractPlot import AbstractPlot

from argo.core.utils.argo_utils import create_list_colors

import numpy as np

import matplotlib.image as mpimg


class PlotImages(AbstractPlot):

    def __init__(self):
        pass

    def suptitle(self, fig, title):
        pass

    def create_fig(self, figsize, folders_cartesian_fixed):
        w, h = figsize
        size = (w, h*len(folders_cartesian_fixed))
        fig = plt.figure(figsize=size)
        return fig
            
    def rows_columns(self, panels, folders_cartesian_fixed):
        n_columns = len(panels)
        n_rows = len(folders_cartesian_fixed)
        return n_rows, n_columns

    def plot_vertical_panel(self,
                            vertical_panels,
                            c,
                            n_rows,
                            n_columns,
                            legend,
                            fig,
                            folders_cartesian_fixed,
                            group_by_values,
                            source,
                            degree):

        # only one vertical panel allowed
        assert(len(vertical_panels)==1)
        panel = vertical_panels[0]
        assert(len(panel)==1)
        panel = panel[0]
        
        r = 0

        #for panel in vertical_panels:
        #for curve in panel:
        for marker, color, directory in self.get_colors_pairs(sorted(folders_cartesian_fixed),
                                                              group_by_values):

            try:
                
                if len(group_by_values) == 0:
                    max_colors = len(folders_cartesian_fixed)
                else:
                    max_colors = len(list(group_by_values.values())[0])

                if max_colors == 0:
                    max_colors = 1
                list_colors = create_list_colors(max_colors)
                
                #pdb.set_trace()
                
                # file_path = source  + "/" + directory + "/" + curve["filename"]
                file_path = directory + "/" + panel["filename"]

                #label = self.create_label(curve, directory, legend, source)
                #self.create_plot(marker, color, curve, ax, degree, label, list_colors, x, y)
                img = mpimg.imread(file_path)

                width = 77
                height = 77
                margin_x = 125
                margin_y = 72

                num_cols_images =  6 # height
                num_rows_images =  4 # width
                portion_imgs = [] #[None] * num_cols * num_rows
                for s in range(num_rows_images):
                    for t in range(num_cols_images):
                        im = img[(margin_y+t*height):(margin_y+(t+1)*height),(margin_x+s*width):(margin_x+(s+1)*width)]
                        portion_imgs.append(im)

                new_img = np.concatenate(portion_imgs, axis=1)
                
                #n_plots = len(panels)
                ax = fig.add_subplot(n_rows, n_columns, r * n_columns + c + 1)
                ax.imshow(new_img)

                #if r==0:
                #    ax.set_title(directory.split("/")[-1], loc='left', pad=20)
                #else:
                ax.set_title(directory.split("/")[-1], loc='left', pad=0)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                
                r += 1
            
            except FileNotFoundError as e:
                print("FileNotFoundError cannot read file " + file_path + " " + str(e))
            

            '''
            # Options for legend_loc
            # 'best', 'upper right', 'upper left', 'lower left', 'lower right',
            # 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
            loc = panel.get("legend_loc", "upper right")
            bbox_to_anchor = curve.get("bbox_to_anchor", None)
            if bbox_to_anchor is None:
                lgd = ax.legend(loc=loc)
            else:
                lgd = ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
            '''
            


        

