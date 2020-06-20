# plot.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) 

import time

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D

class SurfacePlot(object):

    def __init__(self, plot_ax_id=111):
        self.fig = matplotlib.figure.Figure()
        self.surf = None
        self.plot_ax_id = plot_ax_id

    def init(self):
        matplotlib.pyplot.ion()
        if self.surf is not None:
            self.surf.remove()
        self.ax = self.fig.add_subplot(self.plot_ax_id, projection='3d')
        self.surf = None
        
    def draw(self, value, title=None, sleep_time=0):
        rows, cols = value.shape
        x = [[i for i in range(cols)] for _ in range(rows)]
        y = [[i for _ in range(cols)] for i in range(rows)]
        if self.surf is not None:
            self.surf.remove()
        self.surf = self.ax.plot_surface(x, y, value, cmap=matplotlib.cm.RdYlGn)
        if title is not None:
            self.ax.set_title(title, y=1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(sleep_time)
    
    def clear(self):
        if self.surf is not None:
            self.surf.remove()
            self.surf = None


def build_matplotlib_canvas(figure, canvas_master, toolbar_master=None):
    plot_canvas = FigureCanvasTkAgg(figure, master=canvas_master)
    plot_toolbar = None if toolbar_master is None else NavigationToolbar2Tk(plot_canvas, toolbar_master)
    plot_canvas.mpl_connect("key_press_event",
        lambda e: matplotlib.backend_bases.key_press_handler(e, plot_canvas, plot_toolbar)
    )
    return plot_canvas, plot_toolbar
