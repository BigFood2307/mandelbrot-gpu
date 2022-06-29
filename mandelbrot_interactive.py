from PIL import Image as im
from PIL import ImageTk
from tkinter import *
from mandelbrot_gpu_base import mandelbrot_mx_create
import math
import numpy as np
from colour import Color

class MandelApp:

    def __init__(self, master, limit_exp=1.5, limit_scale=75):

        self.limit_exp = limit_exp
        self.limit_scale = limit_scale
        self.center = -1
        #self.center = (-0.16262357752449902-1.0300235973032343j)
        self.width = 1080
        self.height = 720
        self.zoom = self.width/4
        #self.zoom = 1.4671751766684548e+16
        self.limit = 0
        self.set_limit()
        self.img = None

        red = Color("red")
        blue = Color("Blue")
        color_range = list(red.range_to(blue, 100))
        self.fake_range = np.zeros((len(color_range), 3))

        for i, col in enumerate(color_range):
            self.fake_range[i] = [col.get_red() * 255, col.get_green() * 255, col.get_blue() * 255]

        self.canvas = Canvas(master, width=self.width, height=self.height)
        self.canvas.pack(fill="both", expand=True)

        self.recompute()

        self.canvas.bind("<Button-1>", self.recenter)
        self.canvas.bind("<Configure>", self.resize)
        self.canvas.bind("<Button-2>", self.zoom_in)
        self.canvas.bind("<Button-3>", self.zoom_out)
        self.canvas.bind("<Up>", self.limit_exp_up)
        self.canvas.bind("<Down>", self.limit_exp_down)

    def recompute(self):
        print("Recompute with limit", self.limit, "and zoom", self.zoom)
        plane = mandelbrot_mx_create(self.width, self.height, center=self.center, zoom=self.zoom, limit=self.limit, color_range=self.fake_range, blocks=32)
        self.img = ImageTk.PhotoImage(im.fromarray(plane))
        self.canvas.create_image(0, 0, anchor=NW, image=self.img)

    def recenter(self, event):
        print(event.x, event.y)
        step = 1/self.zoom
        dif_r = event.x - (self.width/2) + 0.5
        dif_i = - event.y + (self.height/2) - 0.5
        self.center = self.center + dif_r*step + 1j*dif_i*step
        print(self.center)
        self.recompute()

    def resize(self, event):
        self.width = event.width
        self.height = event.height
        self.recompute()

    def zoom_in(self, event):
        self.zoom = self.zoom*1.5
        self.set_limit()
        self.recompute()

    def zoom_out(self, event):
        self.zoom = self.zoom/1.5
        self.set_limit()
        self.recompute()

    def limit_exp_up(self, event):
        self.limit_exp = self.limit_exp*1.1
        self.set_limit()
        self.recompute()

    def limit_exp_down(self, event):
        self.limit_exp = self.limit_exp/1.1
        self.set_limit()
        self.recompute()

    def set_limit(self):
        self.limit = int((math.log(self.zoom/270, self.limit_exp)+1)*self.limit_scale)

root = Tk()

mapp = MandelApp(root, limit_scale=150)

root.mainloop()