from numba import cuda
import numpy as np
import time
from PIL import Image as im

from numba import vectorize
import numpy as np
from PIL import Image as im
import time
from mandelbrot_gpu_base import mandelbrot_mx_create, mandelbrot_mx_create_aio
from colour import Color
import math

red = Color("red")
blue = Color("Blue")
color_range = list(red.range_to(blue, 50))
fake_range = np.zeros((len(color_range), 3))

for i, col in enumerate(color_range):
    fake_range[i] = [col.get_red()*255, col.get_green()*255, col.get_blue()*255]

width = 1920
height = 1080
zoom = 270
p_step = 1/zoom
zoom_center = (-0.6872041226014051+0.29258340765161583j)
zoom_center_x = int((width/2) + zoom_center.real/p_step)
zoom_center_y = int((height/2) - zoom_center.imag/p_step)
print(zoom_center_x, zoom_center_y)
center = -1
limit = limit = int((math.log(zoom/270, 2)+1)*120)
max_zoom = 1932082537176566
i = 0
startTimeExt = time.time()
while zoom < max_zoom:
    startTimeInt = time.time()
    print(i, ":", zoom)

    plane = mandelbrot_mx_create(width, height, center=center, limit=limit, zoom=zoom, color_range=fake_range)
    data = im.fromarray(plane)
    data.save("Data/mandelbrot_col2/mb_" + "{:06d}".format(i) + ".png")

    zoom *= 1.015
    p_step = 1 / zoom
    limit = limit = int((math.log(zoom / 270, 1.5) + 1) * 100)

    top_left = center - width / (zoom * 2) + 0.5 * p_step + 1j * height / (zoom * 2) - 0.5j * p_step
    c_at_zc = top_left + p_step*zoom_center_x - 1j*p_step*zoom_center_y

    center -= c_at_zc - zoom_center

    i += 1
    print("Took", time.time()-startTimeInt, "s")
print(i, "images generated")
print("Took", time.time()-startTimeExt, "s")
