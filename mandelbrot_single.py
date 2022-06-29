from numba import vectorize
import numpy as np
from PIL import Image as im
import time
from mandelbrot_gpu_base import mandelbrot_mx_create, mandelbrot_mx_create_aio
from colour import Color

red = Color("red")
blue = Color("Blue")
color_range = list(red.range_to(blue, 200))
fake_range = np.zeros((len(color_range), 3))

for i, col in enumerate(color_range):
    fake_range[i] = [col.get_red()*255, col.get_green()*255, col.get_blue()*255]

print(fake_range)

startTime = time.time()

plane = mandelbrot_mx_create(2560, 1080, center=(-0.5613979088698615+0.5574960478496346j), zoom=75386851016186.7, color_range=fake_range, limit=10000)
data = im.fromarray(plane)
endTime = time.time()
data.save('data/mandelbrotwide.png')

timeDiff = endTime - startTime
print(timeDiff)

print(timeDiff)
