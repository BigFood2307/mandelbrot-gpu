from numba import cuda
import numpy as np
import math

@cuda.jit
def mandelbrot_values(c, limit, out):
    start = cuda.grid(1)
    step = cuda.gridsize(1)

    for idx in range(start, out.shape[0], step):
        i = 0
        z = 0
        while i < limit:
            i += 1
            z = z * z + c[idx]
            if (z.imag**2 + z.real**2) >= 4:
                out[idx] = i
                break
        if out[idx] == 0:
            out[idx] = -1

@cuda.jit
def grid_values(top_left, width, height, step, out):
    start_y, start_x = cuda.grid(2)
    step_y, step_x = cuda.gridsize(2)

    for i in range(start_y, height, step_y):
        for k in range(start_x, width, step_x):
            out[i * width + k] = (top_left + k * step - 1j * i * step)

@cuda.jit
def post_mandelbrot(m, min_it, max_it, color_range, out):
    expo = 0.25
    start = cuda.grid(1)
    step = cuda.gridsize(1)

    for idx in range(start, out.shape[0], step):
        if m[idx] >= 0:
            cuda.atomic.max(max_it, 0, m[idx])
            cuda.atomic.min(min_it, 0, m[idx])

    cuda.syncthreads()

    adj_min = math.log(np.float64(min_it[0]))
    dif = math.log(np.float64(max_it[0])) - adj_min
    step_per_color = dif/color_range.shape[0]

    for idx in range(start, out.shape[0], step):
        if m[idx] == -1:
            out[idx] = (0, 0, 0)
        else:
            dif_int = math.log(np.float64(m[idx])) - adj_min
            col_idx = int(dif_int//step_per_color)
            out[idx, 0] = color_range[col_idx, 0]
            out[idx, 1] = color_range[col_idx, 1]
            out[idx, 2] = color_range[col_idx, 2]


@cuda.jit
def mandelbrot_aio(top_left, width, height, step, limit, out):
    start_y, start_x = cuda.grid(2)
    step_y, step_x = cuda.gridsize(2)

    for n in range(start_y, height, step_y):
        for k in range(start_x, width, step_x):
            c = (top_left + k * step + 1j * n * step)
            i = 0
            z = 0
            m = 0
            while i < limit:
                i += 1
                z = z * z + c
                if abs(z) >= 2:
                    m = i
                    break
            if m == 0:
                m = -1
            idx = n * width + k
            out[idx] = 255
            if m == -1: out[idx] = 0


def mandelbrot_mx_create_aio(width, height, center=-1, min_x=-3, max_x=1, limit=200, zoom=-1, blocks = 16):
    threads_pb_sqrt = 16
    blocks_sqrt = np.sqrt(blocks).astype(np.uint32)

    if zoom == -1:
        zoom = width/(max_x-min_x)
    step = 1/zoom

    top_left = center - width/(zoom*2) +0.5*step + 1j*height/(zoom*2) - 0.5j*step

    mandel_post_d = cuda.device_array(width*height, dtype=np.uint8)
    mandelbrot_aio[(blocks_sqrt, blocks_sqrt), (threads_pb_sqrt, threads_pb_sqrt)](top_left, width, height, step, limit, mandel_post_d)

    mandel_post = np.empty(mandel_post_d.shape, dtype=np.uint8)

    mandel_post_d.copy_to_host(mandel_post)
    mandel_plane = mandel_post.reshape((height, width))
    return mandel_plane

def mandelbrot_mx_create(width, height, center=-1, min_x=-3, max_x=1, limit=200, zoom=-1, color_range=[[255, 255, 255]], blocks = 32):
    threads_pb_sqrt = 16
    threads_pb = threads_pb_sqrt**2
    blocks_sqrt = np.sqrt(blocks).astype(np.uint32)
    color_range = np.uint8(color_range)

    if zoom == -1:
        zoom = width/(max_x-min_x)
    step = 1/zoom

    top_left = center - width/(zoom*2) +0.5*step + 1j*height/(zoom*2) - 0.5j*step

    c_list_d = cuda.device_array(width*height, dtype=np.complex128)

    grid_values[(blocks_sqrt, blocks_sqrt), (threads_pb_sqrt, threads_pb_sqrt)](top_left, width, height, step, c_list_d)
    mandel_list_d = cuda.device_array(c_list_d.shape, dtype=np.int32)

    mandelbrot_values[blocks, threads_pb](c_list_d, limit, mandel_list_d)

    mandel_post_d = cuda.device_array((width*height, 3), dtype=np.uint8)
    max_it_d = cuda.to_device(np.uint32(0))
    min_it_d = cuda.to_device(np.uint32(limit))

    post_mandelbrot[blocks, threads_pb](mandel_list_d, min_it_d, max_it_d, color_range, mandel_post_d)

    mandel_post = np.empty(mandel_post_d.shape, dtype=np.uint8)

    mandel_post_d.copy_to_host(mandel_post)
    mandel_plane = mandel_post.reshape((height, width, 3))
    return mandel_plane