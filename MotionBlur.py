import math

from numba import njit
import numpy as np

@njit
def get_line(start, end):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    is_steep = abs(dy) > abs(dx)
 
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    dx = x2 - x1
    dy = y2 - y1
 
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (int(y), int(x)) if is_steep else (int(x), int(y))
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    if swapped:
        points.reverse()
    return points



@njit
def EasyIn(t):
   return math.pow(t, 2)

@njit
def EasyOut(t):
     return (t * (1 - t))
     #return math.pow(1 - t, 2)

@njit
def EasyInOut(t):
    if t < 0.5:
        return 2*t*t
    else:
        return -1+(4-2*t)*t


@njit
def Linear(t):
     return t

@njit
def Dist(x0, y0, x1, y1):
  return ( (x1 - x0)**2 + (y1 - y0)**2 )


@njit
def BlurIt(image, vector, smooth, type = 0, strength = 1):
    w = image.shape[0]
    h = image.shape[1]

    im = np.empty((w, h, 4))
    im.fill(0)

    method = None

    '''
    for y in range(0, h):
        for x in range(0, w):
            xx, yy = vector[x,y]
            
            xx = math.floor(xx * smooth)
            yy = math.floor(yy * smooth)
            
            color = image[x,y]
            
            dist = max(1, Dist(x, y, x+xx, y+yy))
            points = get_line((x, y), (x+xx, y+yy))

            im[x, y, :3] += color
            im[x, y, 3] += 1

            if xx == 0 and yy == 0:
                continue

            for p in points:
            if p[0] == x and p[1] == y:
                continue
            if p[0] < 0 or p[1] < 0 or p[0] >= w - 1 or p[1] >= h - 1:
                continue
            zX, zY = p
            if type == 0:
                force =   EasyIn(1- (Dist(x, y,zX, zY) / dist))
            elif type == 1:
                force =   EasyOut(1- (Dist(x, y,zX, zY) / dist))
            elif type == 2:
                force =   Linear(1- (Dist(x, y,zX, zY) / dist))
            if(force > 1 or force < 0):
                print(force)
                
            #im[zX, zY, :3] += color * force
            #im[zX, zY, 3] += force
            im[zX, zY, :3] += (color * force) + (image[zX,zY] * (1 - force))
            im[zX, zY, 3] += 1

    im[:,:, 0] /= im[:,:, 3]
    im[:,:, 1] /= im[:,:, 3]
    im[:,:, 2] /= im[:,:, 3]
    im = im[:,:, :3]
    return im

    '''


    for y in range(0, h):
        for x in range(0, w):
            xx, yy = vector[x,y]
            
            xx = math.floor(xx * smooth)
            yy = math.floor(yy * smooth)
            
            color = image[x,y]
            
            dist = max(1, Dist(x, y, x+xx, y+yy))
            points = get_line((x, y), (x+xx, y+yy))

            #im[x, y, :3] += color
            #im[x, y, 3] += 1

            if xx == 0 and yy == 0:
                continue

            for p in points:
                if p[0] == x and p[1] == y:
                    continue
                if p[0] < 0 or p[1] < 0 or p[0] >= w - 1 or p[1] >= h - 1:
                    continue
                zX, zY = p
                if type == 0:
                    force =   EasyIn(1- (Dist(x, y,zX, zY) / dist))
                elif type == 1:
                    force =   EasyInOut(1- (Dist(x, y,zX, zY) / dist))
                elif type == 2:
                    force =   Linear(1- (Dist(x, y,zX, zY) / dist))
                if(force > 1 or force < 0):
                    print(force)

                f = min(1, force * strength)
                    
                #im[zX, zY, :3] += color * force
                #im[zX, zY, 3] += 1
                im[zX, zY, :3] += (color * f) + (image[zX,zY] * (1 - f))
                im[zX, zY, 3] += 1
    blurIm = im[:,:, :3]
    return im
    return blurIm, im[:,:, 3]
    #image[:, :, 0] = (image[:, :, 0] + blurIm[:, :, 0]) / (im[:,:, 3] + 1)
    #image[:, :, 1] = (image[:, :, 1] + blurIm[:, :, 1]) / (im[:,:, 3] + 1)
    #image[:, :, 2] = (image[:, :, 2] + blurIm[:, :, 2]) / (im[:,:, 3] + 1)
    #return image