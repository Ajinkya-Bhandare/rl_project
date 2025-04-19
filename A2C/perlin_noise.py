import numpy as np

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    return a + t * (b - a)

def grad(hash, x, y):
    h = hash & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else 0)
    return (u + v) * (1 if h & 1 == 0 else -1)

def perlin_noise(width, height, scale=10):
    # Generate random gradient vectors
    perm = np.random.permutation(256)
    perm = np.tile(perm, 2)

    noise = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            # Determine grid cell coordinates
            X = int(x // scale) & 255
            Y = int(y // scale) & 255

            # Relative position within the grid cell
            xf = (x / scale) - int(x / scale)
            yf = (y / scale) - int(y / scale)

            # Fade curves
            u = fade(xf)
            v = fade(yf)

            # Hash and gradient calculations
            a = perm[X] + Y
            b = perm[X + 1] + Y
            aa = perm[a]
            ab = perm[a + 1]
            ba = perm[b]
            bb = perm[b + 1]

            # Interpolate the noise values
            x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u)
            x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u)
            noise[y, x] = lerp(x1, x2, v)

    return noise