import numpy as np


# Ref https://github.com/lucasrla/wsi-tile-cleanup/blob/master/wsi_tile_cleanup/filters/pens.py
def pen_percent(tile, pen_color_palette, pen_color):
    r, g, b, _ = np.rollaxis(tile, -1)
    thresholds = pen_color_palette[pen_color]

    if pen_color == "red":
        t = thresholds[0]
        mask = (r > t[0]) & (g < t[1]) & (b < t[2])

        for t in thresholds[1:]:
            mask = mask | ((r > t[0]) & (g < t[1]) & (b < t[2]))

    elif pen_color == "green":
        t = thresholds[0]
        mask = (r < t[0]) & (g > t[1]) & (b > t[2])

        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g > t[1]) & (b > t[2])

    elif pen_color == "blue":
        t = thresholds[0]
        mask = (r < t[0]) & (g < t[1]) & (b > t[2])

        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g < t[1]) & (b > t[2])

    else:
        raise Exception(f"Error: pen_color='{pen_color}' not supported")

    percentage = mask.avg() / 255.0

    return percentage


def get_pen_color_palette():
    return {
    "red": [
        (150, 80, 90),
        (110, 20, 30),
        (185, 65, 105),
        (195, 85, 125),
        (220, 115, 145),
        (125, 40, 70),
        (200, 120, 150),
        (100, 50, 65),
        (85, 25, 45),
    ],
    "green": [
        (150, 160, 140),
        (70, 110, 110),
        (45, 115, 100),
        (30, 75, 60),
        (195, 220, 210),
        (225, 230, 225),
        (170, 210, 200),
        (20, 30, 20),
        (50, 60, 40),
        (30, 50, 35),
        (65, 70, 60),
        (100, 110, 105),
        (165, 180, 180),
        (140, 140, 150),
        (185, 195, 195),
    ],
    "blue": [
        (60, 120, 190),
        (120, 170, 200),
        (120, 170, 200),
        (175, 210, 230),
        (145, 210, 210),
        (37, 95, 160),
        (30, 65, 130),
        (130, 155, 180),
        (40, 35, 85),
        (30, 20, 65),
        (90, 90, 140),
        (60, 60, 120),
        (110, 110, 175),
    ],
}