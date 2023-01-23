import numpy as np


# Ref https://github.com/lucasrla/wsi-tile-cleanup/blob/master/wsi_tile_cleanup/filters/pens.py
def get_pen_mask(r, g, b, pen_color_palette, pen_color):
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

    return mask


def get_pen_color_palette():
    return {
    # more strict red filter to avoid filter tissue tiles.
    "red": [
        (165, 72, 81),
        (121, 18, 27),
        (203, 58, 94),
        (214, 76, 112),
        (242, 103, 130),
        (137, 36, 63),
        (220, 108, 135),
        (110, 45, 58),
        (93, 22, 40),
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
