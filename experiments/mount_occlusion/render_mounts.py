"""Render the two visibility-sweep rigs with a shared orthographic projection."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visibility_sweep import CAMERAS, CORNERS, EDGES, HALF_RIG, SIGNS

WIDTH, HEIGHT = 1320, 690
SUPERSAMPLE = 2
SCALE_PX_PER_M = 235
INK = '#172638'
MUTED = '#516579'
BLUE = '#246b9c'
AMBER = '#a34e08'
GREEN = '#2e775e'
OBSERVER = np.array([0.38, 0.28, 0.88])
OBSERVER /= np.linalg.norm(OBSERVER)
RIGHT = np.cross([0, 1, 0], OBSERVER)
RIGHT /= np.linalg.norm(RIGHT)
UP = np.cross(OBSERVER, RIGHT)
FONT_DIR = Path('C:/Windows/Fonts')


def font(size, bold=False):
    names = ('segoeuib.ttf', 'DejaVuSans-Bold.ttf') if bold else ('segoeui.ttf', 'DejaVuSans.ttf')
    for source in (FONT_DIR / names[0], names[1]):
        try:
            return ImageFont.truetype(str(source), size * SUPERSAMPLE)
        except OSError:
            continue
    return ImageFont.load_default(size=size * SUPERSAMPLE)


def project(points, origin):
    return np.asarray(points) @ np.stack((RIGHT, -UP)).T * SCALE_PX_PER_M + origin


def px(point):
    return tuple(round(coord * SUPERSAMPLE) for coord in point)


def dashed(draw, start, end, color):
    offset = end - start
    distance = np.linalg.norm(offset)
    for i in range(0, int(distance), 10):
        draw.line((px(start + i / distance * offset), px(start + min(i + 5, distance) / distance * offset)),
                  fill=color, width=SUPERSAMPLE)


def arrow(draw, start, end, color, width):
    draw.line((px(start), px(end)), fill=color, width=width * SUPERSAMPLE)
    direction = (end - start) / np.linalg.norm(end - start)
    head = end - 12 * direction
    side = 5 * np.array([-direction[1], direction[0]])
    draw.polygon((px(end), px(head + side), px(head - side)), fill=color)


def dot(draw, point, radius, color):
    draw.ellipse((px(point - radius), px(point + radius)), fill=color, outline='white', width=2 * SUPERSAMPLE)


def render_panel(draw, layout, origin):
    wire = project(SIGNS * HALF_RIG, origin)
    for a, b in EDGES:
        if not np.any((SIGNS[a] == SIGNS[b]) & (SIGNS[a] > 0)):
            dashed(draw, wire[a], wire[b], '#aec0cf')
    for a, b in EDGES:
        if np.any((SIGNS[a] == SIGNS[b]) & (SIGNS[a] > 0)):
            draw.line((px(wire[a]), px(wire[b])), fill='#9caebe', width=2 * SUPERSAMPLE)

    centers = CAMERAS[layout]
    for index, point in enumerate(centers):
        diagonal = index == 3
        color = AMBER if diagonal else BLUE
        start, end = project(np.array([point, point * (0.16 if diagonal else 0.23)]), origin)
        arrow(draw, start, end, color, 3 if diagonal else 2)

    box = project(CORNERS, origin)
    for axis, face_color in ((2, '#c1e3cf'), (0, '#9acbb3'), (1, '#dcf0e4')):
        face = np.flatnonzero(SIGNS[:, axis] == 1)
        draw.polygon([px(box[index]) for index in face[[0, 1, 3, 2]]],
                     fill=face_color, outline=GREEN, width=SUPERSAMPLE)
    draw.text(px(origin + [11, 18]), 'O', fill=GREEN, font=font(17, True), anchor='mt')

    labels = [('Side +x', (15, 4)), ('Top +y', (0, -14)), ('Front +z', (-55, 15)),
              ('Cam 4 (+,+,+)' if layout == 'A' else 'Cam 4 (−,−,−)', (11, -12) if layout == 'A' else (-20, 20))]
    for index, (point, (label, offset)) in enumerate(zip(centers, labels)):
        screen = project(point, origin)
        color = AMBER if index == 3 else BLUE
        dot(draw, screen, 9 if index == 3 else 7, color)
        anchor = ('mb' if index == 1 else 'mt') if index in (1, 2) or (index == 3 and layout == 'B') else 'lm'
        draw.text(px(screen + offset), label, fill=color, font=font(18, True), anchor=anchor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('camera_mounts.png'))
    args = parser.parse_args()
    image = Image.new('RGB', (WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE), '#f5f8fa')
    draw = ImageDraw.Draw(image)
    draw.text(px((40, 34)), 'Four-camera mounting geometry', fill=INK, font=font(34, True), anchor='lt')
    draw.text(px((41, 81)), '1.30 m cube · common orthographic scale · box 190 × 120 × 85 mm',
              fill=MUTED, font=font(19), anchor='lt')
    for layout, left, title in (('A', 40, 'A   Same-side diagonal'), ('B', 670, 'B   Opposite-side diagonal')):
        draw.rounded_rectangle((px((left, 122)), px((left + 610, 612))), radius=15 * SUPERSAMPLE,
                               fill='white', outline='#d8e4ed', width=SUPERSAMPLE)
        draw.text(px((left + 25, 143)), title, fill=INK, font=font(23, True), anchor='lt')
        render_panel(draw, layout, np.array([left + 300, 385]))
    dot(draw, np.array([47, 650]), 7, BLUE)
    draw.text(px((61, 650)), 'Axis camera center', fill=MUTED, font=font(16), anchor='lm')
    dot(draw, np.array([244, 650]), 8, AMBER)
    draw.text(px((259, 650)), 'Fourth camera center', fill=MUTED, font=font(16), anchor='lm')
    arrow(draw, np.array([459, 650]), np.array([499, 650]), BLUE, 2)
    draw.text(px((508, 650)), 'Optical direction to O', fill=MUTED, font=font(16), anchor='lm')
    image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(args.output, dpi=(160, 160))


if __name__ == '__main__':
    main()
