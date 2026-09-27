"""Create a printable vector ArUco target with explicit physical dimensions."""

import argparse
from pathlib import Path

import cv2

p = argparse.ArgumentParser()
p.add_argument("--output", default="calibration/aruco_0_100mm.svg")
p.add_argument("--id", type=int, default=0)
p.add_argument("--size-mm", type=float, default=100)
a = p.parse_args()
if not 0 < a.size_mm <= 170:
    p.error("Size must fit A4 with white borders (0..170 mm)")
bits = cv2.aruco.generateImageMarker(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), a.id, 6
)
s = a.size_mm / 6
x0 = (210 - a.size_mm) / 2
y0 = 60
parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
    '<rect width="210" height="297" fill="white"/>',
    f'<text x="15" y="20" font-family="sans-serif" font-size="6">DICT_4X4_50 — ID {a.id}</text>',
    f'<text x="15" y="31" font-family="sans-serif" font-size="4">Black square: {a.size_mm:g} mm. Print at 100% / actual size.</text>',
    '<text x="15" y="39" font-family="sans-serif" font-size="4">Measure the printed black square before calibration.</text>',
]
for y in range(6):
    for x in range(6):
        if bits[y, x] == 0:
            parts.append(
                f'<rect x="{x0 + x * s:.8f}" y="{y0 + y * s:.8f}" width="{s:.8f}" height="{s:.8f}" fill="black"/>'
            )
parts += [
    '<path d="M 55 260 H 155 M 55 255 V 265 M 155 255 V 265" fill="none" stroke="black" stroke-width="0.3"/>',
    '<text x="70" y="275" font-family="sans-serif" font-size="4">Reference line: 100 mm</text>',
    "</svg>",
]
path = Path(a.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(parts))
print(path)
