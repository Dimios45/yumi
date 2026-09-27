"""One A4 sheet with the required board and optional spare finger tags."""

from pathlib import Path

import cv2

source = Path("calibration/aruco_0_100mm.svg").read_text()
parts = [
    '<text x="15" y="188" font-family="sans-serif" font-size="4">Optional spare finger tags — black square 10 mm each</text>'
]
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for marker, x0 in [(13, 50), (14, 100)]:
    bits = cv2.aruco.generateImageMarker(dictionary, marker, 6)
    parts.append(
        f'<text x="{x0}" y="200" font-family="sans-serif" font-size="4">ID {marker}</text>'
    )
    for y in range(6):
        for x in range(6):
            if bits[y, x] == 0:
                parts.append(
                    f'<rect x="{x0 + x * 10 / 6:.8f}" y="{205 + y * 10 / 6:.8f}" width="{10 / 6:.8f}" height="{10 / 6:.8f}" fill="black"/>'
                )
parts += [
    '<text x="15" y="230" font-family="sans-serif" font-size="3.5">Keep a white border. Only one visible copy of each ID on the gripper.</text>',
    '<text x="15" y="239" font-family="sans-serif" font-size="3.5">Do not replace working finger tags unless needed; replacements require recalibration.</text>',
]
Path("calibration/PRINT_ALL_A4.svg").write_text(
    source.replace("</svg>", "\n".join(parts) + "\n</svg>")
)
print("calibration/PRINT_ALL_A4.svg")
