import cv2
from pathlib import Path
import numpy as np

BOARD_SIZE = (7, 7)
SQUARE_SIZE = 32.0

def get_object_points():

    objp = np.zeros(
        (BOARD_SIZE[0] * BOARD_SIZE[1], 3),
        dtype=np.float32
    )

    objp[:, :2] = np.mgrid[
        0:BOARD_SIZE[0],
        0:BOARD_SIZE[1]
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE

    return objp



input_dir = Path("./data")

obj_points = []
img_points = []

for image_path in input_dir.iterdir():

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Erro ao carregar imagem. {image_path}")
        continue

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray_image, BOARD_SIZE)

    if not found:
        print(f"Tabuleiro não encontrado em: {image_path}")
        continue

    # Refina cantos
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

    obj_points.append(get_object_points())
    img_points.append(corners_refined)

image_size = gray_image.shape[::-1]
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

print("Número de imagens:", len(img_points))
print("RMS:", rms)
print("K:\n", K)
print("Distorção:", dist)

width, height = image_size

new_k, roi = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 1, image_size)
undistorted = cv2.undistort(image, K, dist, None, new_k)

print("Coeficientes de Distorção: ", dist)

comparison = np.hstack((image, undistorted))

cv2.namedWindow("Original | Corrigida ", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original | Corrigida ", 1200, 600)

cv2.imshow("Original | Corrigida ", comparison)
cv2.waitKey(0)

print("image_size:", image_size)
print("Centro esperado:", image_size[0] / 2, image_size[1] / 2)
print("Centro estimado:", K[0, 2], K[1, 2])

canvas = np.zeros(
    (image_size[1], image_size[0], 3),
    dtype=np.uint8
)

for corners in img_points:
    for corner in corners:
        x, y = corner.ravel()

        cv2.circle(
            canvas,
            (int(x), int(y)),
            3,
            (255, 255, 255),
            -1
        )

cv2.namedWindow("Distribuicao dos pontos", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Distribuicao dos pontos", 600, 800)

cv2.imshow("Distribuicao dos pontos", canvas)
cv2.waitKey(0)
