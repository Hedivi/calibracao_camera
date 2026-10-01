import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
valid_images = []

for image_path in input_dir.iterdir():
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Erro ao carregar imagem. {image_path}")
        continue

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray_image, BOARD_SIZE, flags)

    if not found:
        print(f"Tabuleiro não encontrado em: {image_path}")
        continue

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

    cv2.drawChessboardCorners(image, BOARD_SIZE, corners_refined, found)

    cv2.namedWindow("A Processar Imagens", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("A Processar Imagens", 800, 600)
    cv2.imshow("A Processar Imagens", image)

    cv2.waitKey(0)

    obj_points.append(get_object_points())
    img_points.append(corners_refined)
    valid_images.append(str(image_path))

cv2.destroyAllWindows()

if not obj_points:
    exit()

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

img_projecao = cv2.imread(valid_images[0])
pontos_projetados, _ = cv2.projectPoints(obj_points[0], rvecs[0], tvecs[0], K, dist)

for p_real, p_proj in zip(img_points[0], pontos_projetados):
    x_real, y_real = p_real.ravel()
    x_proj, y_proj = p_proj.ravel()
    cv2.circle(img_projecao, (int(x_real), int(y_real)), 6, (0, 255, 0), -1)
    cv2.circle(img_projecao, (int(x_proj), int(y_proj)), 3, (0, 0, 255), -1)

cv2.imwrite("projecao_3d_2d.jpg", img_projecao)
print("Experimento de projecao 3D para 2D salvo como 'projecao_3d_2d.jpg'")

comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(comparison_rgb)
axes[0].set_title("Original | Corrigida")
axes[0].axis('off')

axes[1].imshow(canvas_rgb)
axes[1].set_title("Distribuicao dos pontos")
axes[1].axis('off')

plt.tight_layout()
plt.show()