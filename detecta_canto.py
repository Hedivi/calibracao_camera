import cv2
from pathlib import Path

BOARD_SIZE = (7, 7)

input_dir = Path("./data")

for image_path in input_dir.iterdir():

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Erro ao carregar imagem. {image_path}")
        continue

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # adicionado flags de otimização
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + CV2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray_image, BOARD_SIZE, flags)

    if found:
        cv2.drawChessboardCorners(image, BOARD_SIZE, corners, found)

        cv2.namedWindow("Cantos Encontrados", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cantos Encontrados", 800, 600)
    cv2.imshow("Cantos Encontrados", image)
    cv2.waitKey(0)

