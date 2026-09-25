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

    found, corners = cv2.findChessboardCorners(gray_image, BOARD_SIZE)

    if found:
        cv2.drawChessboardCorners(image, BOARD_SIZE, corners, found)

        cv2.namedWindow("Cantos Encontrados", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cantos Encontrados", 800, 600)
        cv2.imshow("Cantos Encontrados", image)
        cv2.waitKey(0)
    

    exit(0)
