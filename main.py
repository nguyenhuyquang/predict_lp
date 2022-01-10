import cv2

from recognition import load_model

def regMain(image_path):
    # Read Image
    image = cv2.imread(str(image_path))

    # Load Model
    model = load_model()

    # recognize license plate
    license_plate = model.predict(image)

    return license_plate

