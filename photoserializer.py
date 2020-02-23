import time
from datetime import datetime
import os


FORMAT_STRING = "%Y-%m-%d_%H-%M-%S"
DELAY_TIME_SEC = 5

def installPackages(packages):
    packages = packages.split()
    for package in packages:
        global installPythonPackage
        installPythonPackage = 'pip install --save ' + package
        os.system('start cmd /c ' + installPythonPackage)

def main():
    import cv2
    cap = cv2.VideoCapture(1)
    path = '''C:\\Users\\CEE_182\\Documents\\Arduino\\Camera\\image_'''
    while (True):
        _, frame = cap.read()
        t = datetime.now().strftime(FORMAT_STRING)
        cv2.imwrite(f'''{path}{t}.jpg''', frame)
        time.sleep(DELAY_TIME_SEC)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    packages = 'opencv-python'
    installPackages(packages)
    main()
