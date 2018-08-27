import cv2 as cv2


def main():
    img = cv2.imread('training/Anemone_canadensis.88485.3642.jpg')
    img = cv2.resize(img, (576, 576))
    cv2.imwrite('test.jpg', img)
    return 0

if __name__ == '__main__':
    main()