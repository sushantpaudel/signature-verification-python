import cv2


class PreProcessing:
    def pre_process(self, path=""):
        img = cv2.imread(path)
        img_height, img_width = img.shape[:2]
        scale_height = int(img_height / 100)
        scale_width = int(img_width / 100)
        cv2.imshow("Output", img)
        img = cv2.resize(img, (scale_width, scale_height), interpolation=cv2.INTER_AREA)
        return img
