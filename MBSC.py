import cv2
import numpy as np


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global position
        position = [y, x]


position = []
measures = []
bgr_color = []
hsv_color = []
lower_bound = np.array([3, 170, 200])
higher_bound = np.array([14, 255, 255])

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("Camera", on_mouse_click)

while cam.isOpened():
    _, image = cam.read()
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, higher_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mod = cv2.bitwise_and(blurred, blurred, mask=mask)
    gray = cv2.cvtColor(mod, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    fg = np.uint8(fg)
    confuse = cv2.subtract(thresh, fg)
    _, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[confuse==255] = 0
    wmarkers = cv2.watershed(mod, markers.copy())
    contours, hierarchy = cv2.findContours(wmarkers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # IMAGE TO CAMERA
    output = thresh

    if position:
        pxl = image[position[0], position[1]]
        measures.append(pxl)
        if len(measures) >= 10:
            bgr_color = np.uint8([[np.average(measures, 0)]])
            hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
            bgr_color = bgr_color[0, 0]
            hsv_color = hsv_color[0, 0]
            measures.clear()
        print(hsv_color)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(output, contours, i, (255, 0, 0), 10)
    cnt = int(len(hierarchy[0] - 1) / 2)
    cv2.putText(output, f"{cnt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", output)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()