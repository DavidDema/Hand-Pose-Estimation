import cv2

camera = cv2.VideoCapture(0)
return_value, image = camera.read()
for i in range(3):
    cv2.imwrite('img/opencv_rgb_'+str(i)+'.png', image[:,:,i])
del(camera)