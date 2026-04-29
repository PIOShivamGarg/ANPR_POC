import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import time

print("="*50)
print("ANPR - Automatic Number Plate Recognition")
print("="*50)

start_time = time.time()

print("\n[1/6] Loading image...")
step_start = time.time()
img = cv2.imread('images.jpg')
# img = cv2.imread('number_plate.jpg')
print(f"    ✓ Image loaded in {time.time() - step_start:.3f}s")

print("\n[2/6] Preprocessing image...")
step_start = time.time()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.title('Grayscale Image')
plt.show()

bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)
print(f"    ✓ Preprocessing completed in {time.time() - step_start:.3f}s")
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.title('Edge Detection')
plt.show()

print("\n[3/6] Detecting license plate...")
step_start = time.time()
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

if location is not None:
    print(f"    ✓ License plate detected in {time.time() - step_start:.3f}s")
else:
    print(f"    ✗ No license plate found")

location

print("\n[4/6] Extracting plate region...")
step_start = time.time()
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title('Masked License Plate')
plt.show()

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
print(f"    ✓ Plate region extracted in {time.time() - step_start:.3f}s")

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title('Cropped License Plate')
plt.show()

print("\n[5/6] Running OCR...")
step_start = time.time()
reader = easyocr.Reader(['en',"tr"])
result = reader.readtext(cropped_image)
print(f"    ✓ OCR completed in {time.time() - step_start:.3f}s")
result

print("\n[6/6] Generating result...")
step_start = time.time()
text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=0.5, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title('Final Result with Detected Plate')
plt.show()
print(f"    ✓ Result generated in {time.time() - step_start:.3f}s")

total_time = time.time() - start_time
print("\n" + "="*50)
print(f"✓ DETECTED PLATE NUMBER: {text}")
print(f"✓ Total execution time: {total_time:.3f}s")
print("="*50)
print("="*50)