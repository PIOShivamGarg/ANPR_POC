import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import time

print("="*50)
print("Starting License Plate Recognition")
print("="*50)

start_time = time.time()

# Load image
print("\n1. Loading image...")
img_start = time.time()
img = cv2.imread('images.jpg')
print(f"   Time taken: {time.time() - img_start:.2f}s")

# Grayscale conversion
print("\n2. Converting to grayscale...")
gray_start = time.time()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"   Time taken: {time.time() - gray_start:.2f}s")
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.title('Grayscale Image')
# plt.show()

print("\n3. Applying bilateral filter and edge detection...")
filter_start = time.time()
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)
print(f"   Time taken: {time.time() - filter_start:.2f}s")
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.title('Edge Detection')
# plt.show()

print("\n4. Finding contours and detecting plate location...")
contour_start = time.time()
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
print(f"   Time taken: {time.time() - contour_start:.2f}s")
if location is not None:
    print(f"   Plate location found with {len(location)} corners")
else:
    print("   Warning: No 4-corner plate location found!")

print("\n5. Creating mask and extracting plate region...")
mask_start = time.time()
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)
print(f"   Time taken: {time.time() - mask_start:.2f}s")

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title('Masked License Plate')
# plt.show()

print("\n6. Cropping the plate region...")
crop_start = time.time()
(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
print(f"   Time taken: {time.time() - crop_start:.2f}s")
print(f"   Cropped region size: {cropped_image.shape}")

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title('Cropped License Plate')
# plt.show()

print("\n7. Initializing EasyOCR and reading text...")
ocr_start = time.time()
reader = easyocr.Reader(['en',"tr"])
print(f"   EasyOCR initialization: {time.time() - ocr_start:.2f}s")

ocr_read_start = time.time()
result = reader.readtext(cropped_image)
print(f"   Text reading: {time.time() - ocr_read_start:.2f}s")
print(f"   Total OCR time: {time.time() - ocr_start:.2f}s")

print("\n8. Drawing results on image...")
draw_start = time.time()
text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=0.5, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
print(f"   Time taken: {time.time() - draw_start:.2f}s")

plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title('Final Result with Detected Plate')
# plt.show()

total_time = time.time() - start_time

print("\n" + "="*50)
print(f"Detected Text: {text}")
print(f"Total Execution Time: {total_time:.2f}s")
print("="*50)