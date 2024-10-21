import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang ảnh xám
image = cv2.imread('anhnguoi.jpg', cv2.IMREAD_GRAYSCALE)

# Dò biên bằng toán tử Sobel
def sobel_edge_detection(image):
    # Áp dụng bộ lọc Sobel theo trục X và Y
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Dò biên theo trục X
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Dò biên theo trục Y
    
    # Tính độ lớn của gradient
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = cv2.convertScaleAbs(sobel)  # Chuyển đổi về dạng 8-bit
    
    return sobel

# Dò biên bằng Laplace của Gaussian (LoG)
def log_edge_detection(image):
    # Làm mượt ảnh bằng Gaussian để giảm nhiễu
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Áp dụng bộ lọc Laplacian
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)  # Chuyển đổi về dạng 8-bit
    
    return laplacian

# Áp dụng phương pháp Sobel và LoG
sobel_edges = sobel_edge_detection(image)
log_edges = log_edge_detection(image)

# Hiển thị ảnh gốc và kết quả
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Dò biên - Sobel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(log_edges, cmap='gray')
plt.title('Dò biên - Laplace of Gaussian (LoG)')
plt.axis('off')

plt.show()
