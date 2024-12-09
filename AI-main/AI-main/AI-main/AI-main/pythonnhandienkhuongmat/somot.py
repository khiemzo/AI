import cv2
import numpy as np

# Biến toàn cục để lưu trữ các thông tin về ROI
roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
selecting = False

# Hàm để xử lý việc vẽ vùng ROI
def select_roi(event, x, y, flags, param):
    global roi_x, roi_y, roi_w, roi_h, selecting
    if event == cv2.EVENT_LBUTTONDOWN:
        # Khi nhấn chuột trái, lưu tọa độ bắt đầu
        roi_x, roi_y = x, y
        selecting = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            # Khi di chuyển chuột, tính toán chiều rộng và chiều cao của ROI
            roi_w, roi_h = x - roi_x, y - roi_y
    elif event == cv2.EVENT_LBUTTONUP:
        # Khi thả chuột trái, dừng việc chọn vùng và xử lý ROI
        selecting = False

# Hàm để phát hiện hình tròn trong ROI
def detect_circles_in_roi(image, roi_x, roi_y, roi_w, roi_h):
    # Cắt vùng ROI từ ảnh
    roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Chuyển ảnh ROI sang ảnh xám
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Áp dụng Adaptive Thresholding
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

    # Phát hiện các hình tròn trong ảnh ROI
    circles = cv2.HoughCircles(thresholded, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=100,
                                param2=30, minRadius=5, maxRadius=50)

    # Kiểm tra nếu có hình tròn được phát hiện và vẽ chúng
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Vẽ hình tròn và tâm của nó lên ảnh
            cv2.circle(roi, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(roi, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    return roi, circles

# Đọc ảnh
image = cv2.imread('K:\\New folder\\codepython\\12.jpg')  # Thay 'image.png' bằng đường dẫn đến ảnh của bạn

# Thay đổi kích thước ảnh về 512x512
image = cv2.resize(image, (512, 512))

# Tạo cửa sổ và đăng ký hàm callback chuột
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", select_roi)

# Tạo ảnh copy để vẽ ROI mà không ảnh hưởng đến kết quả
image_for_roi = image.copy()

# Hiển thị ảnh để chọn ROI
while True:
    temp_image = image_for_roi.copy()  # Sao chép ảnh gốc để vẽ lại vùng ROI mới
    if selecting:
        # Vẽ hình chữ nhật khi người dùng đang kéo chuột
        cv2.rectangle(temp_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    # Nếu có ROI đã được chọn và thả chuột, thực hiện phát hiện hình tròn
    if not selecting and roi_w > 0 and roi_h > 0:
        # Sao chép lại ảnh gốc để vẽ kết quả
        result_image = image.copy()

        # Xử lý phát hiện hình tròn và vẽ lên ảnh kết quả
        result_roi, circles = detect_circles_in_roi(result_image, roi_x, roi_y, roi_w, roi_h)

        # Vẽ lại vùng ROI đã chọn
        cv2.rectangle(result_roi, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

        # Hiển thị ảnh kết quả
        cv2.imshow("Detected Circles", result_roi)

    # Hiển thị ảnh đang chọn vùng
    cv2.imshow("Select ROI", temp_image)

    # Nếu nhấn phím 'q', thoát vòng lặp
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
