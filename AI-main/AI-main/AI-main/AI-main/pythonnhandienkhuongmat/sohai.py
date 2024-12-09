import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Biến toàn cục
img = np.ones((512, 512, 3), dtype=np.uint8) * 255  # Ảnh trắng ban đầu
roi = None  # Lưu vùng ROI
start_point = None  # Điểm bắt đầu của ROI
temp_img = None  # Ảnh tạm để vẽ ROI

# Hàm chọn ROI bằng Tkinter Event
def select_roi(event):
    global start_point, roi, temp_img
    if event.type == EventType.ButtonPress:  # Bắt đầu vẽ ROI
        start_point = (event.x, event.y)
    elif event.type == EventType.Motion and start_point:  # Đang kéo chuột
        temp_img = img.copy()
        cv2.rectangle(temp_img, start_point, (event.x, event.y), (255, 0, 0), 2)
        display_image(temp_img)
    elif event.type == EventType.ButtonRelease:  # Kết thúc vẽ ROI
        roi = (min(start_point[0], event.x), min(start_point[1], event.y),
               max(start_point[0], event.x), max(start_point[1], event.y))
        start_point = None
        temp_img = img.copy()
        cv2.rectangle(temp_img, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
        display_image(temp_img)
        process_image()  # Cập nhật ảnh và kết quả ngay sau khi chọn ROI

# Hàm phát hiện hình tròn trong ROI
def process_image():
    global roi, img
    if img is None:
        return

    param1 = param1_slider.get()
    param2 = param2_slider.get()
    min_radius = min_radius_slider.get()
    max_radius = max_radius_slider.get()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    result_image = img.copy()
    num_circles = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Kiểm tra nếu nằm trong ROI
            if roi is None or (roi[0] <= i[0] <= roi[2] and roi[1] <= i[1] <= roi[3]):
                cv2.circle(result_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(result_image, (i[0], i[1]), 2, (0, 0, 255), 3)
                num_circles += 1  # Chỉ đếm hình tròn được vẽ

    # Vẽ ROI nếu có
    if roi is not None:
        cv2.rectangle(result_image, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)

    display_image(result_image)
    update_circle_count(num_circles)  # Cập nhật số lượng hình tròn trên giao diện

# Cập nhật số lượng hình tròn
def update_circle_count(count):
    circle_count_label.config(text=f"Circles Detected: {count}")

# Tải ảnh từ file
def load_image():
    global img, temp_img
    file_path = filedialog.askopenfilename()
    if file_path:
        img_loaded = cv2.imread(file_path)
        img_resized = cv2.resize(img_loaded, (512, 512))
        img[:] = img_resized
        temp_img = img.copy()
        display_image(img)
        process_image()

# Hiển thị ảnh
def display_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)
    image_label.config(image=image_tk)
    image_label.image = image_tk

# Tạo cửa sổ giao diện
root = Tk()
root.title("Circle Detection with ROI and Real-Time Adjustment")

main_frame = Frame(root, bg="#f0f0f0")
main_frame.pack(fill=BOTH, expand=True)

# Khung ảnh
image_frame = Frame(main_frame, bg="#e6e6e6", relief=RIDGE, borderwidth=3)
image_frame.pack(side=LEFT, padx=10, pady=10)

image_label = Label(image_frame)
image_label.pack()

# Gắn callback chuột
image_label.bind("<Button-1>", select_roi)
image_label.bind("<B1-Motion>", select_roi)
image_label.bind("<ButtonRelease-1>", select_roi)

# Khung điều khiển
controls_frame = Frame(main_frame, bg="#e6e6e6", relief=RIDGE, borderwidth=3)
controls_frame.pack(side=RIGHT, padx=10, pady=10)

# Nút tải ảnh
load_button = Button(controls_frame, text="Load Image", command=load_image,
                     bg="#007ACC", fg="white", font=("Arial", 10, "bold"),
                     relief=RAISED, padx=5, pady=5)
load_button.grid(row=0, column=0, columnspan=3, pady=5, sticky=W+E)

# Thanh trượt
def create_slider(root, label_text, row, from_, to_, initial_value):
    label = Label(root, text=label_text, bg="#e6e6e6", font=("Arial", 10))
    label.grid(row=row, column=0, padx=5, pady=5, sticky=W)

    slider = Scale(root, from_=from_, to=to_, orient=HORIZONTAL, length=300,
                   bg="#dcdcdc", highlightbackground="#e6e6e6", command=lambda _: process_image())
    slider.set(initial_value)
    slider.grid(row=row, column=1, padx=5, pady=5)

    return slider

param1_slider = create_slider(controls_frame, "Canny Upper Threshold", 1, 50, 300, 100)
param2_slider = create_slider(controls_frame, "Accumulator Threshold", 2, 10, 100, 30)
min_radius_slider = create_slider(controls_frame, "Min Radius", 3, 0, 100, 10)
max_radius_slider = create_slider(controls_frame, "Max Radius", 4, 0, 100, 50)

# Số lượng hình tròn
circle_count_label = Label(controls_frame, text="Circles Detected: 0", bg="#e6e6e6", font=("Arial", 12, "bold"))
circle_count_label.grid(row=5, column=0, columnspan=4, pady=10)

# Nút in kết quả
print_button = Button(controls_frame, text="Print Results", command=lambda: print(f"Circles Detected: {circle_count_label.cget('text')}"),
                      bg="#28a745", fg="white", font=("Arial", 12, "bold"),
                      relief=RAISED, padx=5, pady=5)
print_button.grid(row=6, column=0, columnspan=4, pady=10)

root.mainloop()
