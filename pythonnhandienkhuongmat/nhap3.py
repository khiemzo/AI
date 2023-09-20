import face_recognition as fr
import cv2
import numpy as np
import os
'''
face_recognition: Thư viện chứa các hàm liên quan đến nhận diện và mã hóa khuôn mặt.
cv2: Thư viện OpenCV dùng để xử lý hình ảnh.
numpy as np: Thư viện hỗ trợ xử lý mảng nhiều chiều, cần thiết cho việc tính toán toán học.
os: Thư viện hỗ trợ thao tác với hệ điều hành, trong trường hợp này dùng để tải danh sách hình ảnh từ thư mục.
'''
#Thiết lập đường dẫn và biến lưu thông tin người đã biết
path = "./train/"

known_names = []
known_name_encodings = []
'''
path: Đường dẫn đến thư mục chứa hình ảnh của những người đã biết.
known_names: Danh sách tên của những người đã biết.
known_name_encodings: Danh sách các mã hóa khuôn mặt của những người đã biết.
'''
images = os.listdir(path)
for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    encoding = fr.face_encodings(image)[0]

    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())
'''
Duyệt qua tất cả các tệp hình ảnh trong thư mục train.
Tải ảnh và lấy mã hóa khuôn mặt (nếu có).
Thêm mã hóa khuôn mặt vào danh sách known_name_encodings.
Trích xuất tên của người từ tên tệp và thêm vào danh sách known_names.
'''
print(known_names)

test_image = "./test/test.jpg"
image = cv2.imread(test_image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)
'''
Tải hình ảnh cần kiểm tra từ thư mục test.
Sử dụng face_recognition để tìm vị trí của các khuôn mặt trong hình ảnh.
'''
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)
    name = ""

    face_distances = fr.face_distance(known_name_encodings, face_encoding)
    best_match = np.argmin(face_distances)

    if matches[best_match]:
        name = known_names[best_match]
'''
Duyệt qua tất cả khuôn mặt được tìm thấy.
So sánh khuôn mặt với danh sách người đã biết.
Tìm người có khuôn mặt tương tự nhất.
'''
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
'''
Vẽ khung xung quanh khuôn mặt được nhận diện.
Hiển thị tên của người đã biết lên hình ảnh.
'''

cv2.imshow("Result", image)
cv2.imwrite("./output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
Hiển thị hình ảnh với kết quả đã được xử lý.
Ghi kết quả vào một tệp tin hình ảnh mới.
Chờ người dùng nhấn một phím bất kỳ để đóng cửa sổ hình ảnh.
'''