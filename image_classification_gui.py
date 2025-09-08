import os
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tkinter import Tk, Button, Label, filedialog
import sys

# دالة لتحديد المسار الصحيح للملفات سواء في التشغيل العادي أو من الـ EXE
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # لو شغال كـ EXE من PyInstaller
        base_path = sys._MEIPASS
    except Exception:
        # لو شغال كملف Python عادي
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# مسارات الملفات
model_path = r"D:\AI\1. Courses\2. Computer Vision\2. Codes\Computer Vision Codes\keras_model.h5"
labels_path = r"D:\AI\1. Courses\2. Computer Vision\2. Codes\Computer Vision Codes\labels.txt"
# تحميل الموديل
model = load_model(model_path, compile=False)

# تحميل التسميات
with open(labels_path, "r") as f:
    class_names = f.readlines()

category_to_index = {}
for class_name in class_names:
    index, name = class_name.split(' ')
    name = name.strip()
    category_to_index[name] = int(index)

# مصفوفة الـ prediction
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# دالة التنبؤ بالصورة
def predict_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# دالة اختيار صورة من الجهاز
def choose_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    image = cv2.imread(file_path)
    if image is None:
        result_label.config(text="Error: Could not read image.")
        return
    class_name, confidence = predict_image(image)
    result_label.config(text=f"Predicted: {class_name}, Confidence: {confidence:.2f}")

# إنشاء واجهة GUI بسيطة
root = Tk()
root.title("Image Classification GUI")

choose_button = Button(root, text="Choose Image", command=choose_image, width=20, height=2)
choose_button.pack(pady=20)

result_label = Label(root, text="Result will appear here", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()
