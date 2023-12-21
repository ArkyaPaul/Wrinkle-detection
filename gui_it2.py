import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

class FaceAnalyzerApp:
    def __init__(self, master):
        self.master = master
        self.master.geometry('800x600')
        self.master.title('Face Analyzer')
        self.master.configure(background='#CDCDCD')

        self.label_result = Label(self.master, background='#CDCDCD', font=('arial', 15, 'bold'))
        self.sign_image = Label(self.master)

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.wrinkle_model = self.load_model("wrinkle.json", "wrinkle_model.h5")

        self.WRINKLE_CLASSES = ["No Wrinkle", "Wrinkle"]

        self.setup_gui()

    def setup_gui(self):
        upload_button = Button(self.master, text="Upload Image", command=self.upload_image, padx=10, pady=5)
        upload_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
        upload_button.pack(side='bottom', pady=50)

        self.sign_image.pack(side='bottom', expand=True)
        self.label_result.pack(side='bottom', expand=True)

        heading = Label(self.master, text='Face Analyzer', pady=20, font=('arial', 25, 'bold'))
        heading.configure(background='#CDCDCD', foreground="#364156")
        heading.pack()

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((self.master.winfo_width() / 2.25), (self.master.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(uploaded)

            self.sign_image.configure(image=im)
            self.sign_image.image = im
            self.label_result.configure(text='')
            self.show_detect_buttons(file_path)
        except Exception as e:
            print(f"Error uploading image: {e}")

    def show_detect_buttons(self, file_path):
        detect_wrinkle_button = Button(self.master, text="Wrinkle Detection", command=lambda: self.detect_wrinkle(file_path), padx=10, pady=5)
        detect_wrinkle_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        detect_wrinkle_button.place(relx=0.75, rely=0.54)


    def detect_wrinkle(self, file_path):
        try:
            image = cv2.imread(file_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = cv2.resize(gray_image[y:y + h, x:x + w], (100, 100))
                pred = self.WRINKLE_CLASSES[np.argmax(self.wrinkle_model.predict(roi[np.newaxis, :, :, np.newaxis]))]

            print("Predicted Wrinkle Status: " + pred)
            self.label_result.configure(foreground="#011638", text=f"Wrinkle Status: {pred}")
        except Exception as e:
            self.label_result.configure(foreground="#011638", text="Unable to detect")

    def load_model(self, wrinkle, wrinkle_model):
        try:
            with open(wrinkle, "r") as file:
                loaded_model_json = file.read()
                model = model_from_json(loaded_model_json)

            model.load_weights(wrinkle_model)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            return model
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAnalyzerApp(root)
    root.mainloop()
