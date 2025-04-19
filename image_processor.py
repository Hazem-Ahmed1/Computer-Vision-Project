import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("1200x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create buttons
        self.select_button = ttk.Button(self.main_frame, text="Select Image", command=self.select_image)
        self.select_button.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Create processing options
        self.process_var = tk.StringVar(value="edge")
        self.process_options = ttk.Frame(self.main_frame)
        self.process_options.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Radiobutton(self.process_options, text="Edge Detection", variable=self.process_var, 
                       value="edge").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(self.process_options, text="Blur", variable=self.process_var, 
                       value="blur").grid(row=0, column=1, padx=5)
        ttk.Radiobutton(self.process_options, text="Sharpen", variable=self.process_var, 
                       value="sharpen").grid(row=0, column=2, padx=5)
        
        # Create image display areas
        self.original_label = ttk.Label(self.main_frame)
        self.original_label.grid(row=2, column=0, padx=10, pady=10)
        
        self.processed_label = ttk.Label(self.main_frame)
        self.processed_label.grid(row=2, column=1, padx=10, pady=10)
        
        # Add labels for the images
        ttk.Label(self.main_frame, text="Original Image").grid(row=3, column=0)
        ttk.Label(self.main_frame, text="Processed Image").grid(row=3, column=1)
        
        self.original_image = None
        self.processed_image = None

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.process_image()
            self.display_images()

    def process_image(self):
        if self.original_image is None:
            return
            
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        if self.process_var.get() == "edge":
            # Apply Canny edge detection
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            self.processed_image = cv2.Canny(gray, 100, 200)
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
        
        elif self.process_var.get() == "blur":
            # Apply Gaussian blur
            self.processed_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)
        
        elif self.process_var.get() == "sharpen":
            # Apply sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.processed_image = cv2.filter2D(rgb_image, -1, kernel)

    def display_images(self):
        if self.original_image is None:
            return
            
        # Convert original image to RGB for display
        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Resize images to fit the window
        height, width = original_rgb.shape[:2]
        max_size = 500
        scale = min(max_size/width, max_size/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        original_resized = cv2.resize(original_rgb, (new_width, new_height))
        processed_resized = cv2.resize(self.processed_image, (new_width, new_height))
        
        # Convert to PhotoImage
        original_photo = ImageTk.PhotoImage(image=Image.fromarray(original_resized))
        processed_photo = ImageTk.PhotoImage(image=Image.fromarray(processed_resized))
        
        # Update labels
        self.original_label.configure(image=original_photo)
        self.original_label.image = original_photo
        
        self.processed_label.configure(image=processed_photo)
        self.processed_label.image = processed_photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop() 