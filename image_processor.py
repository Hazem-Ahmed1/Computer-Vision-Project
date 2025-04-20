# import tkinter as tk
# from tkinter import filedialog, ttk
# import cv2
# import numpy as np
# from PIL import Image, ImageTk

# class ImageProcessorApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Image Processor")
#         self.root.geometry("1200x600")
        
#         # Create main frame
#         self.main_frame = ttk.Frame(self.root, padding="10")
#         self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
#         # Create buttons
#         self.select_button = ttk.Button(self.main_frame, text="Select Image", command=self.select_image)
#         self.select_button.grid(row=0, column=0, columnspan=2, pady=10)
        
#         # Create processing options
#         self.process_var = tk.StringVar(value="edge")
#         self.process_options = ttk.Frame(self.main_frame)
#         self.process_options.grid(row=1, column=0, columnspan=2, pady=10)
        
#         ttk.Radiobutton(self.process_options, text="Edge Detection", variable=self.process_var, 
#                        value="edge").grid(row=0, column=0, padx=5)
#         ttk.Radiobutton(self.process_options, text="Blur", variable=self.process_var, 
#                        value="blur").grid(row=0, column=1, padx=5)
#         ttk.Radiobutton(self.process_options, text="Sharpen", variable=self.process_var, 
#                        value="sharpen").grid(row=0, column=2, padx=5)
        
#         # Create image display areas
#         self.original_label = ttk.Label(self.main_frame)
#         self.original_label.grid(row=2, column=0, padx=10, pady=10)
        
#         self.processed_label = ttk.Label(self.main_frame)
#         self.processed_label.grid(row=2, column=1, padx=10, pady=10)
        
#         # Add labels for the images
#         ttk.Label(self.main_frame, text="Original Image").grid(row=3, column=0)
#         ttk.Label(self.main_frame, text="Processed Image").grid(row=3, column=1)
        
#         self.original_image = None
#         self.processed_image = None

#     def select_image(self):
#         file_path = filedialog.askopenfilename(
#             filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
#         if file_path:
#             self.original_image = cv2.imread(file_path)
#             self.process_image()
#             self.display_images()

#     def process_image(self):
#         if self.original_image is None:
#             return
            
#         # Convert BGR to RGB
#         rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
#         if self.process_var.get() == "edge":
#             # Apply Canny edge detection
#             gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
#             self.processed_image = cv2.Canny(gray, 100, 200)
#             self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
        
#         elif self.process_var.get() == "blur":
#             # Apply Gaussian blur
#             self.processed_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)
        
#         elif self.process_var.get() == "sharpen":
#             # Apply sharpening
#             kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#             self.processed_image = cv2.filter2D(rgb_image, -1, kernel)

#     def display_images(self):
#         if self.original_image is None:
#             return
            
#         # Convert original image to RGB for display
#         original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
#         # Resize images to fit the window
#         height, width = original_rgb.shape[:2]
#         max_size = 500
#         scale = min(max_size/width, max_size/height)
#         new_width = int(width * scale)
#         new_height = int(height * scale)
        
#         original_resized = cv2.resize(original_rgb, (new_width, new_height))
#         processed_resized = cv2.resize(self.processed_image, (new_width, new_height))
        
#         # Convert to PhotoImage
#         original_photo = ImageTk.PhotoImage(image=Image.fromarray(original_resized))
#         processed_photo = ImageTk.PhotoImage(image=Image.fromarray(processed_resized))
        
#         # Update labels
#         self.original_label.configure(image=original_photo)
#         self.original_label.image = original_photo
        
#         self.processed_label.configure(image=processed_photo)
#         self.processed_label.image = processed_photo

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ImageProcessorApp(root)
#     root.mainloop() 


# import tkinter as tk
# from tkinter import filedialog, ttk
# import cv2
# import numpy as np
# from PIL import Image, ImageTk

# class ImageProcessorApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Image Processor")
#         self.root.minsize(800, 400)
#         self.root.configure(bg="#ffffff")

#         # Configure styles
#         self.style = ttk.Style()
#         self.style.configure("Main.TFrame", background="#ffffff")
#         self.style.configure("TButton", padding=8, font=("Arial", 10))
#         self.style.configure("TLabel", background="#ffffff", font=("Arial", 10))
#         self.style.configure("Header.TLabel", font=("Arial", 12, "bold"))
#         self.style.configure("Accent.TButton",
#                              background="#09c",
#                              foreground="white",
#                              font=("Arial", 10, "bold"))
#         self.style.map("Accent.TButton",
#                        background=[('active', '#007a99')],
#                        foreground=[('active', 'white')])
#         self.style.configure("Header.TFrame",
#                              background="#09c",
#                              relief="flat")

#         # Create header frame (fixed at the top)
#         self.header_frame = tk.Frame(self.root, height=60)
#         self.header_frame.pack(fill=tk.X, side=tk.TOP)

#         # Canvas for gradient + centered text
#         self.header_canvas = tk.Canvas(self.header_frame, height=60, highlightthickness=0, bd=0)
#         self.header_canvas.pack(fill=tk.BOTH, expand=True)

#         # Function to draw gradient and text
#         def draw_header_gradient(event=None):
#             self.header_canvas.delete("all")
#             width = self.header_canvas.winfo_width()
#             height = self.header_canvas.winfo_height()
#             steps = height

#             for i in range(steps):
#                 r = int(0 + (102 - 0) * (i / steps))
#                 g = int(153 + (204 - 153) * (i / steps))
#                 b = int(204 + (255 - 204) * (i / steps))
#                 color = f'#{r:02x}{g:02x}{b:02x}'
#                 self.header_canvas.create_line(0, i, width, i, fill=color)

#             self.header_canvas.create_text(
#                 width // 2,
#                 height // 2,
#                 text="VisionCraft",
#                 fill="white",
#                 font=("Arial", 16, "bold")
#             )

#         # Bind the draw function to resizing
#         self.header_canvas.bind("<Configure>", draw_header_gradient)

#         # Create main frame
#         self.main_frame = ttk.Frame(self.root, style="Main.TFrame", padding=20)
#         self.main_frame.pack(fill=tk.BOTH, expand=True)
#         self.main_frame.columnconfigure(0, weight=1)
#         self.main_frame.columnconfigure(1, weight=1)

#         # Create control panel
#         self.control_panel = ttk.Frame(self.main_frame, style="Main.TFrame")
#         self.control_panel.grid(row=0, column=0, columnspan=2, pady=(10, 20), sticky=(tk.W, tk.E))
#         self.control_panel.columnconfigure(1, weight=1)

#         # Create select image button
#         self.select_button = ttk.Button(
#             self.control_panel,
#             text="Select Image",
#             style="Accent.TButton",
#             command=self.select_image
#         )
#         self.select_button.grid(row=0, column=0, padx=(0, 10))
#         self.style.configure("Accent.TButton", foreground="black")  
#         self.style.map("Accent.TButton", background=[], foreground=[])  
#         self.select_button.configure(cursor="hand2")  

#         # Processing options dropdown
#         self.process_var = tk.StringVar(value="Edge Detection")
#         self.process_options = ttk.Combobox(
#             self.control_panel,
#             textvariable=self.process_var,
#             values=["Edge Detection", "Blur", "Sharpen"],
#             state="readonly",
#             width=20
#         )
#         self.process_options.grid(row=0, column=1, sticky=tk.W)
#         self.process_options.bind("<<ComboboxSelected>>", self.on_process_change)

#         # dropdown appearance
#         self.style.configure("TCombobox", padding=10, font=("Arial", 10))
#         self.style.map("TCombobox",
#                        fieldbackground=[('readonly', '#f0f2f5')],
#                        background=[('readonly', '#ffffff')],
#                        foreground=[('readonly', '#000000')])

#         # Create image display areas
#         self.image_container = ttk.Frame(self.main_frame, style="Main.TFrame")
#         self.image_container.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
#         self.image_container.columnconfigure(0, weight=1)
#         self.image_container.columnconfigure(1, weight=1)

#         # Original image
#         self.original_container = ttk.Frame(self.image_container, style="Main.TFrame")
#         self.original_container.grid(row=0, column=0, padx=15, pady=15, sticky=(tk.W, tk.E))
#         ttk.Label(self.original_container,
#                   text="Original Image",
#                   style="Header.TLabel").grid(row=0, column=0, pady=(0, 8))
#         self.original_label = ttk.Label(self.original_container, style="Main.TLabel")
#         self.original_label.grid(row=1, column=0)

#         # Processed image
#         self.processed_container = ttk.Frame(self.image_container, style="Main.TFrame")
#         self.processed_container.grid(row=0, column=1, padx=15, pady=15, sticky=(tk.W, tk.E))
#         ttk.Label(self.processed_container,
#                   text="Processed Image",
#                   style="Header.TLabel").grid(row=0, column=0, pady=(0, 8))
#         self.processed_label = ttk.Label(self.processed_container, style="Main.TLabel")
#         self.processed_label.grid(row=1, column=0)

#         # Initialize images
#         self.original_image = None
#         self.processed_image = None

#         # Bind window resize
#         self.root.bind("<Configure>", self.on_resize)

#     def select_image(self):
#         file_path = filedialog.askopenfilename(
#             filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
#         if file_path:
#             self.original_image = cv2.imread(file_path)
#             self.process_image()
#             self.display_images()

#     def process_image(self):
#         if self.original_image is None:
#             return

#         # Convert BGR to RGB
#         rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

#         if self.process_var.get() == "Edge Detection":
#             gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
#             self.processed_image = cv2.Canny(gray, 100, 200)
#             self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)

#         elif self.process_var.get() == "Blur":
#             self.processed_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)

#         elif self.process_var.get() == "Sharpen":
#             kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#             self.processed_image = cv2.filter2D(rgb_image, -1, kernel)

#     def display_images(self):
#         if self.original_image is None:
#             return

#         # Get window size
#         window_width = self.root.winfo_width()
#         window_height = self.root.winfo_height()

#         # Convert original image to RGB
#         original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

#         # Calculate responsive size
#         max_size = min(window_width // 2 - 80, window_height - 200)
#         max_size = max(200, min(max_size, 800))  # Limit size range

#         height, width = original_rgb.shape[:2]
#         scale = min(max_size / width, max_size / height)
#         new_width = int(width * scale)
#         new_height = int(height * scale)

#         # Resize images
#         original_resized = cv2.resize(original_rgb, (new_width, new_height))
#         processed_resized = cv2.resize(self.processed_image, (new_width, new_height))

#         # Convert to PhotoImage
#         original_photo = ImageTk.PhotoImage(image=Image.fromarray(original_resized))
#         processed_photo = ImageTk.PhotoImage(image=Image.fromarray(processed_resized))

#         # Update labels
#         self.original_label.configure(image=original_photo)
#         self.original_label.image = original_photo

#         self.processed_label.configure(image=processed_photo)
#         self.processed_label.image = processed_photo

#     def on_process_change(self, event):
#         if self.original_image is not None:
#             self.process_image()
#             self.display_images()

#     def on_resize(self, event):
#         if self.original_image is not None:
#             self.display_images()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ImageProcessorApp(root)
#     root.mainloop()


import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.minsize(800, 400)
        self.root.configure(bg="#ffffff")

        self.style = ttk.Style()
        self.style.configure("Main.TFrame", background="#ffffff")
        self.style.configure("TButton", padding=8, font=("Arial", 10))
        self.style.configure("TLabel", background="#ffffff", font=("Arial", 10))
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        self.style.configure("Accent.TButton",
                             background="#09c",
                             foreground="white",
                             font=("Arial", 10, "bold"))
        self.style.map("Accent.TButton",
                       background=[('active', '#007a99')],
                       foreground=[('active', 'white')])
        self.style.configure("Header.TFrame",
                             background="#09c",
                             relief="flat")

        self.header_frame = tk.Frame(self.root, height=60)
        self.header_frame.pack(fill=tk.X, side=tk.TOP)

        self.header_canvas = tk.Canvas(self.header_frame, height=60, highlightthickness=0, bd=0)
        self.header_canvas.pack(fill=tk.BOTH, expand=True)

        def draw_header_gradient(event=None):
            self.header_canvas.delete("all")
            width = self.header_canvas.winfo_width()
            height = self.header_canvas.winfo_height()
            steps = height

            for i in range(steps):
                r = int(0 + (102 - 0) * (i / steps))
                g = int(153 + (204 - 153) * (i / steps))
                b = int(204 + (255 - 204) * (i / steps))
                color = f'#{r:02x}{g:02x}{b:02x}'
                self.header_canvas.create_line(0, i, width, i, fill=color)

            self.header_canvas.create_text(
                width // 2,
                height // 2,
                text="VisionCraft",
                fill="white",
                font=("Arial", 16, "bold")
            )

        self.header_canvas.bind("<Configure>", draw_header_gradient)

        self.main_frame = ttk.Frame(self.root, style="Main.TFrame", padding=20)
        self.main_frame.pack(expand=True)

        self.control_panel = ttk.Frame(self.main_frame, style="Main.TFrame")
        self.control_panel.pack(anchor="center", pady=(10, 20))

        self.select_button = ttk.Button(
            self.control_panel,
            text="Select Image",
            style="Accent.TButton",
            command=self.select_image
        )
        self.select_button.grid(row=0, column=0, padx=(0, 10))
        self.style.configure("Accent.TButton", foreground="black")
        self.style.map("Accent.TButton", background=[], foreground=[])
        self.select_button.configure(cursor="hand2")

        self.process_var = tk.StringVar(value="Edge Detection")
        self.process_options = ttk.Combobox(
            self.control_panel,
            textvariable=self.process_var,
            values=["Edge Detection", "Blur", "Sharpen"],
            state="readonly",
            width=20
        )
        self.process_options.grid(row=0, column=1, sticky=tk.W)
        self.process_options.bind("<<ComboboxSelected>>", self.on_process_change)

        self.style.configure("TCombobox", padding=10, font=("Arial", 10))
        self.style.map("TCombobox",
                       fieldbackground=[('readonly', '#f0f2f5')],
                       background=[('readonly', '#ffffff')],
                       foreground=[('readonly', '#000000')])

        self.image_container = ttk.Frame(self.main_frame, style="Main.TFrame")
        self.image_container.pack(anchor="center", pady=10)

        self.original_container = ttk.Frame(self.image_container, style="Main.TFrame")
        self.original_container.grid(row=0, column=0, padx=15, pady=15, sticky=(tk.W, tk.E))
        ttk.Label(self.original_container,
                  text="Original Image",
                  style="Header.TLabel").grid(row=0, column=0, pady=(0, 8))
        self.original_label = ttk.Label(self.original_container, style="Main.TLabel")
        self.original_label.grid(row=1, column=0)

        self.processed_container = ttk.Frame(self.image_container, style="Main.TFrame")
        self.processed_container.grid(row=0, column=1, padx=15, pady=15, sticky=(tk.W, tk.E))
        ttk.Label(self.processed_container,
                  text="Processed Image",
                  style="Header.TLabel").grid(row=0, column=0, pady=(0, 8))
        self.processed_label = ttk.Label(self.processed_container, style="Main.TLabel")
        self.processed_label.grid(row=1, column=0)

        self.original_image = None
        self.processed_image = None

        self.root.bind("<Configure>", self.on_resize)

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

        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        if self.process_var.get() == "Edge Detection":
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            self.processed_image = cv2.Canny(gray, 100, 200)
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)

        elif self.process_var.get() == "Blur":
            self.processed_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)

        elif self.process_var.get() == "Sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            self.processed_image = cv2.filter2D(rgb_image, -1, kernel)

    def display_images(self):
        if self.original_image is None:
            return

        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        max_size = min(window_width // 2 - 80, window_height - 200)
        max_size = max(200, min(max_size, 800))

        height, width = original_rgb.shape[:2]
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        original_resized = cv2.resize(original_rgb, (new_width, new_height))
        processed_resized = cv2.resize(self.processed_image, (new_width, new_height))

        original_photo = ImageTk.PhotoImage(image=Image.fromarray(original_resized))
        processed_photo = ImageTk.PhotoImage(image=Image.fromarray(processed_resized))

        self.original_label.configure(image=original_photo)
        self.original_label.image = original_photo

        self.processed_label.configure(image=processed_photo)
        self.processed_label.image = processed_photo

    def on_process_change(self, event):
        if self.original_image is not None:
            self.process_image()
            self.display_images()

    def on_resize(self, event):
        if self.original_image is not None:
            self.display_images()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
