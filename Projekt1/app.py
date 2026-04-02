import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from imageProcessor import ImageProcessor

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Biometria Projekt 1")
        self.processor = None
        
        self.bg_color = "#e0e0e0"
        self.disp_width = 400
        self.disp_height = 400
        
        self.frame_left = tk.Frame(root, width=250, bg=self.bg_color, padx=10, pady=10)
        self.frame_left.pack(side=tk.LEFT, fill=tk.Y)
        self.frame_center = tk.Frame(root, bg="white")
        self.frame_center.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        # horizontal projection
        self.fig_ph = plt.figure(figsize=(4, 1.2))
        self.canvas_ph = FigureCanvasTkAgg(self.fig_ph, master=self.frame_center)
        self.canvas_ph.get_tk_widget().grid(row=0, column=0, padx=(10,0), pady=(10,0), sticky="sw")

        # picture
        self.canvas_img = tk.Canvas(self.frame_center, bg="white", width=self.disp_width, height=self.disp_height, highlightthickness=0)
        self.canvas_img.grid(row=1, column=0, padx=(10,0), pady=(5,0), sticky="nw")

        # vertical projection
        self.fig_pv = plt.figure(figsize=(1.2, 4))
        self.canvas_pv = FigureCanvasTkAgg(self.fig_pv, master=self.frame_center)
        self.canvas_pv.get_tk_widget().grid(row=1, column=1, padx=(5,10), pady=(5,0), sticky="nw")

        # histogram
        self.frame_hist = tk.Frame(self.frame_center, bg="white")
        self.frame_hist.grid(row=2, column=0, padx=10, pady=10, sticky="nw")
        
        self.fig_hist = plt.figure(figsize=(1.5, 1.5), tight_layout=True)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self.frame_hist)
        self.canvas_hist.get_tk_widget().pack()

        self.show_r = tk.BooleanVar(value=True)
        self.show_g = tk.BooleanVar(value=True)
        self.show_b = tk.BooleanVar(value=True)
        f_checks = tk.Frame(self.frame_hist, bg="white")
        f_checks.pack()
        tk.Checkbutton(f_checks, text="R", fg="red", bg="white", variable=self.show_r, command=self.update_plots).pack(side=tk.LEFT)
        tk.Checkbutton(f_checks, text="G", fg="green", bg="white", variable=self.show_g, command=self.update_plots).pack(side=tk.LEFT)
        tk.Checkbutton(f_checks, text="B", fg="blue", bg="white", variable=self.show_b, command=self.update_plots).pack(side=tk.LEFT)

        # sidebar 
        self.create_btn(self.frame_left, "Wczytaj obraz", self.load_image).pack(fill=tk.X, pady=(0, 2))
        self.create_btn(self.frame_left, "Zapisz obraz", self.save_image).pack(fill=tk.X, pady=(0, 2))
        self.create_btn(self.frame_left, "Resetuj obraz", self.reset_image).pack(fill=tk.X, pady=(0, 15))
        
        self.create_btn(self.frame_left, "Szarość", lambda: self.apply_func(lambda: self.processor.grayscaleLum())).pack(fill=tk.X)
        self.create_btn(self.frame_left, "Negatyw", lambda: self.apply_func(lambda: self.processor.negative())).pack(fill=tk.X)

        f_params = tk.Frame(self.frame_left, bg=self.bg_color)
        f_params.pack(pady=5, fill=tk.X)
        
        self.entry_bright = tk.Entry(f_params, width=4, highlightbackground=self.bg_color)
        self.entry_bright.insert(0, "30")
        self.create_btn(f_params, "Jasność:", lambda: self.apply_func(lambda: self.processor.brightness(float(self.entry_bright.get())))).grid(row=0, column=0, sticky="ew")
        self.entry_bright.grid(row=0, column=1, padx=5)

        self.entry_contrast = tk.Entry(f_params, width=4, highlightbackground=self.bg_color)
        self.entry_contrast.insert(0, "1.5")
        self.create_btn(f_params, "Kontrast:", lambda: self.apply_func(lambda: self.processor.contrast(float(self.entry_contrast.get())))).grid(row=1, column=0, sticky="ew")
        self.entry_contrast.grid(row=1, column=1, padx=5)

        self.entry_bin = tk.Entry(f_params, width=4, highlightbackground=self.bg_color)
        self.entry_bin.insert(0, "128")
        self.create_btn(f_params, "Binaryzacja:", lambda: self.apply_func(lambda: self.processor.binarize(float(self.entry_bin.get())))).grid(row=2, column=0, sticky="ew")
        self.entry_bin.grid(row=2, column=1, padx=5)

        tk.Label(self.frame_left, text="--- Maski Filtrów ---", bg=self.bg_color).pack(pady=(15,5))
        f_presets = tk.Frame(self.frame_left, bg=self.bg_color)
        f_presets.pack(fill=tk.X)
        self.create_btn(f_presets, "Uśred", lambda: self.fill_kernel([[1,1,1],[1,1,1],[1,1,1]])).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.create_btn(f_presets, "Gauss", lambda: self.fill_kernel([[1,4,1],[4,12,4],[1,4,1]])).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.create_btn(f_presets, "Ostry", lambda: self.fill_kernel([[0,-2,0],[-2,11,-2],[0,-2,0]])).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.kernel_entries = []
        f_kernel = tk.Frame(self.frame_left, bg=self.bg_color)
        f_kernel.pack(pady=5)
        for i in range(3):
            row = []
            for j in range(3):
                e = tk.Entry(f_kernel, width=3, justify="center", highlightbackground=self.bg_color)
                e.insert(0, "1" if i==1 and j==1 else "0")
                e.grid(row=i, column=j, padx=2, pady=2)
                row.append(e)
            self.kernel_entries.append(row)
        self.create_btn(self.frame_left, "Zastosuj Filtr", self.apply_custom_filter).pack(fill=tk.X)

        tk.Label(self.frame_left, text="--- Krawędzie ---", bg=self.bg_color).pack(pady=(15,5))
        self.create_btn(self.frame_left, "Krzyż Robertsa", lambda: self.apply_func(lambda: self.processor.roberts())).pack(fill=tk.X)
        self.create_btn(self.frame_left, "Operator Sobela", lambda: self.apply_func(lambda: self.processor.sobel())).pack(fill=tk.X)

    def create_btn(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, highlightbackground=self.bg_color, bd=0)

    def load_image(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.processor = ImageProcessor(filepath)
            self.update_canvas()

    def save_image(self):
        if not self.processor: return
        filepath = filedialog.asksaveasfilename(defaultextension=".png")
        if filepath:
            img = self.processor.get_image()
            img.save(filepath)

    def reset_image(self):
        if not self.processor: return
        self.processor.reset()
        self.update_canvas()

    def apply_func(self, func):
        if not self.processor: return
        func()
        self.update_canvas()

    def fill_kernel(self, values):
        for i in range(3):
            for j in range(3):
                self.kernel_entries[i][j].delete(0, tk.END)
                self.kernel_entries[i][j].insert(0, str(values[i][j]))

    def apply_custom_filter(self):
        if not self.processor: return
        try:
            kernel = [[float(e.get()) for e in row] for row in self.kernel_entries]
            self.apply_func(lambda: self.processor.applyFilter(kernel))
        except ValueError:
            messagebox.showerror("Error", "Mask must contain only numbers.")

    def update_canvas(self):
        img = self.processor.get_image()
        img.thumbnail((450, 450))
        self.disp_width, self.disp_height = img.size
        self.image_tk = ImageTk.PhotoImage(img)
        self.canvas_img.config(width=self.disp_width, height=self.disp_height)
        self.canvas_img.delete("all")
        self.canvas_img.create_image(self.disp_width//2, self.disp_height//2, image=self.image_tk)
        
        # changing histogram position based on aspect ratio
        self.frame_hist.grid_forget()
        if self.disp_height > self.disp_width:
            self.frame_hist.grid(row=1, column=2, padx=(10,10), pady=(5,0), sticky="nw")
        else:
            self.frame_hist.grid(row=2, column=0, padx=(10,0), pady=(10,10), sticky="nw")

        self.root.update_idletasks()

        self.update_plots()

    def update_plots(self):
        if not self.processor: return
        self.draw_histogram()
        self.draw_projections()

    def draw_histogram(self):
        self.fig_hist.clf() 
        ax_hist = self.fig_hist.add_axes([0.1, 0.15, 0.85, 0.8]) 
        
        is_gray, pixels = self.processor.get_histograms()
        kwargs = dict(bins=256, range=(0, 256), alpha=0.6, histtype='stepfilled')

        if is_gray:
            ax_hist.hist(pixels[..., 0].flatten(), color='gray', **kwargs)
        else:
            if self.show_r.get(): ax_hist.hist(pixels[..., 0].flatten(), color='red', **kwargs)
            if self.show_g.get(): ax_hist.hist(pixels[..., 1].flatten(), color='green', **kwargs)
            if self.show_b.get(): ax_hist.hist(pixels[..., 2].flatten(), color='blue', **kwargs)
            
        ax_hist.set_xlim(0, 255)
        ax_hist.set_yticks([]) 
        ax_hist.grid(True, linestyle='--', alpha=0.5)
        self.fig_hist.patch.set_facecolor('white')
        
        self.canvas_hist.draw()

    def draw_projections(self):
        proj_h, proj_v = self.processor.get_projections()
        
        max_h_val = self.processor.height * 255 
        max_v_val = self.processor.width * 255

        # horizontal projection
        self.fig_ph.clf()
        ax_ph = self.fig_ph.add_axes([0, 0, 1, 1]) 
        ax_ph.bar(range(len(proj_h)), proj_h, color='black', width=1.0)
        ax_ph.set_xlim(0, len(proj_h))
        ax_ph.set_ylim(0, max_h_val)
        ax_ph.axis('off')
        self.fig_ph.patch.set_facecolor('white')
        
        self.canvas_ph.get_tk_widget().config(width=self.disp_width, height=120)
        self.canvas_ph.draw()

        # vertical projection
        self.fig_pv.clf()
        ax_pv = self.fig_pv.add_axes([0, 0, 1, 1]) 
        y_axis = np.arange(len(proj_v))
        ax_pv.barh(y_axis, proj_v, color='black', height=1.0)
        
        ax_pv.set_ylim(len(proj_v), 0) 
        ax_pv.set_xlim(0, max_v_val)
        ax_pv.axis('off')
        self.fig_pv.patch.set_facecolor('white')
        
        self.canvas_pv.get_tk_widget().config(width=120, height=self.disp_height)
        self.canvas_pv.draw()
