import tkinter as tk
from app import App


if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#e0e0e0")
    app = App(root)
    root.mainloop()