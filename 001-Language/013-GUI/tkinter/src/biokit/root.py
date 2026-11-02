import tkinter as tk 
from windows import main_window


# Create the main application window
root = tk.Tk()
root.title("Biokit")
width, height, scaler = root.winfo_screenwidth(), root.winfo_screenheight(), 3
root.geometry(f"{int(width/scaler)}x{int(height/scaler)}")

# Call the function to create the main window content
root = main_window(root)

# Start the main event loop
root.mainloop()