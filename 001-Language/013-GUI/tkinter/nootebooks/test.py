import tkinter as tk 


root = tk.Tk()

root.title("Hello Tkinter")
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
scaler = 3
root.geometry(f"{int(width/scaler)}x{int(height/scaler)}")
root.resizable(False, False)

text = tk.Label(root, text='Test', bg='yellow', fg='red')
text.pack(side='top')

button_refresh = tk.Button(root, text='Refresh', bg='green', fg='white', command=root.update)
button_refresh.pack(side='left')
button_quit = tk.Button(root, text='Quit', bg='red', fg='white', command=root.quit)
button_quit.pack(side='bottom')

root.mainloop()