import tkinter as tk 


def main_window(root):
    '''Create the main window content'''

    frame1 = tk.LabelFrame(root, bg='lightgray', text='Frame 1')
    frame2 = tk.LabelFrame(root, bg='lightgrey', text='Frame 2')
    frame1.pack(side='top', expand=True, fill='both')
    frame2.pack(side='top', expand=True, fill='both')

    text = tk.Label(frame1, text='Test', bg='yellow', fg='red')
    text.pack(side='top')

    button_refresh = tk.Button(frame2, text='Refresh', bg='green', fg='white', command=root.update)
    button_refresh.pack(side='left', expand=True, fill='both')
    button_quit = tk.Button(frame2, text='Quit', bg='red', fg='white', command=root.quit)
    button_quit.pack(side='left', expand=True, fill='both')

    return root