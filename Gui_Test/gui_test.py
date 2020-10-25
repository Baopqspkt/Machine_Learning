from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def print_path():
    f = filedialog.askopenfilename(
        parent=root, initialdir='C:/Tutorial',
        title='Choose file',
        filetypes=[('png images', '.png'),
                   ('jpg images', '.jpg'),]
        )

    print(f)
    load = Image.open(f)
    load = load.resize((300,400),Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x=0, y=0) 

root = Tk()
canvas = Canvas(root, width = 300, height = 400)      
canvas.pack()      

b1 = Button(root, text='Open Image', command=print_path)
b1.pack(fill='x')

root.mainloop()