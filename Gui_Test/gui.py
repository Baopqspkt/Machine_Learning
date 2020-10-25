#!/usr/bin/env python3
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
from time import strftime 

canvas_width = 200
canvas_height = 200

tuple_1 = tuple(range(1, 25))
brt = len(tuple_1)

class my_gui:
    def __init__(self,master):
        self.init_gui(master)
        self.init_canvas(master)
        self.init_button(master)
        

    def init_canvas(self, master):
        self.w = Canvas(master, 
           width=150, 
           height=150,bg='green',)
        self.w.pack(expand = YES, fill = BOTH)
        self.w.bind( "<B1-Motion>", self.paint )


    def init_gui(self, master):
        self.master = master
        self.master.title = ("Creat By Bpham")

        self.style = Style(self.master)
        self.style.layout('text.Horizontal.TProgressbar',
             [('Horizontal.Progressbar.trough',
               {'children': [('Horizontal.Progressbar.pbar',
                              {'side': 'left', 'sticky': 'ns'})],
                'sticky': 'nswe'}),
              ('Horizontal.Progressbar.label', {'sticky': ''})])
        self.style.configure('TProgressbar', text='0 %',bg='green',foreground = 'green')

        self.label = Label(master, text="Creat By Bpham")
        self.label.pack()
        self.clock = Label(master, font = ('calibri', 27, 'bold'), 
            background = 'purple', 
            foreground = 'white') 
        self.clock.pack(anchor = 'center') 

        self.progressBar = Progressbar(master, style='text.Horizontal.TProgressbar', length=300,
                              maximum=brt, value=0)
        self.progressBar.pack(side=BOTTOM)

    def init_button(self,master):
        self.greet_button = Button(master, text="Greet", command=self.greet)
        self.greet_button.place(x=260, y=200, anchor="center")

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.place(x=260, y=230, anchor="center")
        # self.close_button.pack()

        self.progress_button = Button(master, text="start", command=self.progress_bar_func)
        self.progress_button.place(x=260, y=260, anchor="center")
        # self.progress_button.pack()

    def greet(self):
        print("Greetings!")

    def time(self): 
        self.string = strftime('%H:%M:%S %p') 
        self.clock.config(text = self.string) 
        self.clock.after(1000, self.time) 

    def progress_bar_func(self):
        global num
        num = 1
        self.master.after(500, self.update_progress_bar)

    def message_box(self):
        messagebox.showinfo("NOTICE","OPEN IMAGE DONE")

    def update_progress_bar(self):
        global num

        if num <= brt:
            percentage = round(num/brt*100)  # Calculate percentage.
            #print(num, percentage)
            self.progressBar['value'] = num
            self.style.configure('TProgressbar',
                        text='{:g} %'.format(percentage),bg='green',foreground = 'green', length=300)
            num += 1
            if num > brt:
                print('Done')
                self.message_box()
            else:
                self.master.after(200,self. update_progress_bar)

    def paint(self,event ):
        python_green = "#476042"
        x1, y1 = ( event.x - 1 ), ( event.y - 1 )
        x2, y2 = ( event.x + 1 ), ( event.y + 1 )
        self.w.create_rectangle( x1, y1, x2, y2, fill = python_green, width = 5 )
 

def main():
    root = Tk()
    root.minsize(width=300, height=300)
    root.maxsize(width=300, height=300)
    
    gui = my_gui(root)
    gui.time()
    root.mainloop()
    
if __name__ == '__main__':
    main()


