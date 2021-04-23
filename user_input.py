import tkinter
from tkinter import *
from tkinter import filedialog, ttk
import tkinter.messagebox
from PIL import Image, ImageTk, ImageDraw, ImageGrab
import cv2
import numpy as np
import math

class StrokeCanvas():
    def __init__(self):
        self.master = Tk()
        self.im = None
        self.bg_color = "black"
        self.canvas = Canvas(self.master, bg=self.bg_color)
        self.pen_color = 'red'
        self.pen_thickness = 5
        self.export_canvas = None
        self.export_drawer = None
        self.x0 = None
        self.y0 = None
        self.drawings = []
        self.r = {}
        self.g = {}
        self.b = {}
        self._id = None
        self.shape0 = None
        self.set_dims()
        self.create_widgets()
        
    def change_thickness(self, e):
        self.pen_thickness = int(e)

    def draw(self, e):
        if self.x0 and self.y0:
            if self.pen_color == 'red':
                if self.pen_thickness in self.r:
                    self.r[self.pen_thickness].append((self.x0, self.y0))
                    self.r[self.pen_thickness].append((e.x, e.y))
                else:
                    self.r[self.pen_thickness] = []
                    self.r[self.pen_thickness].append((self.x0, self.y0))
                    self.r[self.pen_thickness].append((e.x, e.y))
            elif self.pen_color == 'green':
                if self.pen_thickness in self.g:
                    self.g[self.pen_thickness].append((self.x0, self.y0))
                    self.g[self.pen_thickness].append((e.x, e.y))
                else:
                    self.g[self.pen_thickness] = []
                    self.g[self.pen_thickness].append((self.x0, self.y0))
                    self.g[self.pen_thickness].append((e.x, e.y))
            else:
                if self.pen_thickness in self.b:
                    self.b[self.pen_thickness].append((self.x0, self.y0))
                    self.b[self.pen_thickness].append((e.x, e.y))
                else:
                    self.b[self.pen_thickness] = []
                    self.b[self.pen_thickness].append((self.x0, self.y0))
                    self.b[self.pen_thickness].append((e.x, e.y))

            self.drawings.append(self.canvas.create_line(self.x0,self.y0, e.x, e.y, 
                                width=self.pen_thickness, fill=self.pen_color, capstyle=ROUND, smooth=True, joinstyle=ROUND))
        self.x0 = e.x
        self.y0 = e.y
    

    
    def clear_canvas(self):
        for line in self.drawings:
            self.canvas.delete(line)
        self.r = {}
        self.g = {}
        self.b = {}

    def reset(self, e):
        self.x0 = None
        self.y0 = None
        

    def save_image(self):
        for thickness, points in self.r.items():
            self.export_drawer.line(points, fill='red', width=thickness, joint='curve')
        for thickness, points in self.g.items():
            self.export_drawer.line(points, fill='green', width=thickness, joint='curve')
        for thickness, points in self.b.items():
            self.export_drawer.line(points, fill='blue', width=thickness, joint='curve')

        try:
            out_canvas = "results/canvas.png"
            self.export_canvas.save(out_canvas)
            mask = np.zeros((self.im.height(), self.im.width()))
            coords = self.canvas.coords(self._id)
            x_s = int(coords[0] - self.im.width()//2)
            if x_s < 0:
                x_s = 0
            x_f = x_s + self.im.width()

            y_s = int(coords[1] - self.im.height()//2)
            if y_s < 0:
                y_s = 0
            y_f = y_s + self.im.height()
            
            strokes = cv2.imread(out_canvas)
            mask[np.where(strokes[y_s:y_f, x_s:x_f])[:2]] = 1
            mask = np.dstack([mask, mask, mask])
            strokes = np.multiply(mask, strokes[y_s:y_f, x_s: x_f])
            im = ImageTk.getimage(self.im)
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2RGB)
            im_copy = im.copy()
            im_copy = strokes + cv2.cvtColor(np.multiply(np.logical_not(mask), im).astype(np.uint8), cv2.COLOR_BGR2RGB)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            out_image = cv2.addWeighted(im_copy.astype(np.uint8), 0.4, im.astype(np.uint8), 1 - 0.4, 0)
            print(self.shape0)
            r = self.shape0[0]/out_image.shape[0]
            out_image = cv2.resize(out_image, (self.shape0[1], self.shape0[0]), interpolation=cv2.INTER_AREA)
            cv2.imwrite('results/out_composite.jpg', out_image)
            tkinter.messagebox.showinfo("Success", "Image Saved")
        except Exception as e:
            print(e)
            tkinter.messagebox.showinfo("Error", "Image not Saved")
    def set_dims(self):
        self.master.config(width=2560, height=1600)
        self.master.update()
        self.canvas.config(width=2560, height=1600)
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>',self.draw)
        self.canvas.bind('<ButtonRelease-1>',self.reset)
        self.canvas.update()

    def select_image(self):
        # try:
        im = filedialog.askopenfilename()
        self.im = ImageTk.PhotoImage(Image.open(im))
        h = self.im.height()
        w = self.im.width()
        self.shape0 = [h, w]
        if self.im.height() > 1000:
            im = np.array(ImageTk.getimage(self.im))
            r = 1000/float(im.shape[0])
            dim = (int(im.shape[1]*r), 1000)
            self.im = ImageTk.PhotoImage(Image.fromarray(cv2.resize(np.array(im).astype(np.uint8), dim, interpolation=cv2.INTER_AREA)))
        self._id = self.canvas.create_image((self.canvas.winfo_screenwidth() + 250)//2, self.canvas.winfo_screenheight()//2, anchor=CENTER, image=self.im)
        self.canvas.update()
        self.export_canvas = Image.new('RGB', (2560, 1600), (0, 0, 0))
        self.export_drawer = ImageDraw.Draw(self.export_canvas)
        # except:
            # tkinter.messagebox.showinfo("Error", "No Image Selected")

    def choose_red(self):
	    self.pen_color = 'red'
    def choose_green(self):
	    self.pen_color = 'green'
    def choose_blue(self):
	    self.pen_color = 'blue'

    def create_widgets(self):
        settings_frame = Frame(self.master, bg='white', height=self.canvas.winfo_screenheight(), width=250)
        settings_window = self.canvas.create_window(0, 0, anchor=NW, window=settings_frame)

        image_button = TkinterCustomButton(text="Select Image", corner_radius=10, command=self.select_image, bg_color="white", text_font=("Avenir", 20), width=130, height=45)
        image_button_window = self.canvas.create_window(250/2, 100, anchor=CENTER, window=image_button)

        red = TkinterCustomButton(text="", width=50, height=50, fg_color="red", corner_radius=10, hover_color="red", bg_color="white", command=self.choose_red)
        red_window = self.canvas.create_window(250/2, 200, anchor=CENTER, window=red)

        green = TkinterCustomButton(text="", width=50, height=50, fg_color="green", corner_radius=10, hover_color="green", bg_color="white", command=self.choose_green)
        green_window = self.canvas.create_window(250/2 - 60, 200, anchor=CENTER, window=green)

        blue = TkinterCustomButton(text="", width=50, height=50, fg_color="blue", corner_radius=10, hover_color="blue", bg_color="white", command=self.choose_blue)
        blue_window = self.canvas.create_window(250/2 + 60, 200, anchor=CENTER, window=blue)
	
        clear_canvas = TkinterCustomButton(text="Clear Canvas", corner_radius=10, command=self.clear_canvas, bg_color="white", text_font=("Avenir", 20), width=130, height=45)
        clear_canvas_window = self.canvas.create_window(250/2, 300, anchor=CENTER, window=clear_canvas)

        thickness_slider = Scale(self.master, from_=5, to=100, length=200, orient=VERTICAL, command=self.change_thickness)
        # s2 = Scale(self.master, from_=0, to=100, tickinterval=100, sliderrelief='flat', orient="horizontal", highlightthickness=1, highlightcolor='red', background='white', fg='black', troughcolor='#2874A6', activebackground='grey', length=200)
        tw = self.canvas.create_window(250/2, 500, anchor=CENTER, window=thickness_slider)

        t = TkinterCustomButton(text="Export Image", corner_radius=10, command=self.save_image, bg_color="white", text_font=("Avenir", 20), width=130, height=45)
        tt = self.canvas.create_window(250/2, 700, anchor=CENTER, window=t)

class TkinterCustomButton(tkinter.Frame):
    """ tkinter custom button with border, rounded corners and hover effect
        Arguments:  master= where to place button
                    bg_color= background color, None is standard,
                    fg_color= foreground color, blue is standard,
                    hover_color= foreground color, lightblue is standard,
                    border_color= foreground color, None is standard,
                    border_width= border thickness, 0 is standard,
                    command= callback function, None is standard,
                    width= width of button, 110 is standard,
                    height= width of button, 35 is standard,
                    corner_radius= corner radius, 10 is standard,
                    text_font= (<Name>, <Size>),
                    text_color= text color, white is standard,
                    text= text of button,
                    hover= hover effect, True is standard,
                    image= PIL.PhotoImage, standard is None"""

    def __init__(self,
                 bg_color=None,
                 fg_color="#2874A6",
                 hover_color="#5499C7",
                 border_color=None,
                 border_width=0,
                 command=None,
                 width=120,
                 height=40,
                 corner_radius=10,
                 text_font=None,
                 text_color="white",
                 text="CustomButton",
                 hover=True,
                 image=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if bg_color is None:
            self.bg_color = self.master.cget("bg")
        else:
            self.bg_color = bg_color

        self.fg_color = fg_color
        self.hover_color = hover_color
        self.border_color = border_color

        self.width = width
        self.height = height

        if corner_radius*2 > self.height:
            self.corner_radius = self.height/2
        elif corner_radius*2 > self.width:
            self.corner_radius = self.width/2
        else:
            self.corner_radius = corner_radius

        self.border_width = border_width

        if self.corner_radius >= self.border_width:
            self.inner_corner_radius = self.corner_radius - self.border_width
        else:
            self.inner_corner_radius = 0

        self.text = text
        self.text_color = text_color
        if text_font is None:
            if sys.platform == "darwin":  # macOS
                self.text_font = ("Avenir", 13)
            elif "win" in sys.platform:  # Windows
                self.text_font = ("Century Gothic", 11)
            else:
                self.text_font = ("TkDefaultFont")
        else:
            self.text_font = text_font

        self.image = image

        self.function = command
        self.hover = hover

        self.configure(width=self.width, height=self.height)

        if sys.platform == "darwin" and self.function is not None:
            self.configure(cursor="pointinghand")

        self.canvas = tkinter.Canvas(master=self,
                                     highlightthicknes=0,
                                     background=self.bg_color,
                                     width=self.width,
                                     height=self.height)
        self.canvas.place(x=0, y=0)

        if self.hover is True:
            self.canvas.bind("<Enter>", self.on_enter)
            self.canvas.bind("<Leave>", self.on_leave)

        self.canvas.bind("<Button-1>", self.clicked)
        self.canvas.bind("<Button-1>", self.clicked)

        self.canvas_fg_parts = []
        self.canvas_border_parts = []
        self.text_part = None
        self.text_label = None
        self.image_label = None

        self.draw()

    def draw(self):
        self.canvas.delete("all")
        self.canvas_fg_parts = []
        self.canvas_border_parts = []
        self.canvas.configure(bg=self.bg_color)

        # border button parts
        if self.border_width > 0:

            if self.corner_radius > 0:
                self.canvas_border_parts.append(self.canvas.create_oval(0,
                                                                        0,
                                                                        self.corner_radius * 2,
                                                                        self.corner_radius * 2))
                self.canvas_border_parts.append(self.canvas.create_oval(self.width - self.corner_radius * 2,
                                                                        0,
                                                                        self.width,
                                                                        self.corner_radius * 2))
                self.canvas_border_parts.append(self.canvas.create_oval(0,
                                                                        self.height - self.corner_radius * 2,
                                                                        self.corner_radius * 2,
                                                                        self.height))
                self.canvas_border_parts.append(self.canvas.create_oval(self.width - self.corner_radius * 2,
                                                                        self.height - self.corner_radius * 2,
                                                                        self.width,
                                                                        self.height))

            self.canvas_border_parts.append(self.canvas.create_rectangle(0,
                                                                         self.corner_radius,
                                                                         self.width,
                                                                         self.height - self.corner_radius))
            self.canvas_border_parts.append(self.canvas.create_rectangle(self.corner_radius,
                                                                         0,
                                                                         self.width - self.corner_radius,
                                                                         self.height))

        # inner button parts

        if self.corner_radius > 0:
            self.canvas_fg_parts.append(self.canvas.create_oval(self.border_width,
                                                                self.border_width,
                                                                self.border_width + self.inner_corner_radius * 2,
                                                                self.border_width + self.inner_corner_radius * 2))
            self.canvas_fg_parts.append(self.canvas.create_oval(self.width - self.border_width - self.inner_corner_radius * 2,
                                                                self.border_width,
                                                                self.width - self.border_width,
                                                                self.border_width + self.inner_corner_radius * 2))
            self.canvas_fg_parts.append(self.canvas.create_oval(self.border_width,
                                                                self.height - self.border_width - self.inner_corner_radius * 2,
                                                                self.border_width + self.inner_corner_radius * 2,
                                                                self.height-self.border_width))
            self.canvas_fg_parts.append(self.canvas.create_oval(self.width - self.border_width - self.inner_corner_radius * 2,
                                                                self.height - self.border_width - self.inner_corner_radius * 2,
                                                                self.width - self.border_width,
                                                                self.height - self.border_width))

        self.canvas_fg_parts.append(self.canvas.create_rectangle(self.border_width + self.inner_corner_radius,
                                                                 self.border_width,
                                                                 self.width - self.border_width - self.inner_corner_radius,
                                                                 self.height - self.border_width))
        self.canvas_fg_parts.append(self.canvas.create_rectangle(self.border_width,
                                                                 self.border_width + self.inner_corner_radius,
                                                                 self.width - self.border_width,
                                                                 self.height - self.inner_corner_radius - self.border_width))

        for part in self.canvas_fg_parts:
            self.canvas.itemconfig(part, fill=self.fg_color, width=0)

        for part in self.canvas_border_parts:
            self.canvas.itemconfig(part, fill=self.border_color, width=0)

        # no image given
        if self.image is None:
            # create tkinter.Label with text
            self.text_label = tkinter.Label(master=self,
                                            text=self.text,
                                            font=self.text_font,
                                            bg=self.fg_color,
                                            fg=self.text_color)
            self.text_label.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

            # bind events the the button click and hover events also to the text_label
            if self.hover is True:
                self.text_label.bind("<Enter>", self.on_enter)
                self.text_label.bind("<Leave>", self.on_leave)

            self.text_label.bind("<Button-1>", self.clicked)
            self.text_label.bind("<Button-1>", self.clicked)

            self.set_text(self.text)

        # use the given image
        else:
            # create tkinter.Label with image on it
            self.image_label = tkinter.Label(master=self,
                                             image=self.image,
                                             bg=self.fg_color)

            self.image_label.place(relx=0.5,
                                   rely=0.5,
                                   anchor=tkinter.CENTER)

            # bind events the the button click and hover events also to the image_label
            if self.hover is True:
                self.image_label.bind("<Enter>", self.on_enter)
                self.image_label.bind("<Leave>", self.on_leave)

            self.image_label.bind("<Button-1>", self.clicked)
            self.image_label.bind("<Button-1>", self.clicked)

    def configure_color(self, bg_color=None, fg_color=None, hover_color=None, text_color=None):
        if bg_color is not None:
            self.bg_color = bg_color
        else:
            self.bg_color = self.master.cget("bg")

        if fg_color is not None:
            self.fg_color = fg_color

            # change background color of image_label
            if self.image is not None:
                self.image_label.configure(bg=self.fg_color)

        if hover_color is not None:
            self.hover_color = hover_color

        if text_color is not None:
            self.text_color = text_color
            if self.text_part is not None:
                self.canvas.itemconfig(self.text_part, fill=self.text_color)

        self.draw()

    def set_text(self, text):
        if self.text_label is not None:
            self.text_label.configure(text=text)

    def on_enter(self, event=0):
        for part in self.canvas_fg_parts:
            self.canvas.itemconfig(part, fill=self.hover_color, width=0)

        if self.text_label is not None:
            # change background color of image_label
            self.text_label.configure(bg=self.hover_color)

        if self.image_label is not None:
            # change background color of image_label
            self.image_label.configure(bg=self.hover_color)

    def on_leave(self, event=0):
        for part in self.canvas_fg_parts:
            self.canvas.itemconfig(part, fill=self.fg_color, width=0)

        if self.text_label is not None:
            # change background color of image_label
            self.text_label.configure(bg=self.fg_color)

        if self.image_label is not None:
            # change background color of image_label
            self.image_label.configure(bg=self.fg_color)

    def clicked(self, event=0):
        if self.function is not None:
            self.function()
            self.on_leave()

s = StrokeCanvas()
s.master.mainloop()