import cv2 as cv
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


class SelectImage(Frame):
    img = None

    def __init__(self, master=None):
        Frame.__init__(self, master)
        w, h = 1000, 900
        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)



        self.grid(row=10, column=10)
        self.file = Button(self, text='Browse', command=self.select, width=22, height=1)
        self.choose = Label(self, text='Choose an image').grid(row=9, column=10)
        self.label = Label(image=None)
        self.logs = Label(self, text='No operation!!!')
        
        ##Filters
        self.change_contrast = Button(self, text='Change contrast', command=self.change_contrast, width=22, height=1)
        self.log_transformation = Button(self, text='Log Transformation', command=self.log_transformation, width=22, height=1)
        self.negative_transformation = Button(self, text='Negative Transformation', command=self.negative_transformation, width=22, height=1)
        self.power_low_transformation = Button(self, text='Power-Low Transformation', command=self.power_low_transformation, width=22, height=1)
        self.reduce_gray= Button(self, text='Reducing gray levels', command=self.reduce_gray, width=22, height=1)
        self.thresholding =  Button(self, text='Thresholding', command=self.thresholding, width=22, height=1)
        self.med = Button(self, text='med_min_max', command=self.med, width=22, height=1)
        self.lpfilter2dim = Button(self, text='Lowpass filter', command=self.lpfilter2dim,  width=22, height=1)
        self.hpfilter2dim = Button(self, text='highpass filter', command=self.hpfilter2dim,  width=22, height=1)
        self.sobel = Button(self, text='Sobel Edge Detection', command=self.sobel, width=22, height=1)
        self.erosion = Button(self, text='Erosion', command=self.erosion, width=22, height=1)
        self.dilation = Button(self, text='Dilation', command=self.dilation, width=22, height=1)

        self.file.grid(row=10, column=10)
        self.label.grid(row=14, column=15)
        self.change_contrast.grid(row=12, column=10)
        self.log_transformation.grid(row=13, column=10)
        self.negative_transformation.grid(row=14, column=10)
        self.power_low_transformation.grid(row=15, column=10)
        self.reduce_gray.grid(row=16, column=10)
        self.thresholding.grid(row=17, column=10)
        self.med.grid(row=18, column=10)
        self.lpfilter2dim.grid(row=21, column=10)
        self.hpfilter2dim.grid(row=22, column=10)
        self.sobel.grid(row=23, column=10)
        self.erosion.grid(row=24, column=10)
        self.dilation.grid(row=25, column=10)
        self.logs.grid(row=27, column=10)
        

        #Icon and Title
        self.file2 = Button(self, text='Choose icon', command=self.select2, width=22, height=1)
        self.file2.grid(row=26, column=10)
        root.title("GUI")


        #Menu
        self.my_menu= Menu(root)
        root.config(menu=self.my_menu)
        self.file_menu = Menu(self.my_menu)
        self.my_menu.add_cascade(label='File', menu=self.file_menu)
        # self.file_menu.add_command(label='Undo', command= )
        self.file_menu.add_command(label='Exit', command=root.quit)

        #labels for cordinatig
        mylabel1 = Label(self, text='                        ').grid(row=15, column=11)
        mylabel2 = Label(self, text='                        ').grid(row=15, column=12)


    def select(self):
        ifile = filedialog.askopenfile(parent=self, mode='rb', title='Choose a file', filetypes=[('Image Files', ['.jpeg', '.jpg', '.png', '.gif', '.tiff', '.tif', '.bmp'])])
        self.path = Image.open(ifile).convert('RGB').resize((256,  256))
        
        self.image = ImageTk.PhotoImage(self.path)
        self.label.configure(image=self.image)
        self.label.image = self.image
        self.img = np.array(self.path)
        self.img = self.img[:, :, :].copy()

    def select2(self):
        ifile2 = filedialog.askopenfile(parent=self, mode='rb', title='Choose a file', filetypes=[('Image Files', ['.ico'])])
        self.path2 = Image.open(ifile2)
        
        self.icon = ImageTk.PhotoImage(self.path2)
        root.iconphoto(False, self.icon)



    def change_contrast(self):
        if not np.any(self.img):
            return 

        img_after = change_contrast(self.img, 1.5, 10)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='Contrast changed')

    def log_transformation(self):
        if not np.any(self.img):
            return

        img_after = log_transformation(self.img)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='Log transformed')

    def negative_transformation(self):
        if not np.any(self.img):
            return

        img_after = negative_transformation(self.img)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='Negative transformation')

    def power_low_transformation(self):
        if not np.any(self.img):
            return

        img_after = power_low_transformation(self.img)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='power_low_transformation')

    def reduce_gray(self):
        if not np.any(self.img):
            return

        img_after = reducing_gray_level(self.img)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after 
        self.logs.config(text='reducing_gray')

    def thresholding(self):
        if not np.any(self.img):
            return 
        
        img_after = thresholding(self.img)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='Thresholding')

    def med(self):
        if not np.any(self.img):
            return

        img_after = med(self.img[:, :, 0])
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='median_salt and peper')


    def lpfilter2dim(self):
        if not np.any(self.img):
            return
            
        img_after = cv.filter2D(self.img, -1, lowpass_kernel)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='Low pass filter')

    def hpfilter2dim(self):
        if not np.any(self.img):
            return

        img_after = cv.filter2D(self.img, -1, highpass_kernel)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='High pass filter')

    def sobel(self):
        if not np.any(self.img):
            return

        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        # sobelx = cv.Sobel(src=self.img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
        # sobely = cv.Sobel(src=self.img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) 
        img_after = cv.Sobel(src=self.img, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='Sobel filter')

    def erosion(self):
        if not np.any(self.img):
            return

        kernel = np.ones((5,5), np.uint8)
        img_after = cv.erode(self.img, kernel, iterations=1)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='Implement erosion')

    def dilation(self):
        if not np.any(self.img):
            return

        kernel = np.ones((5,5), np.uint8)
        img_after = cv.dilate(self.img, kernel, iterations=1)
        img_after = ImageTk.PhotoImage(Image.fromarray(img_after))
        self.label.configure(image=img_after)
        self.label.image = img_after
        self.logs.config(text='Implement dilation')



def change_contrast(img, alpha, beta):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                img[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)
    return img


def log_transformation(img):
    c = 255 / (np.log(1 + np.max(img)))
    log_transformed = c + np.log(1 + img)
    log_transformed = np.array(log_transformed, dtype=np.uint8)
    return log_transformed



def negative_transformation(img):
    height, width, _ = img.shape

    for i in range(0, height -1):
        for j in range(0, width -1):
            pixel = img[i, j]

            pixel[0] = 255 - pixel[0]
            pixel[1] = 255 - pixel[1]
            pixel[2] = 255 - pixel[2]

            img[i, j] = pixel
    return img


def power_low_transformation(img):
    for gamma in [0.1, 0.5, 1.2, 2.2]:
        gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    return gamma_corrected


def reducing_gray_level(img):
    width = int(img.shape[0])
    height = int(img.shape[1])

    for i in range(width):
        for j in range(height):
            img[i][j] = img[i][j] * 2
    return img

def thresholding(image, threshold=120):
    image[image<threshold] = 0
    return image


def calc_cdf(histogram):
    cdf = histogram.cumsum()
    cdf = cdf / float(cdf[-1])
    return cdf

def equalize(image):
    high_contrast = np.zeros(image.shape)
    image_hist, bin0 = np.histogram(image.ravel(), bins=256)
    image_cdf = calc_cdf(image_hist)
    height, width = image.shape[:2]
    for y in range(0, height):
        for x in range(0, width):
            high_contrast[y,x] = image_cdf[image[y, x]]

    return high_contrast

def plot_bar(o_img, eq_img):
    plt.subplot(211), plt.bar(range(256), np.histogram(o_img.ravel(), bins=256)[0])
    plt.xlabel('Intensity Values')
    plt.ylabel('Pixel Count')

    plt.subplot(212), plt.bar(range(256), np.histogram(eq_img.ravel(), bins=256)[0])
    plt.xlabel('Intensity Values')
    plt.ylabel('Pixel Count')
    plt.show()



def med(image):
    m = image.shape[0]
    n = image.shape[1]
    image_new = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [image[i-1, j-1],
            image[i-1, j],
            image[i-1, j+1],
            image[i, j-1],
            image[i, j],
            image[i, j+1],
            image[i+1, j-1],
            image[i+1, j],
            image[i+1, j+1]]
            temp = sorted(temp)
            image_new[i, j] = temp[4]
        
    image_new = image_new.astype(np.uint8)
    return image_new


lowpass_kernel = np.ones([3, 3], dtype= int)
lowpass_kernel = lowpass_kernel / 9

highpass_kernel = np.array([[0, -1, 0],
[-1, 5, -1],
[0, -1, 0]])


def padding(image, kernel):
    m = image.shape[0]
    n = image.shape[1]
    km = kernel.shape[0]
    kn = kernel.shape[1]
    padded_img = np.zeros([m + km-1, n + kn-1])
    for i in range(m):
        for j in range(n):
            padded_img[i+int((km-1)/2), j+int((kn-1)/2)] = image[i, j]
    return padded_img
    

def filter2dim(image):
    kernel = np.ones([3, 3], dtype= int)
    kernel = lowpass_kernel / 9
    m = image.shape[0]
    n = image.shape[1]
    z = image.shape[2]
    image_new = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            for k in range(1, z-1):
                temp = image[i-1, j-1, k]*kernel[0,0] + image[i-1, j, k]*kernel[0,1] + image[i-1, j+1, k]*kernel[0,2]
                temp = temp + image[i, j-1, k]*kernel[1,0] + image[i, j, k]*kernel[1,1] + image[i, j+1, k]*kernel[1,2]
                temp = temp + image[i+1, j-1, k]*kernel[2,0] + image[i+1, j, k]*kernel[2,1] + image[i+1, j+1, k]*kernel[2,2]
                image_new[i, j] = temp
    # image_new = image_new.astype(np.uint8)
    return image_new





root = Tk()
app = SelectImage(master=root)
app.mainloop()