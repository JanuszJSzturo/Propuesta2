import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as tkfd

import glob
import pydicom

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

import numpy as np
from skimage import measure
from scipy import ndimage

def update_image(*args):
    """
    Update dicom with selected parameters
    :param args:
    :return:
    """
    slice_id = int(slice_index_var.get())
    #image = np.copy(dcms[slice_id].pixel_array)
    image = np.copy(dcms_images)
    image = image[slice_id,:,:]
    image_plot.clear()

    if windowing_var.get():
        range_w = windowing_range_var.get()
        center_w = windowing_center_var.get()

        min_w = center_w-range_w//2
        max_w = center_w+range_w//2
        min_mask = image < min_w
        max_mask = image > max_w
        image[min_mask] = min_w
        image[max_mask] = max_w

    if isocontour_var.get():
        contours = measure.find_contours(image, int(sel_pixel_value_var.get()))
        for contour in contours:
            image_plot.plot(contour[:, 1], contour[:, 0], linewidth=2, color='green')

    image_plot.imshow(image, cmap=plt.cm.bone)
    canvas.draw()

def windowing_update_image(*args):
    """
    We only need to update the plot in the canvas if the windowing contrast is activated
    :param args:
    :return:
    """
    if windowing_var.get():
        update_image()

# def load_dcm(path):
#     """
#     Function to load all *.dcm files in a specified path and configure widgets values
#     :param path: path to folder
#     :return: list with dicom files and 3D array with dicom images
#     """
#     dcm_files = []
#     dcm_images = []
#     content_time = []
#     instance_number = []
#     for file in glob.glob(path + "/*.dcm"):
#         dcm_file = pydicom.dcmread(file)
#         #content_time.append(dcm_file.ContentTime)
#         instance_number.append(dcm_file.InstanceNumber)
#         dcm_files.append(dcm_file)
#         dcm_images.append(dcm_file.pixel_array)
#
#     # Sort the dicom files by contentTime instead of name
#     sorted_index = np.argsort(np.array(instance_number))
#     dcm_sorted = np.array(dcm_files)[sorted_index]
#     dcm_images_sorted = np.array(dcm_images)[sorted_index]
#
#     windowing_range_selector.configure(to=np.array(dcm_images).max())
#     windowing_center_selector.configure(to=np.array(dcm_images).max())
#
#     # slice_spinbox.configure(to=len(dcm_files)-1)
#     # range_spinbox.configure(to=np.array(dcm_images).max())
#     # center_spinbox.configure(to=np.array(dcm_images).max())
#     # slice_selector.configure(to=len(dcm_files)-1)
#     header_selector.configure(to=len(dcm_files)-1)
#
#     build_tree(dicom_header_tree, dcm_files[0], '')
#     return dcm_sorted, dcm_images_sorted

def load_dcm(path):
    """
    Function to load all *.dcm files in a specified path and configure widgets values
    :param path: path to folder
    :return: list with dicom files and 3D array with dicom images
    """

    dcm_files = []
    instance_number = []
    dcm_images = []
    for file in glob.glob(path + '/*.dcm'):
        dcm_file = pydicom.dcmread(file)

        dcm_files.append(dcm_file)
        instance_number.append(dcm_file.InstanceNumber)
        dcm_images.append(dcm_file.pixel_array)

    if len(dcm_files) == 1:
        dcm_images = dcm_file.pixel_array
    elif len(dcm_files) > 1:
        sorted_index = np.argsort(np.array(instance_number))
        dcm_images = np.array(dcm_images)[sorted_index]
        dcm_files = np.array(dcm_files)[sorted_index]


    windowing_range_selector.configure(to=np.array(dcm_images).max())
    windowing_center_selector.configure(to=np.array(dcm_images).max())
    dcm_images = ndimage.zoom(dcm_images, (1, 0.5078, 0.5078))
    dcm_images = dcm_images[19:,:229, 34:227]
    print(dcm_images.shape)
    # dcm_images = ndimage.zoom(dcm_images, (1 / 2, 1 / 0.5078, 1 / 0.5078))
    # dcm_images = ndimage.zoom(dcm_images, (np.array((193, 229, 193))/np.array(dcm_images.shape)))
    # slice_spinbox.configure(to=len(dcm_files)-1)
    # range_spinbox.configure(to=np.array(dcm_images).max())
    # center_spinbox.configure(to=np.array(dcm_images).max())
    slice_selector.configure(to=dcm_images.shape[0]-1)
    header_selector.configure(to=len(dcm_files)-1)

    build_tree(dicom_header_tree, dcm_files[0], '')
    return dcm_files, dcm_images

def on_press(event):
    """
    Shows information of the las selected pixel
    :param event: event from matplotlib backend
    :return:
    """
    if (event.xdata is None) | (event.ydata is None):
        return

    x_pos = int(event.xdata)
    y_pos = int(event.ydata)
    slice_id = int(slice_index_var.get())

    pixel_value = dcms_images[slice_id, y_pos, x_pos]
    sel_pixel_value_var.set(str(pixel_value))
    sel_pixel_position_var.set(str(x_pos)+', '+str(y_pos))
    sel_pixel_slice_var.set(str(slice_id))

    if isocontour_var.get():
        update_image()


def prompt_ask_directory():
    """
    Asks user to select the directory that contains the dicom files
    :return:
    """
    path = tkfd.askdirectory()
    global dcms
    global dcms_images
    dcms, dcms_images = load_dcm(path)

    # Enable all widgets that need an opened file to work
    # header_spinbox.state(['!disabled'])
    # center_spinbox.state(['!disabled'])
    # range_spinbox.state(['!disabled'])
    # slice_spinbox.state(['!disabled'])
    windowing_checkbutton.state(['!disabled'])
    isocontour_checkbutton.state(['!disabled'])

    windowing_center_selector.configure(state='active')
    windowing_range_selector.configure(state='active')
    slice_selector.configure(state='active')
    header_selector.configure(state='active')
    update_image()


def build_tree(tree, dataset, parent, open_=True):
    """
    :param tree: ttk.Treeview object where to insert the elements of dataset
    :param dataset: elements to insert in tree
    :param parent: parent element iid that the elements of dataset are children of
    :param open_: specify if expanda parent or not
    """
    for data_element in dataset:
        node_id = hex(id(data_element))
        tree.insert(parent=parent, index=tk.END,
                    iid=node_id,
                    text=data_element.tag,
                    open=open_,
                    values=(data_element.name, data_element.VR, data_element.repval))
        if data_element.VR == 'SQ':
            for i, dataset in enumerate(data_element.value):
                item_id = node_id + '.' + str(i + 1)
                sq_item_description = data_element.name.replace('Sequence', '')
                item_text = f'{sq_item_description}, {i+1}'
                tree.insert(parent=node_id, index=tk.END, iid=item_id, text=item_text, open=open_)
                build_tree(tree, dataset, item_id, open_=True)


def update_header(*args):
    """
    Update the dicom header view for the current slice selected
    :param args:
    :return:
    """
    children = dicom_header_tree.get_children()
    if children:
        dicom_header_tree.delete(*children)
    build_tree(dicom_header_tree, dcms[header_index_var.get()], '')


# DEFINE ROOT TK OBJECT WHERE THE WIDGETS WILL BE ADDED
root = tk.Tk()
font_size_var = tk.IntVar(value=12)

# CONFIGURE ROOT
root.title('Simple DICOM visualizer')
root.geometry('977x846+300+300')
root.iconbitmap(default='images/dicom_icon.ico')
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

theme_var = tk.StringVar()

# DEFINE NOTEBOOK TO SEPARATE DICOM IMAGE AND DICOM HEADERS
notebook = ttk.Notebook(root)
notebook.grid(sticky='nwes', padx=5, pady=5)
notebook.enable_traversal()

#################################
# SUB-FRAME FOR THE DICOM IMAGE #---------------------------------------------------------------------------------------
#################################
dicom_exploration_frame = ttk.Frame(notebook)
dicom_exploration_frame.columnconfigure(1, weight=1)
dicom_exploration_frame.rowconfigure(0, weight=1)
notebook.add(dicom_exploration_frame, text='DICOM exploration')

image_labelFrame = ttk.LabelFrame(dicom_exploration_frame, text='DICOM image')
image_labelFrame.columnconfigure(0, weight=1)
image_labelFrame.rowconfigure(1, weight=1)
image_labelFrame.grid(row=0, column=1, sticky='nswe')

###################################
# EMBEDDING MATPLOTLIB IN TKINTER #
###################################
f = Figure(figsize=(5, 5), dpi=100)
image_plot = f.add_subplot(111)

canvas = FigureCanvasTkAgg(f, master=image_labelFrame)
canvas.draw()
canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')
canvas.mpl_connect("button_press_event", on_press)
toolbarFrame = tk.Frame(master=image_labelFrame)
toolbarFrame.grid(row=2, column=0, sticky=tk.S + tk.N + tk.E + tk.W)
toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)


slice_index_var = tk.DoubleVar()
slice_index_var.trace_add('write', update_image)
slice_labelFrame = tk.LabelFrame(image_labelFrame, text='Slice selector')
slice_labelFrame.columnconfigure(0, weight=1)
slice_labelFrame.rowconfigure(0, weight=1)
slice_labelFrame.grid(row=0, column=0, sticky='ew')

slice_selector = tk.Scale(slice_labelFrame,
                          from_=0,
                          to=100,
                          variable=slice_index_var,
                          orient=tk.HORIZONTAL,
                          resolution=1,
                          showvalue=0)
slice_selector.configure(state='disabled')
slice_selector.grid(row=0, column=0, sticky='ew')

# slice_spinbox = ttk.Spinbox(slice_labelFrame, from_=0, to=100, textvariable=slice_index_var, width=5)
# slice_spinbox.state(['disabled'])
# slice_spinbox.grid(row=0, column=1, sticky='we')

#########################
# VISUALIZATION OPTIONS #-----------------------------------------------------------------------------------------------
#########################
options_labelFrame = ttk.LabelFrame(dicom_exploration_frame, text='Options')
options_labelFrame.grid(row=0, column=0, sticky='nsw')

############################
# WINDOWING CONTRAST FRAME #
############################
windowing_frame = ttk.Labelframe(options_labelFrame, text='Windowing')
windowing_frame.grid(row=4, column=0, sticky='nsew')

windowing_var = tk.BooleanVar()
windowing_checkbutton = ttk.Checkbutton(windowing_frame, text='Contrast windowing',
                                        variable=windowing_var, command=update_image)
windowing_checkbutton.state(['disabled'])
windowing_checkbutton.grid(row=0, column=0)
windowing_range_var = tk.IntVar()
windowing_center_var = tk.IntVar()
windowing_center_var.trace_add('write', windowing_update_image)
windowing_range_var.trace_add('write', windowing_update_image)

range_labelframe = ttk.Labelframe(windowing_frame, text='Range')
range_labelframe.grid(row=1, column=0)
windowing_range_selector = tk.Scale(range_labelframe, from_=0, to=1000,
                                     variable=windowing_range_var, orient=tk.HORIZONTAL, resolution=1, showvalue=0)
windowing_range_selector.configure(state='disabled')
windowing_range_selector.grid(row=0, column=0)
# range_spinbox = ttk.Spinbox(range_labelframe, from_=0, to=100, textvariable=windowing_range_var, width=5)
# range_spinbox.state(['disabled'])
# range_spinbox.grid(row=0, column=1, sticky='we')


center_labelframe = ttk.Labelframe(windowing_frame, text='Center')
center_labelframe.grid(row=2, column=0)
windowing_center_selector = tk.Scale(center_labelframe, from_=0, to=1000,
                                      variable=windowing_center_var, orient=tk.HORIZONTAL, resolution=1, showvalue=0)
windowing_center_selector.configure(state='disable')
windowing_center_selector.grid(row=0, column=0)
# center_spinbox = ttk.Spinbox(center_labelframe, from_=0, to=100, textvariable=windowing_center_var, width=5)
# center_spinbox.state(['disabled'])
# center_spinbox.grid(row=0, column=1, sticky='we')

##############################
# SEGMENTATION BY ISOCONTOUR #
##############################
isocontour_labelframe = ttk.Labelframe(options_labelFrame, text='Isocontour')
isocontour_labelframe.grid(row=5, column=0, sticky='nsew')

isocontour_var = tk.BooleanVar()
isocontour_var.trace_add('write', update_image)
isocontour_checkbutton = ttk.Checkbutton(isocontour_labelframe, text='Activate isocontour',
                                         variable=isocontour_var)
isocontour_checkbutton.state(['disabled'])
isocontour_checkbutton.grid(row=0, column=0)
tk.Label(isocontour_labelframe, text='Last pixel selected', anchor='w').grid(row=1, column=0)
tk.Label(isocontour_labelframe, text='Value', anchor='w').grid(row=2, column=0, sticky='w')
tk.Label(isocontour_labelframe, text='Position', anchor='w').grid(row=3, column=0, sticky='w')
tk.Label(isocontour_labelframe, text='Slice', anchor='w').grid(row=4, column=0, sticky='w')

sel_pixel_value_var = tk.StringVar()
sel_pixel_position_var = tk.StringVar()
sel_pixel_slice_var = tk.StringVar()
sel_pixel_value = tk.Label(isocontour_labelframe, text=0, textvariable=sel_pixel_value_var, width=7)
sel_pixel_value.grid(row=2, column=1, sticky='w')
sel_pixel_position = tk.Label(isocontour_labelframe, text=0, textvariable=sel_pixel_position_var, width=7)
sel_pixel_position.grid(row=3, column=1, sticky='w')
sel_pixel_slice = tk.Label(isocontour_labelframe, text=0, textvariable=sel_pixel_slice_var, width=7)
sel_pixel_slice.grid(row=4, column=1, sticky='w')

###########################################
# SUB-FRAME FOR DICOM HEADERS INFORMATION #-----------------------------------------------------------------------------
###########################################
headers_frame = ttk.Frame(notebook)
headers_frame.columnconfigure(0, weight=1)
headers_frame.rowconfigure(1, weight=1)
notebook.add(headers_frame, text='DICOM headers')

# Create the treeview
dicom_header_tree = ttk.Treeview(headers_frame)
dicom_header_tree.grid(row=1, column=0, sticky='nswe')

# Specify column names and create the headings
tree_columns_dicom_header = ('Name', 'VR', 'Value')
dicom_header_tree['columns'] = tree_columns_dicom_header
for heading in tree_columns_dicom_header:
    dicom_header_tree.heading(heading, text=heading)

# Configure columns
dicom_header_tree.heading('#0', text='Tag')
dicom_header_tree.column('#0', anchor='w', minwidth=100, stretch=False, width=150)
dicom_header_tree.column('Name', anchor='w', minwidth=100, stretch=False, width=120)
dicom_header_tree.column('VR', anchor='center', minwidth=40, stretch=False, width=40)
dicom_header_tree.column('Value', anchor='w', minwidth=100, stretch=True, width=120)

# Add vertical scrollbar to treeview
tree_view_scrollbar = ttk.Scrollbar(headers_frame, orient='vertical', command=dicom_header_tree.yview)
tree_view_scrollbar.grid(row=1, column=1, sticky='nse')
dicom_header_tree.configure(yscrollcommand=tree_view_scrollbar.set)

# Add header dicom selector as in slice dicom selector
header_index_var = tk.IntVar()
header_index_var.trace_add('write', update_header)
header_labelFrame = tk.LabelFrame(headers_frame, text='Header selector')
header_labelFrame.columnconfigure(0, weight=1)
header_labelFrame.rowconfigure(0, weight=1)
header_labelFrame.grid(row=0, column=0, sticky='ew')

header_selector = tk.Scale(header_labelFrame, from_=0, to=100, variable=header_index_var, orient=tk.HORIZONTAL, resolution=1, showvalue=0)
header_selector.configure(state='disabled')
header_selector.grid(row=0, column=0, sticky='ew')
# header_spinbox = ttk.Spinbox(header_labelFrame, from_=0, to=100, textvariable=header_index_var, width=5)
# header_spinbox.state(['disabled'])
# header_spinbox.grid(row=0, column=1, sticky='we')


#########
# MENUS #---------------------------------------------------------------------------------------------------------------
#########
menu = tk.Menu(root)
root.configure(menu=menu)

file_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label='File', underline=0, menu=file_menu)
file_menu.add_command(label='Open', command=prompt_ask_directory, underline=0)
file_menu.add_separator()
file_menu.add_command(label='Exit', command=root.destroy)


root.mainloop()
