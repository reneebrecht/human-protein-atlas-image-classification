import matplotlib.image as img  
import matplotlib.pyplot as plt

import xarray as xr
import numpy as np
import hvplot.xarray
import holoviews as hv

import panel as pn
import plotly.express as px
import pandas as pd


import os, random
#%matplotlib inline

#import warnings
#warnings.filterwarnings('ignore')

def display_imgs(x):
    columns = 4
    rows = 2
    fig, axs = plt.subplots(rows,columns, figsize=(columns*8, rows*8))
    axs = axs.ravel()
    for idx, a in enumerate(axs):
          a.axis('off')
          a.imshow((x[idx]*255).astype(int), cmap='Greys_r')
    #plt.show()

    return fig

dropdown_list = ['0.  Nucleoplasm',   
                '1.  Nuclear membrane',   
                '2.  Nucleoli',   
                '3.  Nucleoli fibrillar center',   
                '4.  Nuclear speckles',   
                '5.  Nuclear bodies',   
                '6.  Endoplasmic reticulum',   
                '7.  Golgi apparatus',   
                '8.  Peroxisomes',   
                '9.  Endosomes',   
                '10.  Lysosomes',   
                '11.  Intermediate filaments',   
                '12.  Actin filaments',   
                '13.  Focal adhesion sites',   
                '14.  Microtubules',   
                '15.  Microtubule ends',   
                '16.  Cytokinetic bridge',   
                '17.  Mitotic spindle',   
                '18.  Microtubule organizing center',   
                '19.  Centrosome',   
                '20.  Lipid droplets',   
                '21.  Plasma membrane',   
                '22.  Cell junctions',   
                '23.  Mitochondria',   
                '24.  Aggresome',   
                '25.  Cytosol',   
                '26.  Cytoplasmic bodies',   
                '27.  Rods & rings'] 

targ_list = []
targets = pd.read_csv('../data/train.csv')

def sel_png_plot(dir_in="../data/train/"):
    pred_protein = []
    str_list = ['_red.png', '_blue.png', '_green.png','_yellow.png']
    filename = random.choice(os.listdir(dir_in)).rpartition('_')
    
    target_val = targets.query('Id == @filename[0]').Target.str.split(' ').values[0]
    target_val = target_val[0] # just check the first answer
    targ_list.append(target_val)

    img_array = np.array([img.imread(dir_in+filename[0]+end_str) for end_str in str_list]).transpose([1,2,0])

    ds = xr.Dataset({'image': (('x', 'y','channel'), img_array)},
        coords={'x': np.arange(0,img_array.shape[0]), 'y': np.arange(0,img_array.shape[0]),
                'channel': ['red','green','blue','yellow']})

    img_array_gr= img_array.copy()
    img_array_gr[:,:,[1,3]] = img_array[:,:,[1,3]]*0
    img_array_gb= img_array.copy()
    img_array_gb[:,:,[0,3]] = img_array[:,:,[0,3]]*0
    img_array_yb = img_array.copy()
    img_array_yb[:,:,[0,1]] = img_array[:,:,[0,1]]*0

    img_array_pairs = np.array((img_array[:,:,0:3], img_array_gr[:,:,0:3], 
                                img_array_gb[:,:,0:3], img_array_yb[:,:,1:4]))
    ds_pairs = xr.Dataset({'image': (('pair','x', 'y','channel'), img_array_pairs)},
        coords={'x': np.arange(0,img_array.shape[0]), 'y': np.arange(0,img_array.shape[0]),
                'channel': ['red','green','blue'], 
                'pair':['rgb', 'rg', 'gb', 'yb'] })

    all_img = [img_array[:,:,0], img_array[:,:,1], img_array[:,:,2], img_array[:,:,3],
            img_array[:,:,0:3], img_array_gr[:,:,0:3], img_array_gb[:,:,0:3], img_array_yb[:,:,1:4]]

    return display_imgs(all_img)
#    return display_imgs_plotly(all_img)
#    return display_imgs_hv(ds, ds_pairs)

hv.extension('bokeh')
fig_container = pn.pane.Matplotlib(sel_png_plot())

text_list = []
button = pn.widgets.Button(name="Click me", button_type="primary")
def new_image(event):
    fig_container.loading = True
    fig = sel_png_plot()
    fig_container.object = fig
    fig_container.loading = False

def save_text(event):
    text_list.append(text_input.value)

text_input = pn.widgets.Select(name='Select protein', options=dropdown_list)
button2 = pn.widgets.Button(name="Save selection", button_type="primary")
button2.on_click(save_text)

forward = pn.widgets.Button(name='\u25b6', width=50)
forward.on_click(new_image)

sel_box = pn.WidgetBox('Selection', text_input, button2)
img_box = pn.WidgetBox('Image', fig_container, forward)
#pn.Row(pn.Column(fig_container, forward), pn.Column(text_input, button2), width=200).show()
pn.Column(sel_box, img_box).servable()
