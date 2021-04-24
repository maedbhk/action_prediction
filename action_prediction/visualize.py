import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.express as px
from matplotlib.gridspec import GridSpec
# import cv2
from datetime import date

from action_prediction import constants as const

import warnings
warnings.filterwarnings("ignore")  

def plotting_style():
    plt.figure(num=2, figsize= [2,2], dpi=300)
    plt.style.use('seaborn-poster') # ggplot
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Helvetica Neue') 
    plt.rc('text', usetex='false') 
    plt.rcParams['lines.linewidth'] = 2
    plt.rc('xtick', labelsize=14)     
    plt.rc('ytick', labelsize=14)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["savefig.format"] = 'svg'
    plt.rcParams["savefig.bbox"] = 'tight'
    plt.rc("axes.spines", top=False, right=False) # removes certain axes

def _save_fig(plt, fname):
    fpath = os.path.join(const.FIG_DIR, fname)
    plt.savefig(fpath, bbox_inches= 'tight', pad_inches = .25, dpi= 100)

def _gaussian(x, sdx, y=None, sdy=None):
    
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution
    
    arguments
    -- width in pixels
    -- width standard deviation
    
    keyword argments
    -- height in pixels (default = x)
    -- height standard deviation (default = sdx)
    """
    
    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sdy == None:
        sdy = sdx
    
    x_mid = x/2
    y_mid = y/2
    # matrix of zeros
    M = np.zeros([y,x],dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j,i] = np.exp(-1.0 * (((float(i)-x_mid)**2/(2*sdx*sdx)) + ((float(j)-y_mid)**2/(2*sdy*sdy)) ) )
    
    return M

def _draw_display(dispsize=(1280, 720), img=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it
        Args: 
            dispsize (tuple or list): indicating the size of the display e.g. (1024,768)
        Kwargs: 
            img (np array): image array of size (n, p, 3). Default is None. 
        Returns: 
            (matplotlib.pyplot Figure and its axes): field of zeros
            with a size of dispsize, and an image drawn onto it
            if an image was passed
    """
    plt.figure()
    # construct screen (black background)
    screen = np.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    # if an image location has been passed, draw the image
    if img is not None:
        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = dispsize[0] / 2 - w / 2
        y = dispsize[1] / 2 - h / 2
        # draw the image on the screen
        # screen[y:y + h, x:x + w, :] += img
        screen[int(y):int(y)+h, int(x):int(x)+w, :] += img
    
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    # create a figure
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    # ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen.astype('uint8'))  # , origin='upper')

    return fig, ax

def _get_img(task='action_observation'):
    # get path to example video
    videofile = os.path.join(const.EYE_DIR, f'{task}.mp4')

    # open video and grab first image
    vidcap = cv2.VideoCapture(videofile)
    success, img = vidcap.read()

    return img

def _rescale_fixations(dataframe, dispsize):
    x = dataframe['mean_gx']
    dataframe['mean_gx'] = np.interp(x, (x.min(), x.max()), (0, dispsize[0]))

    y = dataframe['mean_gy']
    dataframe['mean_gy'] = np.interp(y, (y.min(), y.max()), (0, dispsize[1]))

    return dataframe

def visualize_corr(grid_table):
    data = grid_table.drop(["x", "y","total"], axis=1)
    correlations = data.corr()
    labels = correlations.columns
    corr_plot(correlations, labels)
    return correlations

def corr_plot(corr_mat, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_mat,cmap='coolwarm', vmin=0, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(labels),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.show()

def plot_fixation_count(dataframe, x='run_num', hue=None, hue_order=None, x_title = "", legend_title="", fig_title = None, save_title = None, palette = None):
    task = dataframe['task'].unique()[0]
    if hue:
        df = dataframe.groupby([x, hue, 'type', 'subj'])["type"].count().reset_index(name='count')
    else: 
        df = dataframe.groupby([x, 'type', 'subj'])['type'].count().reset_index(name="count")  
    plt.title(fig_title, loc = "left", pad= 40.0)
    plt.xlabel(x_title, labelpad = 20.0)
    plt.ylabel("Fixation Count", size=15, labelpad = 20.0)
    plt.xticks(rotation='45')
    plt.legend(title = legend_title)
    if x=='block_iter_corr':
        plt.xticks(rotation=45)
    elif x=='run_num':
        sns.factorplot(x=x, y='count', hue=hue, hue_order=hue_order, data=df.query('type=="fixations"'), legend=False) 
        plt.axvline(x=7, ymin=0, color='k', linestyle='--')
        plt.xticks(rotation='45', ticks= np.arange(1, 15, step=1), labels = np.arange(1, 15, step=1))
    else:
        plot_fix= sns.barplot(x=x, y='count', hue=hue, hue_order=hue_order, data=df.query('type=="fixations"'), capsize= .2, palette = palette)
        plot_fix.legend(bbox_to_anchor=(1, 1),loc='upper right', title = legend_title)
    
    plt.title(fig_title, loc = "left", pad= 40.0, size= 27)
    plt.xlabel(x_title, labelpad = 20.0, size = 27)
    plt.ylabel("Fixation Count", labelpad = 20.0, size = 27)
    plt.xticks(rotation='45', size = 20, ha= 'left')
    #plt.legend(
   # bbox_to_anchor=(1.30, 1.00), 
    #loc='upper right', title = legend_title)
    if x=='block_iter_corr':
        plt.xticks(rotation=45)
    

    #plt.tight_layout()
    _save_fig(plt, save_title)

def plot_saccade_count(dataframe, x='run_num', hue=None, x_title = "", fig_title = '', save_title = ""):
    task = dataframe['task'].unique()[0]
    if hue:
        df = dataframe.groupby([x, hue, 'type', 'subj'])["type"].count().reset_index(name='count')
    else:
         df = dataframe.groupby([x, 'type', 'subj'])['type'].count().reset_index(name="count")

    sns.lineplot(x=x, y='count', hue=hue, data=df.query('type=="saccade"'))
    plt.title(fig_title, loc = "left", pad= 40.0)
    plt.xticks(rotation='45', ticks= np.arange(1, 15, step=1), labels = np.arange(1, 15, step=1))
    plt.xlabel(x_title, labelpad = 20.0)
    plt.ylabel("Saccade Count", labelpad = 20.0);
    if x=='block_iter_corr':
        plt.xticks(rotation=45)
    elif x=='run_num':
        plt.axvline(x=7, ymin=0, color='k', linestyle='--')
    _save_fig(plt, save_title)

def plot_all_count(dataframe, x='run_num', hue=None, x_title = "", fig_title = "", save_title = " "):
    task = dataframe['task'].unique()[0]
    if hue:
        df = dataframe.groupby([x, hue,'subj'])["type"].count().reset_index(name='count')
    else:
         df = dataframe.groupby([x, 'subj'])['type'].count().reset_index(name="count")

    sns.barplot(x=x, y='count', hue=hue, data=df)
    plt.xticks(rotation='45');
    plt.xlabel(x_title, labelpad = 20.0)
    plt.ylabel("Average Count", size=15, labelpad = 20.0);
    plt.title(fig_title, loc = "left", pad= 40.0)
    if x=='block_iter_corr':
        plt.xticks(rotation=45)
    elif x=='run_num':
        plt.axvline(x=7, ymin=0, color='k', linestyle='--')
    plt.tight_layout()
    _save_fig(plt, save_title)


def plot_diameter(dataframe, x='run_num', event_type='fixations', hue=None):
    task = dataframe['task'].unique()[0]
    sns.factorplot(x=x, y='diameter', hue=hue, data=dataframe.query(f'type=="{event_type}"'))
    plt.xticks(rotation='45'); 
    plt.title(task)
    if x=='block_iter_corr':
        plt.xticks(rotation=45) 
    elif x=='run_num':
        plt.axvline(x=7, ymin=0, color='k', linestyle='--')

def plot_amplitude(dataframe, x='run_num', hue=None, x_title = "", fig_title = "", save_title = ""):
    task = dataframe['task'].unique()[0]
    sns.lineplot(x=x, y='amplitude', hue=hue, data=dataframe.query('type=="saccade"'))
    plt.title(fig_title, loc = "left", pad= 40.0)
    plt.xticks(rotation='45', ticks= np.arange(1, 15, step=1), labels = np.arange(1, 15, step=1))
    plt.xlabel(x_title, labelpad = 20.0)
    plt.ylabel('Amplitude', labelpad = 20.0)
    if x=='block_iter_corr':
        plt.xticks(rotation=45) 
    elif x=='run_num':
        plt.axvline(x=7, ymin=0, color='k', linestyle='--')
    _save_fig(plt, save_title)

def plot_fixation_duration(dataframe, x= "", hue=None, x_title = " ", fig_title = " ", save_title = "", legend_title = None, hue_order = None, palette= None):
    task = dataframe['task'].unique()[0]
    #sns.factorplot(x=x, y='duration', hue=hue, data=dataframe.query('type=="fixations"'), hue_order = hue_order)
    #if x=='block_iter_corr':
      #  plt.xticks(rotation=45) 
    if x=='run_num':
        sns.factorplot(x=x, y='duration', hue=hue, hue_order=hue_order, data= dataframe.query('type=="fixations"'))  
        plt.axvline(x=6, ymin=0, color='k', linestyle='--')
        plt.xticks(ticks= np.arange(0, 14, step=1), labels = np.arange(1, 15, step=1), ha= "left")

    else:
        plot_fix= sns.barplot(x=x, y='duration', hue=hue, hue_order=hue_order, data= dataframe.query('type=="fixations"'), capsize= .2, palette = palette)
        plot_fix.legend(bbox_to_anchor=(1.20, 1),loc='upper right', title = legend_title)
    
    plt.title(fig_title, loc = "left", pad= 40.0)
    plt.xticks(rotation='45', ha= "right")
    plt.xlabel(x_title, labelpad = 20.0)
    plt.ylabel('Fixation Duration (ms)', labelpad = 20.0)

    _save_fig(plt, save_title)

def plot_dispersion(dataframe, x='run_num', hue=None):
    task = dataframe['task'].unique()[0]
    sns.factorplot(x=x, y='dispersion', hue=hue, data=dataframe.query('type=="fixations"'))
    plt.xticks(rotation='45'); 
    plt.title(task)
    if x=='block_iter_corr':
        plt.xticks(rotation=45) 
    elif x=='run_num':
        plt.axvline(x=7, ymin=0, color='k', linestyle='--')

def plot_peak_velocity(dataframe, x='run_num', hue=None):
    task = dataframe['task'].unique()[0]
    sns.factorplot(x=x, y='peak_velocity', hue=hue, data=dataframe.query('type=="saccade"'))
    plt.xticks(rotation='45'); 
    plt.title(task)
    if x=='block_iter_corr':
        plt.xticks(rotation=45) 
    elif x=='run_num':
        plt.axvline(x=7, ymin=0, color='k', linestyle='--')

def plot_gaze_positions(dataframe, hue='task'):
    plt.figure()
    sns.kdeplot(x='mean_gx', y='mean_gy', data=dataframe, hue=hue, n_levels=50, shade=True, shadeLowest=False, alpha=0.5, legend_out=False)
    
    plt.title(f'gaze positions')
    # sns.scatterplot(x='mean_gx', y='mean_gy', hue=hue data=dataframe); 
    # plt.axis('equal');
    # plt.xlim(xmin=0, xmax=dispsize[0])
    # plt.ylim(ymin=0, ymax=dispsize[1])

def plot_heatmap(dataframe, dispsize=(1280, 720), img=None, alpha=0.5):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.
    
    arguments
    
    dataframe		-	a dataframe of fixation events 
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)
    
    keyword arguments
    
    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    durationweight	-	Boolean indicating whether the fixation duration is
                    to be taken into account as a weight for the heatmap
                    intensity; longer duration = hotter (default = True)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    
    returns
    
    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """
    # rescale fixations to disp size height and width
    dataframe = _rescale_fixations(dataframe=dataframe, dispsize=dispsize)

    fig, ax = _draw_display(dispsize, img=img)

    # HEATMAP
    # Gaussian
    gwh = 200
    gsdwh = gwh/6
    gaus = _gaussian(gwh,gsdwh)
    # matrix of zeroes
    strt = gwh/2
    heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0,len(dataframe['duration'])):
        # get x and y coordinates
        #x and y - indexes of heatmap array. must be integers
        x = int(strt) + int(dataframe['mean_gx'].iloc[i]) - int(gwh/2)
        y = int(strt) + int(dataframe['mean_gy'].iloc[i]) - int(gwh/2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj=[0,gwh]
            vadj=[0,gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x-dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y-dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y+vadj[1],x:x+hadj[1]] += gaus[vadj[0]:vadj[1],hadj[0]:hadj[1]] * dataframe['duration'].iloc[i]
            except:
                # fixation was probably outside of display
                pass
        else:				
            # add Gaussian to the current heatmap
            heatmap[y:y+gwh,x:x+gwh] += gaus * dataframe['duration'].iloc[i]
    # resize heatmap
    # heatmap = heatmap[int(strt):int(dispsize[1]+strt),int(strt):int(dispsize[0]+strt)]
    # remove zeros
    lowbound = np.mean(heatmap[heatmap>0])
    heatmap[heatmap<lowbound] = np.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    
    return fig

def plot_acc(dataframe, x='run_num', hue=None, x_title= " ", legend_title = None, fig_title = None, save_title = None, hue_order = None, palette = None):
    #acc_plot = sns.factorplot(x=x, y='corr_resp', hue=hue, hue_order = hue_order, data=dataframe, legend= False)   

    if x=='block_iter_corr':
        plt.xticks(rotation=45)
    elif x== 'run_num':
        sns.factorplot(x=x, y='corr_resp', hue=hue, hue_order = hue_order, data=dataframe, legend= False, palette = palette)
        plt.axvline(x=6, ymin=0, color='k', linestyle='--')
        plt.xticks(rotation='45', ticks= np.arange(0, 14, step=1), labels = np.arange(1, 15, step=1), ha = "right")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='lower right', title = legend_title) 
    else: 
        acc_bar = sns.barplot(x=x, y='corr_resp', hue=hue, hue_order = hue_order, data=dataframe)   
        acc_bar.legend(bbox_to_anchor=(0, 1),loc='upper right', title = legend_title)
    
    plt.ylabel("Accuracy", labelpad = 20.0)
    plt.xlabel(x_title, labelpad = 20.0)
    plt.title(fig_title, loc = "left", pad= 40.0)
    plt.xticks(rotation='45')
    #plt.figtext(x=-.2, y= -.2, s= "Fig. 1: Accuracy across runs for easy vs. hard conditions. Dashed line denotes start of Session 2.", fontstyle = "italic")

    #plt.tight_layout()
    _save_fig(plt, save_title)

def plot_rt(dataframe, x='run_num', hue=None, hue_order=None, x_title= " ", legend_title = " ", fig_title = " ", save_title= ""):
  
    rt_plot= sns.factorplot(x=x,y='rt', hue=hue, hue_order=hue_order, data=dataframe.query('corr_resp==True'), legend=True)
        # rt_plot._legend.set_title(legend_title)
    plt.legend(
    bbox_to_anchor=(1, 1), 
    loc='upper right', title = legend_title) # bbox_to_anchor=(1.05, 1)
    plt.ylabel("Reaction Time (ms)", labelpad = 20.0)
    plt.xlabel(x_title, labelpad = 20.0)
    plt.title(fig_title, loc = "left", pad= 40.0)
    plt.xticks(rotation='45', ticks= np.arange(1, 15, step=1), labels = np.arange(1, 15, step=1))
    if x=='block_iter_corr':
        plt.xticks(rotation=45) 
    elif x=='run_num':
        plt.axvline(x=7, ymin=0, color='k', linestyle='--')
    #plt.tight_layout()
    _save_fig(plt, save_title)


 