import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.gridspec import GridSpec
import cv2
from datetime import date

from learning_connect.constants import Dirs, Defaults

import warnings
warnings.filterwarnings("ignore")

def load_data():
    # initialise defaults
    defaults = Defaults() 
    
    dirs = Dirs(session_type='behavioral')  

    # load data
    dataframe = pd.read_csv(os.path.join(dirs.EYE_DIR, 'pl_msgs_events_group.csv'))

    # recode some variables (easier for visualization)
    dataframe['subj'] = dataframe['subj'].apply(lambda x: defaults.subj_id[x]).str.extract('(\d+)')
    dataframe['run'] = dataframe['run'] + 1
    dataframe['sess'] = dataframe['sess'].str.extract('(\d+)') 

    # filter events
    dataframe = _filter_data(dataframe)

    return dataframe   

def _distance(row, center):
    return np.sqrt((row['mean_gx']-center[0])**2+(row['mean_gy']-center[1])**2)

def _process_fixations(dataframe):
    df = dataframe[dataframe['type']=='fixations']
    df = df.query('mean_gx>=0 and mean_gx<=1 and mean_gy>=0 and mean_gy<=1')
    duration = np.mean(df['duration'])
    center = (np.mean(df['mean_gx']), np.mean(df['mean_gy']))
    df['dispersion'] = df.apply(_distance, args=[center], axis=1)
    return df

def _process_saccades(dataframe):
    return dataframe.query(f'type=="saccade" and amplitude<1')

def _filter_data(dataframe):
    fix = _process_fixations(dataframe=dataframe)
    sacc = _process_saccades(dataframe=dataframe)
    blink = dataframe.query('type=="blink"')
    return pd.concat([fix, sacc, blink])

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

def _draw_display(dispsize=(1280, 768), img=None):
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
    # initialise directoreis
    dirs = Dirs(session_type='behavioral')

    # get path to example video
    videofile = os.path.join(dirs.EYE_DIR, f'{task}.mp4')

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

def plot_count_events(dataframe):
    df = dataframe.groupby(['type', 'task', 'event_type'])['type'].count().reset_index(name="count")
    
    # plot saccades
    plt.figure()
    sns.barplot(x='task', y='count', hue='event_type', data=df.query('type=="saccade"'))
    plt.xticks(rotation='45'); 
    plt.xlabel('')
    plt.ylabel("Number of Saccades", size=15);

    # plot dataframe
    plt.figure()
    sns.barplot(x='task', y='count', hue='event_type', data=df.query('type=="fixations"'))
    plt.xticks(rotation='45'); 
    plt.xlabel('')
    plt.ylabel("Number of Fixations", size=15);

    # plot blinks
    plt.figure()
    sns.barplot(x='task', y='count', hue='event_type', data=df.query('type=="blink"'))
    plt.xticks(rotation='45'); 
    plt.xlabel('')
    plt.ylabel("Number of Blinks", size=15);

def plot_gaze_positions(dataframe, event_type='task', hue='task'):
    plt.figure()
    sns.kdeplot(x='mean_gx', y='mean_gy', hue=hue, data=dataframe.query(f'event_type=="{event_type}"'), n_levels=50, shade=True, shadeLowest=False, alpha=0.5, legend_out=False)
    plt.title(f'gaze positions for {event_type}')
    # sns.scatterplot(x='mean_gx', y='mean_gy', hue='task', data=dataframe.query(f'event_type=="{event_type}"')); 
    plt.axis('equal');

def plot_heatmap(dataframe, dispsize=(1280, 768), img=None, alpha=0.5):
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

def generate_plots(dispsize=(1280, 768)):
    # get data
    dataframe = preprocess_data()

    # count events
    plot_count_events(dataframe)

    # fixation positions
    tasks = dataframe['task'].unique()
    for task in tasks: 

        # filter data
        df = dataframe.query(f'task=="{task}" and type=="fixations"')

        # kdeplot of fixation positions
        plot_gaze_positions(dataframe=df, event_type='task')

        # show background image for select tasks
        if task=='action_observation':
            img = _get_img(task=task)
        else:
            img = None

        # draw heatmap of fixation positions
        plot_heatmap(dataframe=df, 
                    dispsize=dispsize, 
                    img=img, 
                    alpha=0.5)


 