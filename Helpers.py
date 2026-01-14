# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:08:44 2022

@author: wiktor.buchholz
"""
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.parse

import win32com.client #needed for shortcut creation

# =============================================================================
# FILES HANDLING 
# =============================================================================
def get_files(path, extension = None):
    """Retruns list of files in the given path with given extension (if passed to the function)"""
    files = []
    if not os.path.isdir(path):
        raise ValueError(f"path {path} doesn't exist!")
    if extension:
        extension = str(extension)  #force conversion 
        for file in os.listdir(path):
                if file.endswith(extension):
                    files.append(os.path.join(path, file))
                    
    else:
        for file in os.listdir(path):
            files.append(os.path.join(path, file))
        
    if not files:
        raise ValueError(f"No {extension} files found in: {path}")
        
    if len(files)==1:
        return files[0]

    return files    

def create_public_shortcut(name='Shortcut to public folder',
                           destination=None):
    """Create shortcut to (public) folder and create there a folder with the same name as cwd (if it doesn't exist yet)
    
    name: string 
        Shortcut's name
    destination: string, optional, Default None
        Shortcut's destination folder. If None, the shortcut is created in the current folder 
    
    Return: boolean, True if successful
    """
    PUBLIC_FOLDER = 'W:\PD-Engines\Engine Integration Validation\Public\Data Analytics'
    
    print('Deploying shortcut {}...'.format(name))
    if not destination:
        destination = os.getcwd()

    if not name.lower().endswith('.url'):
        name = '{}.url'.format(name)

    path = os.path.join(destination, name)
    
    link = os.path.join(PUBLIC_FOLDER, os.path.basename(os.getcwd()))
    if not os.path.isdir(link): #if folder not exist yet
        os.mkdir(link) #create one
        print(f'Folder {link} created')
            
    # Create link to public directory in the current folder
    try:
        ws = win32com.client.Dispatch("wscript.shell")
        shortcut = ws.CreateShortCut(path)
        shortcut.TargetPath = link
        shortcut.Save()
    except Exception as exception:
        error = 'Failed to deploy shortcut! {}\nArgs: {}, {}, {}'.format(exception, name, link, destination)
        print(error)
        return False
    
    # Create link in public folder to current dir
    try:
        pub_shortcut_name = 'Shortcut to DAR folder.url'
        link = os.path.join(link, pub_shortcut_name)
        ws = win32com.client.Dispatch("wscript.shell")
        shortcut = ws.CreateShortCut(link)
        shortcut.TargetPath = os.getcwd()
        shortcut.Save()
    except Exception as exception:
        error = 'Failed to deploy shortcut! {}\nArgs: {}, {}, {}'.format(exception, name, link, destination)
        print(error)
        return False
    return True

def create_DAR_folder(folder_name):
    DAR_PATH = r'W:\PD-Engines\Engine Integration Validation\Department\Data Analytics\Projects'
    folder_path = os.path.join(DAR_PATH, folder_name)
    if os.path.isdir(folder_path):
        print(f'Folder already exist!')
    else:
        try:
            os.mkdir(folder_path) #create one
            print(f'Folder created \n{folder_path}')
        except Exception as exception:
            error = f'Failed to create folder! {exception}\nArgs: {folder_name}'
            print(error)
            
def get_connectNA_info_path():
    path = 'W:\PD-Engines\Engine Integration Validation\Department\Data Analytics\Documentation\Connect NA\Remote Diagnostics Data Dictionary.xlsx'
    return path

# =============================================================================
# Pcode related
# =============================================================================
def find_corresponding_fmi_spn(pcodes, path_to_cal = r'W:\PD-Engines\Engine Integration Validation\Department\Data Analytics\Ruud\Daily Tasks\CAL\MY22\NA141_B217C\Faultlist_B217C_1078x-x-NA-141-Pre-Gamma_MY22.xlsx'):
    """
    Parameters
    ----------
    pcodes : string or list of strings
        e.g. 'P0263' or ['P0263', 'P1632']
    path_to_cal : string, full path to excel of given CAL
        The default is MY22B 'W:\\PD-Engines\\Engine Integration Validation\\Department\\Data Analytics\\Ruud\\Daily Tasks\\CAL\\MY22\\NA141_B217C\\Faultlist_B217C_1078x-x-NA-141-Pre-Gamma_MY22.xlsx'
    Returns
    -------
    Dataframe with Pcodes, SPN, FMI columns
    """
    if isinstance(pcodes, list):
        pass
    elif isinstance(pcodes, str):
        pcodes = [pcodes]
    else:
        raise ValueError("pcodes type not valid. Pcodes must be either string or list of strings starting with\'P\' followed by number e.g. P0263")

    for p in pcodes:
        if not p.startswith('P'):
            raise ValueError("All pcodes must start with 'P' followed by number. e.g. P0263")
    cal = pd.read_excel(path_to_cal, engine='openpyxl', sheet_name='data')
    SPN = [cal.iloc[i]['SPN'] for i,x in enumerate(cal['DAVIE Fault Code']) if x in pcodes]
    FMI = [cal.iloc[i]['FMI'] for i,x in enumerate(cal['DAVIE Fault Code']) if x in pcodes]
    return pd.DataFrame({'Pcode':pcodes, 'SPN':SPN, 'FMI':FMI})
    
# =============================================================================
# VISUALIZATION RELATED 
# =============================================================================
def add_plt_annotations(ax, 
                        stacked_layers=False, 
                        notation='.1f',
                        skip_number=None,
                        skip_smaller_than = None,
                        suffix='', 
                        prefix='', 
                        only_top_layer=False,
                        text_color_diff=False,
                        fontweight='normal',
                        fontsize=10,
                        xshift=0,
                        every_nth = 1
                        ):
    """Adds annotation to matplotlib.ax graphs, created specificaly for (bar charts)\n
    Parameters:
        notation - notation to be used in the graph default \'1.f\', can be e.g. \'2.e\',
        skip_number - None or int, if int don't label all values that equals given int
        suffix - suffix at the end of each note e.g. unit or % sign,
        prefix - prefix before each note ,
        only_top_layer - adds notes only to the top layer ommiting lower levels,
        text_color_diff - if True color of all but top layer are adjusted to their bar colors,
        fontweight - bold or normal
        fontsize - size of the font, default 10
        xshift - value to shift label in horizontal direction (+ to right / - to left)
        every_nth - label only every nth point in each layer (default = 1 - all points)
    """
    annot_format = prefix + '{:' + notation + '}' + suffix 
    heights = [p.get_height() for p in ax.patches]
    x_pos = [p.get_x() for p in ax.patches]
    colors = [p.get_facecolor() for p in ax.patches]
    layers_count = int(len(heights) / len(np.unique(x_pos)))
    
    #Reshape 
    heights = np.reshape(heights, (layers_count, -1))
    if stacked_layers:
        heights = heights.cumsum(axis=0)
    x_pos = np.reshape(x_pos, (layers_count, -1))
    #set top layer to black by default 
    colors = np.reshape(colors, (layers_count, -1, 4))       

    if only_top_layer and (layers_count > 1):
        heights=heights[-1]
        # make it 2d again
        heights = np.expand_dims(heights, axis=0)
        x_pos = x_pos[0 : int(len(x_pos)/layers_count)] 
        layers_count = 1
        # set only one layer and make it black
        colors = colors[-1]
        colors = np.expand_dims(colors, axis=0)
        colors.fill(0)
        colors[: ,:,3] =1

    to_skip = False
    x_pos = x_pos[:, ::every_nth]
    heights = heights[:, ::every_nth]
    colors = colors[:, ::every_nth, :]
    for layer in list(range(layers_count)):
        for i, x in enumerate(x_pos[layer]):
            if skip_number is not None:
                # set flag to true if number == number we want to skip
                to_skip = heights[layer][i] == skip_number
            
            if skip_smaller_than is not None:
                to_skip = heights[layer][i] < skip_smaller_than

            if to_skip:
                to_skip = False # set to false again
                pass    # and jump to next itter
            else:
                if text_color_diff:
                    ax.annotate(annot_format.format(heights[layer][i]),
                                (x + xshift, heights[layer][i] * 1.005),
                                color=colors[layer][i], 
                                fontweight=fontweight,
                                fontsize=fontsize)        
                else:
                    ax.annotate(annot_format.format(heights[layer][i]), 
                                (x + xshift, heights[layer][i] * 1.005), 
                                fontweight=fontweight,
                                fontsize=fontsize)

def add_watermark(ax, rotation=0, alpha = 0.3):
    ax.text(0.92, 0.1, 'WB', transform=ax.transAxes,
    fontsize=40, color='gray', alpha=alpha,
    ha='center', va='center', rotation=rotation)
            
# =============================================================================
# DATA SCIENCE RELATED        
# =============================================================================
def get_high_corr_heatmap(df, corr_limit = 0.8):
    """Calculate, plot and return correlation coefficients within dataframe \'df\' above specified (absolute) corr_limit (default 0.8)"""
    corr_coefs = df.corr()
    corr_coefs_cols_above_limit = []
    for mode in df['en_ToesmEmissionDemState'].unique():
        corr_coefs_cols_above_limit.extend(corr_coefs[(corr_coefs[mode]>corr_limit) | (corr_coefs[mode]<-corr_limit)].index.values)
    indexes_to_drop = [col for col in corr_coefs.columns if col not in corr_coefs_cols_above_limit]
    corr_coefs_above_limit = corr_coefs.drop(index=indexes_to_drop)
    sns.heatmap(corr_coefs_above_limit[corr_coefs_cols_above_limit], cmap='coolwarm')
    plt.tight_layout()
    return corr_coefs_above_limit
            
# =============================================================================
# OTHER
# =============================================================================
def get_continuous_intervals(signal: "pd.Series[bool]") -> "pd.Series[float]":
    """
    Returns duration for continuous intervals. E.g. 
    >>> get_continuous_intervals(pd.Series([True, True, False, True, True, True])) 
    0    1.0
    1    2.0
    2    0.0
    3    1.0
    4    2.0
    5    3.0
    dtype: float64
    
    Parameters
    ----------
    signal : pd.Series([bool])
        Series with bool dtype.

    Returns
    -------
    continous_int : pd.Series([bool])
        Duration of intervals
        
    """
    return (signal.cumsum() - signal.cumsum().where(signal==False).ffill().fillna(0))

def print_df(df, nrows=None, ncols=None):
    with pd.option_context('display.max_rows', nrows, 'display.max_columns', ncols): 
        display(df)

