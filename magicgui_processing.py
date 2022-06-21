# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:51:37 2022

@author: GiorgiaT
"""

from magicgui import magicgui 
import datetime
import pathlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def take_data(excel_file, selected_sheets):

    sheets = pd.read_excel(excel_file, sheet_name = None)
 
    intensities = []
    yx_coordinates = []
    t_indices = []

    for name, sheet in sheets.items():
        if name in selected_sheets:
           t_index = sheet['t_index']
           intensity = sheet['intensity']
           yx = sheet[['y', 'x']]
           t_indices.append(t_index)
           intensities.append(intensity)
           yx_coordinates.append(yx)
    
    t_indices_array = np.squeeze(np.array(t_indices))
    intensities_array = np.squeeze(np.array(intensities))
    yx_array = np.squeeze(np.array(yx_coordinates))
    
    return t_indices_array, intensities_array, yx_array

def get_data(rois_list, filenames): 
    
    t_idx_list = []
    int_list = []
    yx_list = []
    for idx,filename in enumerate(filenames):  
            roi_t_idx_list = []
            roi_int_list = []
            roi_yx_list = []
            for roi_idx,roi in enumerate(rois_list):
                t_indices, intensities, yx = take_data(filename,selected_sheets = [f'Roi_{roi}'])
                roi_t_idx_list.append(t_indices)
                roi_int_list.append(intensities) 
                roi_yx_list.append(yx)
            t_idx_list.append(roi_t_idx_list)
            int_list.append(roi_int_list)
            yx_list.append(roi_yx_list) 
    return t_idx_list, int_list, yx_list

def process(delta, t_list, i_list):
    
    reference_t_idx =[]
    reference_intensities = []
    normalized_intensities_list = []
    polished_intensities_list = []
    pol_t_idx_list = []
    
    dataset_num = len(i_list)
    
    for dataset_index in range(dataset_num):
        
        _norm_intensities_list = []
        _pol_intensities_list = []
        _pol_t_idx_list = []
        
        
        for roi_index in range(len(i_list[dataset_index])):
            
            int_array = i_list[dataset_index][roi_index]
            t_idx_array = t_list[dataset_index][roi_index]
                    
            
            plt.plot(t_idx_array, int_array)
            plt.scatter(t_idx_array, int_array, c = 'red', s = 5)
            pts = plt.ginput(1, timeout=-1, show_clicks=True)
            plt.show()
            ref_t= int(pts[0][0])
            print('ref t', ref_t)
            #intensity = int_array[ref_t]
            intensity = pts[0][1]
            reference_t_idx.append(ref_t)
            reference_intensities.append(intensity)
            plt.close()
            normalized_intensities = (int_array-intensity)/intensity
            _norm_intensities_list.append(normalized_intensities)
            start_time = max(ref_t-delta,0)
            polished_intensities_array = np.array(normalized_intensities[start_time::])
            polished_t_idx_array = np.array(t_idx_array[start_time::])
            # n_zeros = len(intensities_array[..., col_idx]) - len(polished_intensities_array)
            # polished_intensities_array = np.append(polished_intensities_array, np.zeros(n_zeros))
            _pol_intensities_list.append(polished_intensities_array)
            _pol_t_idx_list.append(polished_t_idx_array)
            # print(polished_intensities_array.shape)
        normalized_intensities_list.append(_norm_intensities_list)
        polished_intensities_list.append(_pol_intensities_list)
        pol_t_idx_list.append(_pol_t_idx_list)
    # print('ref t', reference_t_idx)
    
    return reference_t_idx, reference_intensities, normalized_intensities_list, pol_t_idx_list, polished_intensities_list 

def plot_data(data_to_save, t_index_list, intensities_list, names, rois):
    
    # int_rois_list = [int(roi) for roi in rois] 
    # transposed_t_idx_list = [list(i) for i in zip(*t_index_list)]
    # transposed_int_list = [list(i) for i in zip(*intensities_list)]
    dataset_num = len(intensities_list)

    for roi_idx,roi in enumerate(rois):
        plt.figure()
        legend = []
        for dataset_index in range(dataset_num):
            name = names[dataset_index].name
            t = t_index_list[dataset_index][roi_idx]
            intens = intensities_list[dataset_index][roi_idx]
            plt.plot(t, intens)
            if data_to_save == 'Raw':
                plt.title('Raw data Roi '+ roi)
            else:
                plt.title('Corrected data Roi '+ roi)
            legend.append(name)
        plt.xlabel('t_index')
        plt.ylabel('intensity')
        plt.legend(legend,loc='best')
        plt.show()
        
def save_in_excel(save_filename, data, names, rois, velocities):
  
    writer = pd.ExcelWriter(save_filename + '.xlsx')  
    dataset_num = len(data)
    
    for dataset_index in range(dataset_num): 
        name = names[dataset_index].name     
        headers = rois
        d = data[dataset_index]
        table = pd.DataFrame(d, index = headers).transpose()
        table.index.name = 't_index'
        table.to_excel(writer, name)        
            
    writer.save()
        
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def parabola(x, a, b, c):
    y = a*(x)**2 +b*x +c
    return y

def fit(function, hm_t, hm_intens):
    
    guess = [-1, 20, 100]
    parameters, covariance = curve_fit(parabola, hm_t, hm_intens, p0 = guess)

    fit_a = parameters[0]
    fit_b = parameters[1]
    fit_c = parameters[2]
    # print(fit_A)
    # print(fit_B)

    fit_y = parabola(hm_t, fit_a, fit_b, fit_c)
    plt.plot(hm_t, hm_intens, 'o', label='data')
    plt.plot(hm_t, fit_y, '-', label='fit')
    plt.legend()

def data_in_hm(t_idx, intensities, rois):
    
    dataset_num = len(intensities)
    rois_num = len(rois)
    
    hm_intensities = []
    hm_t = []
    
    for dataset_idx in range(dataset_num):
        
        _hm_intensities = []
        _hm_t = []
        
        for roi_idx in range (rois_num):    
        
            plt.plot(t_idx[dataset_idx][roi_idx], intensities[dataset_idx][roi_idx], 'o')    
        
            t_interp = np.linspace(0,int(max(t_idx[dataset_idx][roi_idx])),1000)
            
            f = interp1d(t_idx[dataset_idx][roi_idx], intensities[dataset_idx][roi_idx], kind='cubic')
            
            intens_interp = f(t_interp)
            
            plt.plot(t_interp, intens_interp)
            
            t_idx_max = np.argmax(intens_interp, axis=0)
            # print('t max', t_idx_max)
            x_max = intens_interp[t_idx_max]
            # print('x max', x_max)
            
            t_idx_right = find_nearest(intens_interp[t_idx_max:], x_max/2) + t_idx_max
            t_idx_left = find_nearest(intens_interp[0:t_idx_max], x_max/2)
            # print('t_idx_right', t_idx_right)
            # print('t_idx_left', t_idx_left)
            new_t = t_interp[t_idx_left:t_idx_right]
            new_intens = intens_interp[t_idx_left:t_idx_right]
            _hm_intensities.append(new_intens)
            _hm_t.append(new_t)
            fit(parabola, new_t,new_intens)
            # print(new_intens.shape)
            # print(new_t.shape)
            # plt.plot(new_t,new_intens)
        hm_intensities.append(_hm_intensities)
        hm_t.append(_hm_t)
    
@magicgui(call_button="Process",
          data_to_save = {"choices": ['Raw','Normalized','Polished']})
def select_data(
    deltaT = 10,
    filename_0 = pathlib.Path(),
    filename_1 = pathlib.Path(),
    filename_2 = pathlib.Path(),
    filename_3 = pathlib.Path(),
    filename_4 = pathlib.Path(),
    filename_5 = pathlib.Path(),
    filename_6 = pathlib.Path(),
    filename_7 = pathlib.Path(),
    filename_8 = pathlib.Path(),
    filename_9 = pathlib.Path(),
    rois = '0',
    save_data: bool = True,
    saving_file_name = '\\temp',
    data_to_save: str = 'Polished'
    ):
    '''
    

    Parameters
    ----------
    deltaT : int
        time to.....
    filename_0 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_1 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_2 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_3 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_4 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_5 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_6 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_7 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_8 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    filename_9 : TYPE, optional
        DESCRIPTION. The default is pathlib.Path().
    rois : TYPE, optional
        ..... for example '0,1,5'.
    saving_file_name : TYPE, optional
        DESCRIPTION. The default is 'temp'.

    Returns
    -------
    None.

    '''
    
    
    rois_list = rois.split(',')
    #data = [[None]*len(rois_list)]
    # data = []
    print(rois_list)
    # rois_num = len(rois_list)
    
    filenames = [filename_0,filename_1,
                 filename_2,filename_3,
                 filename_4,filename_5,
                 filename_6,filename_7,
                 filename_8,filename_9]
    files = []
    for idx,filename in enumerate(filenames):  
        if filename !=pathlib.Path(): 
            files.append(filename)
            folder = filename.parent  
    
    t_idx, intensities, yx = get_data(rois_list, files)
    ref_t, ref_intensity, normalized, polished_t, polished_intensities = process(deltaT,
                                                                                 t_idx,
                                                                                 intensities)   
    # data_in_hm(t_idx, normalized, rois)
    # plot_data(polished_t, polished_intensities,
    #           names = files,
    #           rois = rois_list)
    dataset_num = len(polished_intensities)  
    velocities = 0
    savefilename = str(folder) +'\\'+ saving_file_name  
    
    if data_to_save == 'Polished':
        
        plot_data(data_to_save,
                  polished_t, polished_intensities,
                  names = files,
                  rois = rois_list)
        if save_data == True:
            save_in_excel(savefilename,
                          names = files,
                          data = polished_intensities,
                          rois = rois_list,
                          velocities = velocities)
    elif data_to_save == 'Normalized':
        
        plot_data(data_to_save,
                  t_idx, normalized,
                  names = files,
                  rois = rois_list)
        if save_data == True:
            save_in_excel(savefilename,
                          names = files,
                          data = normalized,
                          rois = rois_list,
                          velocities = velocities)
    elif data_to_save == 'Raw':
        
        plot_data(data_to_save,
                  t_idx, intensities,
                  names = files,
                  rois = rois_list)
        if save_data == True:
            save_in_excel(savefilename,
                          names = files,
                          data = intensities,
                          rois = rois_list,
                          velocities = velocities)


select_data.show()
