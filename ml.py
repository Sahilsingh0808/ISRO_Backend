import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
import scipy.interpolate
import plotly.express as px
from math import sqrt
import os
from scipy.optimize import curve_fit
from numpy import exp, log
import time
import cdflib

MIN_MA_WINDOW_SZ = 20
MIN_MULTIPLIER = 4

""""File Paths"""
def return_folder_paths(file_name):
    raw_files_path = f'./{file_name}/xsm/data/{file_name[8:12]}/{file_name[12:14]}/{file_name[14:16]}/raw'
    calibrated_files_path = f'./{file_name}/xsm/data/{file_name[8:12]}/{file_name[12:14]}/{file_name[14:16]}/calibrated'
    return raw_files_path, calibrated_files_path

"""Returning a Light Curve from a file"""
def lightcurve(file):
    tmp_time = []
    tmp_rate = []

    if (file.endswith(".lc") or file.endswith(".fits")):
        t = Table.read(file)
        for i in range(len(t)):
            tmp_time.append(int(t[i][0] - t[0][0]))
            tmp_rate.append(t[i][1])
    elif (file.endswith(".xls") or file.endswith(".xlsx")):
        t = pd.read_excel(file, header=None)
        tmp_rate = t[1]
        for i in range(len(tmp_rate)):
            tmp_time.append(int(t[0][i] - t[0][0]))
    elif (file.endswith(".csv")):
        t = pd.read_csv(file, header=None)
        tmp_rate = t[1]
        for i in range(len(tmp_rate)):
            tmp_time.append(int(t[0][i] - t[0][0]))
    elif (file.endswith(".txt")):
        t = pd.read_csv(file, header=None, delimiter=" ", skipinitialspace=True)
        tmp_rate = t[1]
        for i in range(len(tmp_rate)):
            tmp_time.append(int(t[0][i] - t[0][0]))
    elif (file.endswith(".cdf")):
        t = cdflib.CDF(file)
        for i in range(len(t[0][0])):
            tmp_time.append(int(t[0][0][i][0] - t[0][0][0][0]))
            tmp_rate.append(int(t[0][0][i][1]))
    
    prev = 0
    x_arr = []
    y_arr = []
    __x = []
    __y = []
    for i in range(len(tmp_time)-1):
        if (tmp_time[i+1]-tmp_time[i] > 2):
            __x = []
            __y = []
            for j in range(prev, i+1):
                __x.append(int(tmp_time[j]))
                __y.append(int(tmp_rate[j]))
                if (tmp_time[j+1]-tmp_time[j] == 2):
                    __x.append(int(tmp_time[j]+1))
                    __y.append(int((tmp_rate[j] + tmp_rate[j+1])/2))
            x_arr.append(__x)
            y_arr.append(__y)
            prev = i+1
    __x = []
    __y = []
    for i in range(prev, len(tmp_time)-1):
        __x.append(int(tmp_time[i]))
        __y.append(int(tmp_rate[i]))
    x_arr.append(__x)
    y_arr.append(__y)
    return x_arr, y_arr

"""Plotting the curve using Plotly"""
def plot_as_plotly_line(_x, _y, _title):
    _df = pd.DataFrame(zip(_x, _y))
    _df.columns = ['X-Axis', 'Y-Axis']
    _fig = px.line(_df, x='X-Axis', y='Y-Axis', title=_title)
    _fig.show()
    return

def plot_as_plotly_scatter(_x, _y, _title):
    _df = pd.DataFrame(zip(_x, _y))
    _df.columns = ['X-Axis', 'Y-Axis']
    _fig = px.scatter(_df, x='X-Axis', y='Y-Axis', title=_title)
    _fig.show()
    return

"""Return the slope calculated between two data points in 2-Dimensional space"""
def get_slope(x1, x2, y1, y2):
    return ((y1-y2) / (x1-x2))

"""Return the Eucleidean distance between two points"""
def pythagorean(x1, x2, y1, y2):
    _y = (y1-y2)*(y1-y2)
    _x = (x1-x2)*(x1-x2)
    return sqrt(_x + _y)

"""Helper functions to fit on the detected flares"""
k = 0.5

def exp_fit_func(x, ln_a, b):
    t = (x ** k)
    return (ln_a - b*t)

def exp_func(x, a, b):
    t = -1 * b * (x ** k)
    return (a * np.exp(t))

def inverse_exp_func(y, a, b):
    t1 = log(y) - log(a)
    t2 = -1 * t1 / b
    return int(t2 ** (1. /k))

def smoothening_ma(__x, __y, window_sz, shift):
    new_norm = []
    new_norm_data_points = []
    new_norm.append(np.mean(__y[0:window_sz]))
    new_norm_data_points.append(__x[0])
    for i in range(window_sz, len(__y), shift):
        tmp = np.mean(__y[i:i+shift])
        new_norm.append(tmp)
        new_norm_data_points.append(__x[i])
    new_norm = np.array(new_norm)
    new_norm_data_points = np.array(new_norm_data_points)
    xnew = np.linspace(__x[0], __x[0]+len(__x), __x[0]+len(__x))
    f = scipy.interpolate.interp1d(new_norm_data_points, new_norm, fill_value='extrapolate', kind='linear')
    ynew = f(xnew)
    return xnew, ynew

def smoothening_fft(lc, thresh=200, should_plot=False):
    lc_fft = np.fft.fft(lc)
    lc_fft[thresh:len(lc)-thresh]=0
    lc_smooth = np.abs(np.fft.ifft(lc_fft)) + 1e-5
    if should_plot:
        px.line(pd.DataFrame(lc_smooth))
    xnew = np.linspace(0, len(lc), len(lc))
    return xnew, abs(lc_smooth)

"""Getting extremas functions"""
def get_lvl_0_extremas(xnew, ynew, should_plot=False):
    _s0 = []
    _p0 = []
    if ynew[0] <= ynew[1]:
        _s0.append(0)
    for i in range(1, len(ynew)-1):
        if (ynew[i]>ynew[i-1]) and (ynew[i]>ynew[i+1]):
            _p0.append(i)
        elif (ynew[i]<ynew[i-1]) and (ynew[i]<ynew[i+1]):
            _s0.append(i)
    if ynew[-2] >= ynew[-1]:
        _s0.append(len(xnew)-1)
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Level 0 Maximas")
        plt.plot([xnew[i] for i in _p0], [ynew[i] for i in _p0], 'o', xnew, ynew)
        plt.show()
        plt.figure(figsize=(30, 10))
        plt.title("Level 0 Minimas")
        plt.plot([xnew[i] for i in _s0], [ynew[i] for i in _s0], 'o', xnew, ynew)
        plt.show()
    return _s0, _p0

def get_lvl_1_extremas(xnew, ynew, _s0, _p0, should_plot=False):
    _s1 = []
    _p1 = []
    for i in range(len(_p0)):
        for j in range(len(_s0)-1):
            if (xnew[_s0[j+1]] > xnew[_p0[i]]):
                _s1.append(_s0[j])
                _p1.append(_p0[i])
                break
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Level 1 Maximas")
        plt.plot([xnew[i] for i in _p1], [ynew[i] for i in _p1], 'o', xnew, ynew)
        plt.show()
        plt.figure(figsize=(30, 10))
        plt.title("Level 1 Minimas")
        plt.plot([xnew[i] for i in _s1], [ynew[i] for i in _s1], 'o', xnew, ynew)
        plt.show()
        print(len(_s1))
    return _s1, _p1

def get_lvl_2_extremas(xnew, ynew, _s1, _p1, should_plot=False):
    _s2 = []
    _p2 = []
    _slopes = np.array([get_slope(xnew[_s1[i]], xnew[_p1[i]], ynew[_s1[i]], ynew[_p1[i]]) for i in range(len(_s1))])
    mean_sl = np.mean(_slopes)
    for i in range(len(_s1)):
        if (_slopes[i] > mean_sl) or ():
            _s2.append(_s1[i])
            _p2.append(_p1[i])
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Level 2 Maximas")
        plt.plot([xnew[i] for i in _p2], [ynew[i] for i in _p2], 'o', xnew, ynew)
        plt.show()
        plt.figure(figsize=(30, 10))
        plt.title("Level 2 Minimas")
        plt.plot([xnew[i] for i in _s2], [ynew[i] for i in _s2], 'o', xnew, ynew)
        plt.show()
    return _s2, _p2

def get_lvl_3_extremas(xnew, ynew, _s2, _p2, f, should_plot=False):
    _s3 = []
    _p3 = []
    _std = np.std(np.array([ynew[_p2[i]] - ynew[_s2[i]] for i in range(len(_s2))]))
    for i in range(len(_s2)):
        if (ynew[_p2[i]]-ynew[_s2[i]] > _std * f):                                      #! SET THIS PARAMETER!!!!!!!
            _s3.append(_s2[i])
            _p3.append(_p2[i])
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Level 3 Maximas")
        plt.plot([xnew[i] for i in _p3], [ynew[i] for i in _p3], 'o', xnew, ynew)
        plt.show()
        plt.figure(figsize=(30, 10))
        plt.title("Level 3 Minimas")
        plt.plot([xnew[i] for i in _s3], [ynew[i] for i in _s3], 'o', xnew, ynew)
        plt.show()
    return _s3, _p3

def get_lvl_4_extremas(xnew, ynew, _s3, _p3, should_plot=False):
    _s4 = []
    _p4 = []
    s = set()
    for i in range(len(_s3)):
        if _s3[i] in s:
            continue
        s.add(_s3[i])
        _s4.append(_s3[i])
        _p4.append(_p3[i])
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Level 4 Maximas")
        plt.plot([xnew[i] for i in _p4], [ynew[i] for i in _p4], 'o', xnew, ynew)
        plt.show()
        plt.figure(figsize=(30, 10))
        plt.title("Level 4 Minimas")
        plt.plot([xnew[i] for i in _s4], [ynew[i] for i in _s4], 'o', xnew, ynew)
        plt.show()
    return _s4, _p4

def get_lvl_5_extremas(xnew, ynew, _s4, _p4, should_plot=False):
    _s5 = []
    _p5 = []
    i = 0
    while i < len(_p4)-1:
        flag = True
        if (xnew[_p4[i+1]]-xnew[_p4[i]] < 500) and (np.abs(ynew[_p4[i+1]] - ynew[_p4[i]]) < 30):
            if (ynew[_p4[i+1]] > ynew[_p4[i]]):
                _s5.append(_s4[i])
                _p5.append(_p4[i+1])
                flag = False
            elif ((ynew[_p4[i+1]] < ynew[_p4[i]])):
                _s5.append(_s4[i])
                _p5.append(_p4[i])
                flag = False
        if flag:
            _s5.append(_s4[i])
            _p5.append(_p4[i])
        else:
            i += 1
        i += 1
    _s5.append(_s4[-1])
    _p5.append(_p4[-1])
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Final Start and Peak")
        plt.plot([xnew[i] for i in _p5], [ynew[i] for i in _p5], 'o', [xnew[i] for i in _s5], [ynew[i] for i in _s5], 'x', xnew, ynew)
        plt.show()
    return _s5, _p5

def get_lvl_0_ends(xnew, ynew, _s4, _p4, _s0, should_plot=False):
    _e0 = []
    for i in range(len(_p4)):
        for j in range(_p4[i], len(xnew)):
            if i == len(_p4) - 1:
                if (xnew[_s0[-1]] < xnew[_p4[i]]):
                    _e0.append(len(xnew)-1)
                    break
                elif j == len(xnew) - 1:
                    _e0.append(len(xnew)-1)
            if (ynew[j] < (ynew[_p4[i]] + ynew[_s4[i]])/2):
                _e0.append(j)
                break
            if i+1<len(_s4):
                if (xnew[j] > xnew[_s4[i+1]]):
                    _e0.append(j-1)
                    break
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Level 0 Ends")
        plt.plot([xnew[i] for i in _p4], [ynew[i] for i in _p4], 'o', [xnew[i] for i in _e0], [ynew[i] for i in _e0], 'x', xnew, ynew)
        plt.show()
    return _e0

def get_lvl_1_ends(xnew, ynew, _s0, _p4, _e0, should_plot=False):
    _e1 = []
    for i in range(len(_e0)):
        if (ynew[_e0[i]] < ynew[_p4[i]]):
            _e1.append(_e0[i])
        else:
            for j in range(len(_s0)):
                if (xnew[_s0[j]] > xnew[_p4[i]]):
                    if j+1 >= len(_s0):
                        _e1.append(_s0[j])
                        break
                    if (ynew[_s0[j+1]] > ynew[_s0[j]]):
                        _e1.append(_s0[j])
                        break
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Peaks and Ends")
        plt.plot([xnew[i] for i in _p4], [ynew[i] for i in _p4], 'o', [xnew[i] for i in _e1], [ynew[i] for i in _e1], 'x', xnew, ynew)
        plt.show()
    return _e1

def get_interm_zip_features(ynew, _s4, _p4, _e1):
    start_times = []
    peak_times = []
    end_times = []
    peak_intensities = []
    for i in range(len(_s4)):
        if (_p4[i]-_s4[i] > 0) and (_e1[i]-_p4[i] > 0):
            start_times.append(_s4[i])
            peak_times.append(_p4[i])
            end_times.append(_e1[i])
            peak_intensities.append(ynew[_p4[i]])
    return start_times, peak_times, end_times, peak_intensities

def get_interm_zip(h1, h2, h3, h4):
    _zip = pd.DataFrame(zip(h1, h2, h3, h4))
    _zip.columns = ['start_time', 'peak_time', 'end_time', 'peak_intensity']
    return _zip

def get_final_zip_features(xnew, ynew, _zip):
    st = _zip['start_time']
    pt = _zip['peak_time']
    et =  _zip['end_time']
    pi = _zip['peak_intensity']
    y_min = np.min(ynew)
    final_st = []
    final_pt = []
    final_et = []
    est_et = []
    final_si = []
    final_pi = []
    final_err = []
    final_bc = []
    _class = []
    for i in range(len(st)):
        x_range = [int(xnew[j]-xnew[pt[i]]) for j in range(pt[i], et[i])]
        ln_y_range = [np.log(ynew[j]) for j in range(pt[i], et[i])]
        try:
            popc, pcov = curve_fit(exp_fit_func, x_range, ln_y_range)
            ln_a, b = popc
            a = np.exp(ln_a)
            if (b < 0):
                continue
            _calc_et = inverse_exp_func(ynew[st[i]], a, b)
            final_st.append(st[i])
            final_pt.append(pt[i])
            final_et.append(et[i])
            final_pi.append(pi[i])
            final_si.append(ynew[st[i]])
            est_et.append(_calc_et + pt[i])
            final_bc.append((ynew[st[i]]+ynew[et[i]])/2)
            y_dash = []
            y_diff = []
            y_proj = []
            x_proj = []
            for _i, j in enumerate(x_range):
                __y = exp_func(xnew[j], a, b)
                y_dash.append(__y)
                y_diff.append(abs(exp(ln_y_range[_i]) - __y))
            for j in range(et[i]-pt[i], _calc_et):
                if ((j + pt[i]) < len(xnew)):
                    x_proj.append(xnew[j + pt[i]])
                    y_proj.append(exp_func(xnew[j], a, b))
            final_err.append((np.sum(y_dash)) / ((pi[i] - y_min) * (len(x_range))))
            val = np.log10(pi[i] / 25)
            _str = ""
            _val = str(int(val * 100) / 10)[-3:]
            if (int(val) < 1):
                _str = "A"+_val
            elif (int(val) == 1):
                _str = "B"+_val
            elif (int(val) == 2):
                _str = "C"+_val
            elif (int(val) == 3):
                _str = "M"+_val
            elif (int(val) > 3):
                _str = "X"+_val
            _class.append(_str)
        except:
            print("Error in curve fitting")
    return final_st, final_pt, final_et, est_et, final_si, final_pi, final_bc, final_err, _class

def get_final_zip(g1, g2, g3, g4, g5, g6, g7, g8, g9):
    final_zip = pd.DataFrame(zip(g1, g2, g3, g4, g5, g6, g7, g8, g9))
    final_zip.columns = ['start_time', 'peak_time', 'end_time', 'est_end_time', 'start_intensity', 'peak_intensity', 'background_counts', 'error', 'class']
    return final_zip

def get_model_features(final_zip, file):
    f0 = []
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    for i in range(len(final_zip)):
        f0.append(str((final_zip['start_time'][i] + final_zip['end_time'][i])//2) + '_' + file)
        t1 = (final_zip['est_end_time'][i] - final_zip['start_time'][i]) / (final_zip['end_time'][i] - final_zip['peak_time'][i])
        f1.append(t1*t1)
        t2 = (final_zip['peak_intensity'][i] - final_zip['background_counts'][i]) / (final_zip['background_counts'][i] - final_zip['start_intensity'][i])
        f2.append(t2)
        t3 = (final_zip['peak_intensity'][i] - final_zip['start_intensity'][i]) / (final_zip['peak_time'][i] - final_zip['start_time'][i])
        f3.append(t3)
        t4 = final_zip['error'][i]
        f4.append(t4)
        t5 = (final_zip["peak_intensity"][i] - final_zip["start_intensity"][i]) / (final_zip["est_end_time"][i] - final_zip["peak_time"][i])
        f5.append(t5)
    tmp = pd.DataFrame(zip(f0, f1, f2, f3, f5, f4))
    tmp.columns = ['_id', 'time_ratio', 'intensity_ratio', 'bandwidth_1', 'bandwidth_2', 'error']
    return tmp

"""Main function"""
def main(path_to_lc):
    x_arr, y_arr = lightcurve(path_to_lc)
    x_new = []
    y_new = []
    for i in range(len(x_arr)):
        window_sz = 20 + 100 * int(1 / (1 + np.exp(-1 * (len(x_arr[i]) - 240))))
        if (len(x_arr[i]) >= 120):
            _x, _y = smoothening_ma(x_arr[i], y_arr[i], 2*window_sz, window_sz//2)
            for j in range(len(_x)):
                x_new.append(_x[j])
                y_new.append(_y[j])

    xnew = np.linspace(int(x_new[0]), int(x_new[-1]-x_new[0]), int(x_new[-1]-x_new[0]))
    f__ = scipy.interpolate.interp1d(x_new, y_new, fill_value='extrapolate', kind='linear')
    ynew = f__(xnew)
    _s0, _p0 = get_lvl_0_extremas(xnew, ynew, should_plot=False)
    _s1, _p1 = get_lvl_1_extremas(xnew, ynew, _s0, _p0, should_plot=False)
    _s2, _p2 = get_lvl_2_extremas(xnew, ynew, _s1, _p1, should_plot=False)
    _s3, _p3 = get_lvl_3_extremas(xnew, ynew, _s2, _p2, 0.3, should_plot=False)
    _s4, _p4 = get_lvl_4_extremas(xnew, ynew, _s3, _p3, should_plot=False)
    _s5, _p5 = get_lvl_5_extremas(xnew, ynew, _s4, _p4, should_plot=False)

    _e0 = get_lvl_0_ends(xnew, ynew, _s5, _p5, _s0, should_plot=False)
    _e1 = get_lvl_1_ends(xnew, ynew, _s0, _p5, _e0, should_plot=False)
    if len(_e1) != 0:
        h1, h2, h3, h4 = get_interm_zip_features(ynew, _s5, _p5, _e1)
        if len(h1) != 0:
            _zip = get_interm_zip(h1, h2, h3, h4)
            g1, g2, g3, g4, g5, g6, g7, g8, g9 = get_final_zip_features(xnew, ynew, _zip)
            if len(g1) != 0:
                final_zip = get_final_zip(g1, g2, g3, g4, g5, g6, g7, g8, g9)
                model_zip = get_model_features(final_zip, path_to_lc)
                print(final_zip)

if __name__ == '__main__':
    main("./ch2_xsm_20211012_v1_level2.lc")