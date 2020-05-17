"""
A simple implementation of Ultimatum Game visualization
@date: 2020.2.10
@author: Tingyu Mo
"""

import numpy as np
import pandas as pd
import os
import time
import fractions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

def viz_frequency_map(RecordName,Epoch):
    result_dir = './result'

    if RecordName != None:
        Record_dir = os.path.join(result_dir,RecordName)
        Epoch_list = os.listdir(Record_dir)
        Epoch_template_str = Epoch_list[0].split('_')
        # Epoch_str = Epoch_template_str[0]+'_ep{}_'.format(Epoch)+Epoch_template_str[2]
        Epoch_str = ''
        for i in Epoch_template_str[:-1]:
            Epoch_str += i+"_"
        gragh_dir = os.path.join(result_dir,RecordName,Epoch_str+"graph")
        if not os.path.exists(gragh_dir):
            os.mkdir(gragh_dir)
        Epoch_str += str(Epoch)
        f_str = Epoch_str+'_frequency_matrix.csv'

        frequency_path = os.path.join(Record_dir,Epoch_str,f_str)
        # frequency_path = os.path.join(result_dir,'2020-02-09-23-11-44\w100_ep9400000_u0.0562\frequency_w100_ep9400000_u0.0562.csv')
        w = Epoch_template_str[3]
        u = Epoch_template_str[4]
        outputname =  os.path.join(gragh_dir,Epoch_str+str(Epoch)+'_Gragh.jpg')
        frequency = pd.read_csv(frequency_path,index_col=0,header=0)


    f = np.around(frequency.values,4)
    meta_element = np.arange(6)/6
    p = meta_element.copy()
    q = meta_element.copy()
    
    np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})# decimals to fractions

    plt.figure(figsize=(12.8,9.6),dpi=100,frameon=True)
    # set the grids density
    levels = MaxNLocator(nbins=10).tick_values(np.min(f),np.max(f))
    cm = plt.cm.get_cmap('autumn_r')
    nm = BoundaryNorm(levels,ncolors=cm.N,clip=True)

    plt.pcolormesh(p,q,f,cmap=cm,norm=nm)
    # contourf method is much smoother than pcolormesh!
    # plt.contourf(p,q,f,levels=levels,cmap=cm)

    bar = 'bar'
    if bar == 'bar':
        cbar = plt.colorbar()
        cbar.set_label('Frequency',rotation=-90,va='bottom',fontsize=40)
        tic = np.around(np.arange(np.min(f),np.max(f),(np.max(f)-np.min(f))/10),4)
        cbar.set_ticks(tic)
        
        # set the font size of colorbar
        cbar.ax.tick_params(labelsize=32) 

    # ax_label = ['0',' ','1/6',' ','1/3',' ','1/2',' ','2/3',' ','5/6',' ','1']
    ax_label = ['0','1/4','1/2','3/4','1']
    plt.title("w={} u={} Epoch={}".format(w,u,Epoch),fontsize = 40)
    plt.xticks(meta_element,ax_label,fontsize=16)
    plt.yticks(meta_element,ax_label,fontsize=16)
    plt.xlabel('Offer(p)',fontsize=40)
    plt.ylabel('Demand(q)',fontsize=40)
    plt.tight_layout()
    plt.savefig(outputname,dpi=300)
    plt.show()

if __name__ == '__main__':
    # RecordName = '2020-02-15-23-02-44'    #0.3162
    # RecordName ='2020-02-15-13-35-16'     #0.036
    # RecordName ='2020-02-15-13-36-11'   #100
    RecordName ='2020-05-15-12-10-37' # 0.1
    
    Epoch_list = [100000,500000,1000000,2000000,5000000,10000000]
    for Epoch in Epoch_list:
        viz_frequency_map(RecordName,Epoch)