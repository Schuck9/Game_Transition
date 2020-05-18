"""
A simple implementation of Ultimatum Game visualization
@date: 2020.5.18
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

def bar_viz(data_path):
	# matplotlib模块绘制直方图
	# 读入数据
	data = pd.read_excel(data_path)
	save_path = os.path.join(os.getcwd(),"T1.jpg")
	# 绘制直方图
	# print(list(data.p))
	data.dropna(subset=['p'], inplace=True)
	data.dropna(subset=['q'], inplace=True)
	plt.bar([1,3,5,7,9],list(data.p),label="Offer(p)")
	plt.bar([2,4,6,8,10],list(data.q),label="Demond(q)")
	# 添加x轴和y轴标签
	plt.xlabel('mode')
	plt.ylabel('Offer Or Demond')
	meta_element = np.arange(10)
	ax_label = [" ","0.5/1"," "," "," ","0.5"," "," "," "," 1"]
	plt.xticks(meta_element,ax_label,fontsize=16)

	# 添加标题
	plt.legend()
	plt.title('RG_D_EF_w0.1_u0.001  ')
	# 显示图形
	plt.savefig(save_path)
	print("Figure has been saved to: ",save_path)
	plt.show()

if __name__ == '__main__':

    # RecordName ='2020-03-03-09-14-20'   
    # time_option = "all"
    # pq_distribution_viz(RecordName,time_option)
    # avg_pq_viz()
    data_path ='./Hist.xlsx'
    bar_viz(data_path)