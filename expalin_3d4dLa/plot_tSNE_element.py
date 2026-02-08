import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(  {'legend.frameon':False}   )
mpl.rcParams.update({'savefig.dpi': 1200})
mpl.rcParams.update({'font.family': 'Arial'})
###########################################################
# mpl.rcParams.update({'figure.figsize': (1*3.25, 1*3.25*0.75)})
mpl.rcParams.update({'figure.figsize': (4*2/3*3.25 , 3*2/3*3.25*0.75)})#width height#mpl.rcParams.update({'figure.figsize': (3.25, 3.25*0.75)})
mpl.rcParams.update({'font.size': 6})
mpl.rcParams.update({'lines.linewidth': 0.5})
print("Figure size:",      mpl.rcParams['figure.figsize'])
print("Figure dpi_saved:", mpl.rcParams['savefig.dpi'])
print("Figure font:",      mpl.rcParams['font.family'])
print("Figure font_size:", mpl.rcParams['font.size'])
print("Figure line_width:",mpl.rcParams['lines.linewidth'])
print("Finish Figure setting Print")
###########################################################
mpl.rcParams.update({'lines.markersize': 3.0})#plt.rcParams['lines.markersize']
mpl.rcParams.update({'lines.markerfacecolor': 'none'})
############################################################
import matplotlib.gridspec as gridspec


import numpy as np

setting_prefix="tsne2"#"tsne2MoRuRhPd"
setting_prefix_save=setting_prefix+"MoRuRhPd"
data=np.loadtxt(setting_prefix+".txt")
zmat=np.load("La4d3d_unconv_valid_z.npy",allow_pickle=True)
z_list=[]
for i in range(len(zmat)):
    iz=zmat[i][0]
    z_list.append(iz)
z_array=np.array(z_list).astype(int)

# 获取唯一的z值
unique_z = np.unique(z_array)
print(f"Unique z values: {unique_z}")
print(f"Number of unique z values: {len(unique_z)}")

# 创建z值到元素名称的映射
# zgroup_list=[[21,22,23,24,25,26,27,28,29,30],[57,58,59,60,61,62,63,64,65,66,67,68,69,70,71]]
# color_list=['black','red']
# # z_to_element = {26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn'}

# # 为五个元素分别设置不同的颜色
# z_to_color = {26: 'blue', 27: 'green', 28: 'red', 29: 'purple', 30: 'orange'}

# 为了确保绘制1行5列，我们最多显示5个z值
# max_plots = len(zgroup_list)
# selected_z = unique_z[:max_plots]

# 创建1x5的子图布局


# 计算所有数据点的坐标范围，以便统一设置所有子图的范围
zgroup_list=[24,26,27,28]+[42,44,45,46]+[60,62,63,64]
#[57,58,70,71]
#[21,22,23,24,25,26,27,28,29,30]+[57,58,60,62,63,64,65,66,68,70]
#Pr59  Pm61
# total_min_x, total_max_x = data[:, 0].min(), data[:, 0].max()
# total_min_y, total_max_y = data[:, 1].min(), data[:, 1].max()
total_min_x=-150;total_min_y=-150
total_max_x=150;total_max_y=150
z_to_element = {21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn'
,57:"La",58:"Ce",59:"Pr",60:"Nd",62:"Sm",63:"Eu",64:"Gd",65:"Tb",66:"Dy",68:"Er",70:"Yb",71:"Lu",42:"Mo",44:"Ru",45:"Rh",46:"Pd"}

ABC_list=['A','B','C','D','E','F','G','H','I','J','K','L']


# fig, axes = plt.subplots(1, 2, figsize=(25, 5))
# axes = axes.flatten()  # 将2D数组展平为1D以便迭代
# # 为每个z值创建一个子图
# for i in range(len(zgroup_list)) :
#     # 筛选当前z值的数据点
#     igroup=zgroup_list[i]
#     mask = np.array([x in igroup for x in z_array])
#     x_data = data[mask, 0]
#     y_data = data[mask, 1]
    
#     # 绘制散点图，使用元素对应的特定颜色
#     ax = axes[i]
#     # 使用特定颜色而不是colormap
#     scatter = ax.scatter(x_data, y_data, c=color_list[i], s=50, alpha=0.8)
    
#     # 设置统一的坐标轴范围
#     ax.set_xlim(total_min_x, total_max_x)
#     ax.set_ylim(total_min_y, total_max_y)
    
#     # 设置子图标题和坐标轴标签
#     # element_name = z_to_element.get(z_val, f'Z={z_val}')
#     # ax.set_title(element_name, fontsize=14, fontweight='bold')
#     ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
#     ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    
#     # 显示网格线
#     ax.grid(True, linestyle='--', alpha=0.5)




fig, axes = plt.subplots(3, 4)#width height
axes = axes.flatten()  # 将2D数组展平为1D以便迭代
# ax3=axes[-1]
# ax3.set_xlim(total_min_x, total_max_x)
# ax3.set_ylim(total_min_y, total_max_y)
# 为每个z值创建一个子图
for i in range(len(zgroup_list)) :
    ax = axes[i]
    ax.set_xlim(total_min_x, total_max_x)
    ax.set_ylim(total_min_y, total_max_y)
    # 筛选当前z值的数据点
    igroup=zgroup_list[i]
    mask = np.array([x==igroup for x in z_array])
    x_data = data[mask, 0]
    y_data = data[mask, 1]

    if igroup in [21,22,23,24,25,26,27,28,29,30]:
        setting_color='black'
    elif igroup in [57,58,60,62,63,64,65,66,68,70,71]:
        setting_color='red'
    else:
        setting_color='blue'

    

    scatter = ax.scatter(x_data, y_data,label=z_to_element[igroup],c=setting_color)#, c=color_list[i]  s=50, alpha=0.8
    ax.legend(loc='upper right',fontsize=15)
    ax.set_title(ABC_list[i],loc="left")
 

# 显示图形
plt.tight_layout()
plt.savefig(setting_prefix_save+".png")
plt.savefig(setting_prefix_save+".tiff")
plt.savefig(setting_prefix_save+".eps")
plt.show()

print("DEBUG")