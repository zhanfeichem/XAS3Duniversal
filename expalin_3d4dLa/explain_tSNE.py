from sklearn.manifold import TSNE
import numpy as np


setting_finput="exp1.txt"
setting_foutput="tsne1.txt"
setting_fatomnumber="La4d3d_unconv_valid_z.npy"


# 读取exp0.txt文件
data=np.loadtxt(setting_finput)
zmat=np.load(setting_fatomnumber,allow_pickle=True)
z_list=[]
for i in range(len(zmat)):
    iz=zmat[i][0]
    z_list.append(iz)
z_array=np.array(z_list).astype(int)
print(f"原始数据形状: {data.shape}")

# 执行tSNE降维
tsne = TSNE(n_components=2, random_state=0,n_jobs=4)
data_2d = tsne.fit_transform(data)
print(f"降维后数据形状: {data_2d.shape}")

# 保存降维后的数据到txt文件
np.savetxt(setting_foutput, data_2d, fmt='%.6f', delimiter=' ')
print("tSNE降维完成，结果已保存到tsne_result.txt")

print("DEBUG")