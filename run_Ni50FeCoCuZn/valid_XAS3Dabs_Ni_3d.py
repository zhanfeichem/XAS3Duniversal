

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.nn.functional as F
###DIG
# from dig.threedgraph.dataset import QM93D
# from dig.threedgraph.method import SphereNet
# from dig.threedgraph.method import DimeNetPP,ComENet
# from dig.threedgraph.evaluation import ThreeDEvaluator
# from dig.threedgraph.method import run
###PYG
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
######
from scipy.interpolate import interp1d
import numpy as np
import json
import argparse
import os.path as osp
import time
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as 
##########output
#from pymatgen.io.cif import CifWriter
##########ComENet
from XAS3D2026 import XAS3Dabs
##########ComENet  END
setting_fmodel="Ni50FeCoCuZn_best256_3_0.0005_5_2_82_5.pt"
setting_prefix_train="Ni50FeCoCuZn8.5k"
setting_prefix_valid="FeCoCuZn1k"
setting_prefix_valid_look="Ni_CCDC" #"Ni_CCDC38"

# setting_prefix_test="Ni_CCDC38"#not using
setting_n_ep=500





class train(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        #define my own data from Feterpy
        data_list = []
        prefix=setting_prefix_train
        print("Prefix is: ",prefix)
        pos_list=np.load(prefix+"_train_pos.npy", allow_pickle=True)
        z_list = np.load(prefix+"_train_z.npy", allow_pickle=True)
        y_list = np.load(prefix+"_train_y.npy", allow_pickle=True)
        # r_list = np.load(prefix+"_train_r.npy",allow_pickle=True)

        for i in range(len(pos_list)):
            iz=z_list[i]
            iz=iz.astype(int)
            pos = torch.tensor(pos_list[i])
            z = torch.tensor(iz)
            y = torch.tensor(y_list[i])
            y=y.reshape([1,-1])
            # r = torch.tensor(r_list[i])
            tmp = Data(z=z, pos=pos, y=y)
            data_list.append(tmp)


        # 放入datalist
        data_list = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class test(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        #define my own data from Feterpy
        data_list = []
        prefix=setting_prefix_test
        print("Prefix is: ",prefix)
        pos_list=np.load(prefix+"_test_pos.npy", allow_pickle=True)
        z_list = np.load(prefix+"_test_z.npy", allow_pickle=True)
        y_list = np.load(prefix+"_test_y.npy", allow_pickle=True)
        # r_list = np.load(prefix + "_test_r.npy", allow_pickle=True)

        for i in range(len(pos_list)):
            iz=z_list[i]
            iz=iz.astype(int)
            pos = torch.tensor(pos_list[i])
            z = torch.tensor(iz)
            y = torch.tensor(y_list[i])
            y=y.reshape([1,-1])
            # r=torch.tensor(r_list[i])
            tmp = Data(z=z, pos=pos, y=y)
            data_list.append(tmp)


        # 放入datalist
        data_list = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class valid(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        #define my own data from Feterpy
        data_list = []
        prefix=setting_prefix_valid
        print("Prefix is: ",prefix)
        pos_list=np.load(prefix+"_valid_pos.npy", allow_pickle=True)
        z_list = np.load(prefix+"_valid_z.npy", allow_pickle=True)
        y_list = np.load(prefix+"_valid_y.npy", allow_pickle=True)
        # r_list = np.load(prefix + "_test_r.npy", allow_pickle=True)

        for i in range(len(pos_list)):
            iz=z_list[i]
            iz=iz.astype(int)
            pos = torch.tensor(pos_list[i])
            z = torch.tensor(iz)
            y = torch.tensor(y_list[i])
            y=y.reshape([1,-1])
            # r=torch.tensor(r_list[i])
            tmp = Data(z=z, pos=pos, y=y)
            data_list.append(tmp)


        # 放入datalist
        data_list = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class valid_look(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        #define my own data from Feterpy
        data_list = []
        prefix=setting_prefix_valid_look
        print("Prefix is: ",prefix)
        pos_list=np.load(prefix+"_valid_pos.npy", allow_pickle=True)
        z_list = np.load(prefix+"_valid_z.npy", allow_pickle=True)
        y_list = np.load(prefix+"_valid_y.npy", allow_pickle=True)
        # r_list = np.load(prefix + "_test_r.npy", allow_pickle=True)

        for i in range(len(pos_list)):
            iz=z_list[i]
            iz=iz.astype(int)
            pos = torch.tensor(pos_list[i])
            z = torch.tensor(iz)
            y = torch.tensor(y_list[i])
            y=y.reshape([1,-1])
            # r=torch.tensor(r_list[i])
            tmp = Data(z=z, pos=pos, y=y)
            data_list.append(tmp)


        # 放入datalist
        data_list = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
'''END THE DATASET DEFINATION'''



t0=time.time()
n_ep=setting_n_ep
n_ep_save=10000
cutoff=8.0
n_block=3
npt=120;#out_channels=npt  other define x_pred
setting_batch_size=32;setting_vt_batch_size=32#32;
# setting_batch_size=64;setting_vt_batch_size=32;
setting_ini_model=False
''' END SETTING '''


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:",device)
print("EPOACH NUMBER:",n_ep)
###load dataset
dataset_train = train("train")
dataset_valid = valid("valid")
dataset_look = valid_look("valid_look")
dataset_train=dataset_train


print("Finish dataset building")


global Nf
Nf=1


def f(params,irun,str_save):
    best=999
    global Nf
    Nf=Nf+1
    print("Nf:",Nf)
    num_layers=params['num_layers']
    hidden_channels=params['hidden_channels']
    middle_channels=params['middle_channels']
    num_output_layers=params['num_output_layers']
    num_radial=params['num_radial']
    num_spherical=params['num_spherical']
    lr=params['lr']
    print("parameter:","num_layers",num_layers,"hidden_channels",hidden_channels)
    print("num_radial",num_radial,"num_spherical",num_spherical)
    print("middle_channels",middle_channels,"num_output_layers",num_output_layers)
    print("lr",lr)
    # cutoff = 8.0,
    # num_layers = 4,
    # hidden_channels = 256,
    # middle_channels = 64,
    # out_channels = 1,
    # num_radial = 3,
    # num_spherical = 2,
    # num_output_layers = 3,

    print("CUTOFF:",cutoff)
    print("NBLOCK:",n_block)
    # model = ComENet(out_channels=dim_out,cutoff=cutoff,num_layers=n_block)
    model = XAS3Dabs(out_channels=npt,cutoff=cutoff
                      ,num_layers=num_layers,hidden_channels=hidden_channels,middle_channels=middle_channels
                      ,num_output_layers=num_output_layers,num_radial=num_radial,num_spherical=num_spherical)

    import torch
    model.load_state_dict(torch.load(setting_fmodel))
    print("AFTER Model define")




    loss_func = torch.nn.L1Loss()
    import time
    ta=time.time()
    tb=time.time()
    print("TIME SPEND:",tb-ta)
    import time
    import os
    import torch
    from torch.optim import Adam
    from torch_geometric.data import DataLoader
    import numpy as np
    from torch.autograd import grad
    #from torch.utils.tensorboard import SummaryWriter
    from torch.optim.lr_scheduler import StepLR
    from tqdm import tqdm
    epochs=n_ep
    lr=lr;weight_decay=0;lr_decay_step_size=50;lr_decay_factor=0.5;
    batch_size=setting_batch_size;vt_batch_size=setting_vt_batch_size;
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

    train_loader = DataLoader(dataset_train, batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, vt_batch_size, shuffle=False)
    look_loader = DataLoader(dataset_look, vt_batch_size, shuffle=False)

    best_valid = float('inf')
    best_test = float('inf')

    save_loss=[]
    save_loss_valid=[]
    save_loss_look=[]
    save_time=[]
    for epoch in range(1, epochs + 1):
        ta=time.time()
        print("\n=====Epoch {}".format(epoch), flush=True)
        ###BEGIN TRAIN
        loss_func = torch.nn.L1Loss()
        ###train
        model.train()
        loss_accum = 0
        for step, batch_data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            out = model(batch_data)
            # loss = loss_func(out, batch_data.y.unsqueeze(1))
            # print("NOT CHANGE")
            loss = loss_func(out, batch_data.y)
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
        train_mae = loss_accum / (step + 1)
        tb=time.time()
        save_time.append(tb-ta)
        save_loss.append(train_mae)
        ###valid
        loss_accum_valid = 0
        for step_valid, batch_valid in enumerate(tqdm(valid_loader)):
            batch_valid = batch_valid.to(device)
            out_valid = model(batch_valid)
            loss_valid = loss_func(out_valid, batch_valid.y)
            loss_accum_valid += loss_valid.detach().cpu().item()
        valid_mae = loss_accum_valid / (step_valid + 1)
        save_loss_valid.append(valid_mae)
        ###look
        loss_accum_look = 0
        for step_look, batch_look in enumerate(tqdm(look_loader)):
            batch_look = batch_look.to(device)
            out_look = model(batch_look)
            loss_look = loss_func(out_look, batch_look.y)
            loss_accum_look += loss_look.detach().cpu().item()
        look_mae = loss_accum_look / (step_look + 1)
        save_loss_look.append(look_mae)
        print("MAE_VALID_FeCoNiCuZn", valid_mae,"MAE_LOOK_Ni",look_mae)

        print("MAE_TRAIN:", train_mae, "MAE_VALID_FeCoNiCuZn", valid_mae,"MAE_LOOK_Ni",look_mae)
  
        if valid_mae < best:
            
            torch.save(model.state_dict(), 'best'+str_save+'.pt')
            print("BETTER Model saved")
            xp_list=[]
            xt_list=[]
            i = 0
            loss_accum_valid = 0
            for step_valid, batch_valid in enumerate( tqdm(valid_loader) ):
                batch_valid = batch_valid.to(device)
                out_valid = model(batch_valid)
                loss_valid = loss_func(out_valid, batch_valid.y)
                loss_accum_valid += loss_valid.detach().cpu().item()

                xt_list.append(out_valid.cpu().detach().numpy())
                xp_list.append(batch_valid.y.cpu().detach().numpy())
                # np.savetxt("xas_pre_test.txt", xas_pre_test)
                # np.savetxt("xas_true_test.txt", xas_true_test)
                i=i+1
            xp=np.vstack(xp_list)
            xt=np.vstack(xt_list)
            np.savetxt("yp_"+str_save+".txt", xp)
            np.savetxt("yt_"+str_save+".txt", xt)
            mae_saved = loss_accum_valid / (step_valid + 1)
            best=mae_saved
            print("BEST",best)
            print("Debug")
        


        if epoch%5==0:
            torch.save(model.state_dict(), 'tmp.pt')
            print("Model saved")
        # scheduler.step()
    
    loss_mat=np.vstack([np.array(save_loss),np.array(save_loss_valid),np.array(save_loss_look)])
    np.savetxt("lossmat_"+str_save+".txt",loss_mat.T)
    # np.savetxt("loss"+str_save+".txt",np.array(save_loss))
    # np.savetxt("loss_valid"+str_save+".txt",np.array(save_loss_valid))

    save_time=np.array(save_time)
    print("TIME AVERAGE:",save_time.mean())

    res=save_loss_valid[-10:]
    res=np.array(res)
    res=res.mean()




    # ipar = dict(  dict({'hidden_channels': int(tmp[0]), 'num_layers': int(tmp[1]), 'lr': tmp[2],'num_radial':int(tmp[3]), 'num_spherical':int(tmp[4]) ,'middle_channels':int(tmp[5]) ,  'num_output_layers':int(tmp[6])    } )  )

    # aa=np.array([res,num_layers,hidden_channels,middle_channels,num_output_layers,num_radial,num_spherical])
    with open("hyper"+str(irun)+".log", "a+") as flog:
        print(best,hidden_channels,num_layers,lr,num_radial,num_spherical,middle_channels,num_output_layers,file=flog)

    return res



def f2(params):
    num_layers=params['num_layers']
    hidden_channels=params['hidden_channels']
    print(num_layers,hidden_channels)
    # middle_channels=params['middle_channels']
    # num_output_layers=params['num_output_layers']
    # num_radial=params['num_radial']
    # num_spherical=params['num_spherical']
    return None


def run(irun):
    fname = "./name" + str(irun) + ".txt"
    parmat=np.loadtxt(fname)
    if parmat.ndim==1:
        tmp=parmat
        str_list=[str(int(tmp[0])),str(int(tmp[1])),str(tmp[2]),str(int(tmp[3])),str(int(tmp[4])),str(int(tmp[5])),str(int(tmp[6]))]
        str_save="_".join(str_list)
        ipar = dict(  dict({'hidden_channels': int(tmp[0]), 'num_layers': int(tmp[1]), 'lr': tmp[2],'num_radial':int(tmp[3]), 'num_spherical':int(tmp[4]) ,'middle_channels':int(tmp[5]) ,  'num_output_layers':int(tmp[6])    } )  )
        
        print(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])
        f(ipar, irun,str_save)
    else:
        for i in range(parmat.shape[0]):
            tmp=parmat[i]
            str_list=[str(int(tmp[0])),str(int(tmp[1])),str(tmp[2]),str(int(tmp[3])),str(int(tmp[4])),str(int(tmp[5])),str(int(tmp[6]))]
            str_save="_".join(str_list)
            ipar = dict(  dict({'hidden_channels': int(tmp[0]), 'num_layers': int(tmp[1]), 'lr': tmp[2],'num_radial':int(tmp[3]), 'num_spherical':int(tmp[4]) ,'middle_channels':int(tmp[5]) ,  'num_output_layers':int(tmp[6])    } )  )
            print(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5])
            f(ipar,irun,str_save)
    return None



# tmp=np.array([3,64,256,2,6,3])
# ipar = dict(
#     {'num_layers': tmp[0], 'hidden_channels': int(tmp[1]), 'middle_channels': tmp[2]
#         , 'num_output_layers': tmp[3], 'num_radial': tmp[4], 'num_spherical': tmp[5]}
# )
# print(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])
# f(ipar,1)


if __name__ == '__main__':
    import sys
    ta=time.time()
    run( int(sys.argv[1]) )
    tb=time.time()
    with open("hyper_time"+str(sys.argv[1])+".log", "a+") as flog:
        print(tb-ta,file=flog)




# print("Grid search:")
# for num_layers in [3]:
#     for hidden_channels in [64,128,256]:
#         for middle_channels in [64,128,256]:
#             for num_output_layers in [2,3,4]:
#                 for num_radial in [3,6,12]:
#                     for num_spherical in [3,6]:
#
#                         ipar = dict(
#                             {'num_layers': num_layers, 'hidden_channels': hidden_channels, 'middle_channels': middle_channels
#                             ,'num_output_layers': num_output_layers, 'num_radial': num_radial, 'num_spherical': num_spherical})
#                         f(ipar)
