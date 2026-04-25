import json
import argparse
import os.path as osp
import os
import time
from tqdm import tqdm
###torch
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
###PYG
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
######
from scipy.interpolate import interp1d
import numpy as np
##########output
#from pymatgen.io.cif import CifWriter
##########GNN model
# from myGNN import MYGNN
# from comenet import ComENet
# from comenet_f1r import  ComENetF1R
# from comenet_f1 import  ComENetF1
from  XAS3D2026 import XAS3Dabs
# from input_Fe3O4 import *
# from input_Mn  import *

#################
import matplotlib.pyplot as plt


setting_f_model="model_example_MnJT_best256_3_0.0005_5_2_82_5.pt"
setting_prefix1="Mn_example"
#model = ComENetF1(out_channels=npt,cutoff=cutoff,num_layers=3,hidden_channels=64,middle_channels=256,num_output_layers=2,num_radial=6,num_spherical=3)
model = model = XAS3Dabs(out_channels=241,cutoff=8.0,num_layers=3,hidden_channels=256,middle_channels=82,num_output_layers=5,num_radial=5,num_spherical=2)
model.load_state_dict(torch.load(setting_f_model))


# print("####################INPUT####################")
# print("INPUT prefix is",setting_prefix)
# print("INPUT Nsite is",setting_Nsite)
# print("INPUT Nstruct is",setting_Nstruct)
# print("INPUT TF_conv is",setting_TF_conv)
# print("INPUT TF_sort_atom is",setting_TF_sort_atom)
# print("INPUT TF_absorber_IHS",setting_TF_absorber_IHS)
# print("####################INPUT####################")
# print("INPUT n_ep:",setting_n_ep)
# print("INPUT n_ep_save:",setting_n_ep_save)
# print("INPUT npt:",setting_npt)
# print("INPUT npar :",setting_npar)
# print("INPUT f_exp:",setting_f_exp)
# print("####################INPUT####################")



class valid1(InMemoryDataset):
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
        prefix=setting_prefix1
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






device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:",device)

model = model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f'#Params: {num_params}')
###load dataset

dataset_valid1 = valid1("valid1")
valid_loader1 = DataLoader(dataset_valid1, 32, shuffle=False)

lr=0.0005;weight_decay=0;lr_decay_step_size=50;lr_decay_factor=0.5;
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

best_valid = float('inf')
best_test = float('inf')
save_loss=[]
save_loss_valid=[]
save_time=[]
loss_func = torch.nn.L1Loss()
# for epoch in range(1):
#     #valid
#     loss_accum_valid = 0
#     for step_valid, batch_valid in enumerate(tqdm(valid_loader1)):
#         batch_valid = batch_valid.to(device)
#         out_valid = model(batch_valid)
#         loss_valid = loss_func(out_valid, batch_valid.y)
#         loss_accum_valid += loss_valid.detach().cpu().item()
#     valid_mae = loss_accum_valid / (step_valid + 1)
#     save_loss_valid.append(valid_mae)
    
#     print("MAE_VALID", valid_mae)

############################################################################################################
import nlopt
from numpy import *
import pymatgen.io.feff as feff
from pymatgen.core import Structure,Lattice
from pymatgen.core.periodic_table import Element
# from frechetdist import frdist




def integral(x,y):
    my = (y[1:]+y[:-1])/2
    dx = x[1:]-x[:-1]
    return np.sum(my*dx)
def kernelCauchy(x, a, sigma): return sigma/2/math.pi/((x-a)**2+sigma**2/4)
def kernelGauss(x, a, sigma): return 1/sigma/math.sqrt(2*math.pi)*np.exp(-(x-a)**2/2/sigma**2)
def YvesWidth(e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    ee = (e-Efermi)/Ecent
    ee[ee==0] = 1e-5
    return Gamma_hole + Gamma_max*(0.5+1/math.pi*np.arctan( math.pi/3*Gamma_max/Elarg*(ee-1/ee**2) ))
def smooth_fdmnes(e, xanes, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi):
    xanes = np.copy(xanes)
    lastValueInd = xanes.size - int(xanes.size*0.05)
    #lastValue = utils.integral(e[lastValueInd:], xanes[lastValueInd:])/(e[-1] - e[lastValueInd])
    lastValue = integral(e[lastValueInd:], xanes[lastValueInd:]) / (e[-1] - e[lastValueInd])
    E_interval = e[-1] - e[0]
    xanes[e<Efermi] = 0
    sigma = YvesWidth(e, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    virtualStartEnergy = e[0]-E_interval; virtualEndEnergy = e[-1]+E_interval
    norms = 1.0/math.pi*( np.arctan((virtualEndEnergy-e)/sigma*2) - np.arctan((virtualStartEnergy-e)/sigma*2) )
    toAdd = 1.0/math.pi*( np.arctan((virtualEndEnergy-e)/sigma*2) - np.arctan((e[-1]-e)/sigma*2) ) * lastValue
    kern = kernelCauchy(e.reshape(-1,1), e.reshape(1,-1), sigma.reshape(-1,1))
    assert (kern.shape[0]==e.size) and (kern.shape[1]==e.size)
    de = (e[1:]-e[:-1]).reshape(1,-1);
    f = xanes.reshape(1,-1) * kern
    new_xanes = (0.5*np.sum((f[:,1:]+f[:,:-1])*de, axis=1).reshape(-1) + toAdd)/norms
    return e, new_xanes

def fdminp(pos,z,fout="fdm.inp",nameout="out"):
    feffpar=[]
    feffpar.extend("Filout\n")#output file name
    feffpar.extend(nameout+"\n")
    with open("fdmnesmodel.inp") as f:
        feffpar.extend(f.readlines())
    atom_lines=[]
    natom=len(z)
    for i in range(natom):
        tmp="%d %.5f %.5f %.5f \n"%( z[i],pos[i,0],pos[i,1],pos[i,2]   )
        atom_lines.append(tmp)
    feffpar.extend(atom_lines)
    feffpar.append("END\n")
    with open(fout,"w") as f:
        f.write( "".join(feffpar) )
    return feffpar

def gjf(pos,z,fout="opt_res.gjf"):
    feffpar=[]
    with open("model.gjf") as f:
        feffpar.extend(f.readlines())
    atom_lines=[]
    natom=len(z)
    for i in range(natom):
        tmp="%d %.5f %.5f %.5f \n"%( z[i],pos[i,0],pos[i,1],pos[i,2]   )
        atom_lines.append(tmp)
    feffpar.extend(atom_lines)
    with open(fout,"w") as f:
        f.write( "".join(feffpar) )
    return feffpar

def getz(atom_list):
    cluster=atom_list.cluster
    z_array=np.zeros(len(cluster))
    for i in range( len(cluster) ):
        iele=Element(cluster[i].species_string)
        z_array[i]=iele.Z
    return z_array

def res_r2(y,yp):
    aa = np.square(y-yp)
    bb = np.square(y)
    tmp=sum(aa)/sum(bb)
    return tmp
def mae_weight(y,yp):
    global weight_obj
    tmp=np.abs(y-yp)
    tmp=weight_obj*tmp
    res=np.mean(tmp)
    return res

def diff_exp_the_save(energy_exp,muexp,energy,mu):
    # fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    # fexp = interp1d(xexp,yexp,kind="cubic",fill_value='extrapolate')
    # energy_use=xexp#using exp energy grid
    # muexp=fexp(energy_use)
    fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    muthe=fthe(energy_exp)
    # res = mae_weight(muexp, muthe)
    # res=np.mean(np.abs(muexp - muthe))
    res = res_r2(muexp, muthe)
    # res=F.l1_loss( torch.Tensor(muexp),torch.Tensor(muthe) )#ERROR
    return res

def diff_exp_the(energy_exp,muexp,energy,mu):
    # fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    # fexp = interp1d(xexp,yexp,kind="cubic",fill_value='extrapolate')
    # energy_use=xexp#using exp energy grid
    # muexp=fexp(energy_use)
    fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    muthe=fthe(energy_exp)
    res=np.mean(np.abs(muexp - muthe))
    # res = mae_weight(muexp, muthe)
    
    res = res_r2(muexp, muthe)
    # res=F.l1_loss( torch.Tensor(muexp),torch.Tensor(muthe) )#ERROR

    return res

def diff_exp_the_weight(energy_exp,muexp,energy,mu):
    fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    muthe=fthe(energy_exp)
    res=mae_weight(muexp,muthe)
    return res


def diff_exp_the_obj(energy_exp,muexp,energy,mu):
    res=diff_exp_the(energy_exp,muexp,energy,mu)
    return res
def diff_exp_the_obj2(energy_exp,muexp,energy,mu):
    res=diff_exp_the(energy_exp,muexp,energy,mu)
    return res
def diff_exp_the_frechet(energy_exp,muexp,energy,mu):
    fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    # fexp = interp1d(xexp,yexp,kind="cubic",fill_value='extrapolate')
    # energy_use=xexp#using exp energy grid
    muthe=fthe(energy_exp)
    # muexp=fexp(energy_use)
    spcexp=np.vstack([energy_exp,muexp]).T
    spcthe=np.vstack([energy_exp,muthe]).T
    # res=res_r2(muexp,muthe)
    res=frdist(spcexp,spcthe)
    return res
def diff_exp_the_pearson(energy_exp,muexp,energy,mu):
    fthe = interp1d(energy, mu, kind="cubic", fill_value='extrapolate')
    # fexp = interp1d(xexp,yexp,kind="cubic",fill_value='extrapolate')
    # energy_use=xexp#using exp energy grid
    muthe=fthe(energy_exp)
    # muexp=fexp(energy_use)
    res=-pearson(muexp,muthe)
    return res

def pearson(arr1, arr2):
    return np.corrcoef(arr1, arr2)[0][1]

##############################################
##########BASIC
setting_R=5.0
setting_Nsite=1 #Number of Absorber
setting_f_element="Mn_start.ele"
setting_f_cart="Mn_start.cart"
setting_abc=[5.5, 5.5, 8.5,90,90,90]
##########BASIC
setting_TF_conv=True
    

setting_TF_sort_atom=False# 排序相关设置
setting_TF_absorber_IHS=True# 吸收原子是否参与IHS，兼容旧版本数据集False
##########Fitting Begin
setting_maxtime=3600*10
setting_maxeval=10000*3
setting_npar=84
setting_opt_lower_bounds=-0.3
setting_opt_upper_bounds=0.3
setting_f_exp="JT.exp"
setting_npt=241#point model output
setting_conv_xip = np.linspace(6539-20+1,6539+100,241)
setting_energy_unconv=np.linspace(6539-20+1,6539+100,241)#using for unconv model pred‘s energy grid#np.linspace(6537,6639,205)
setting_energy_output=np.linspace(6536,6632,240)
setting_energy_ori=setting_conv_xip#out = np.vstack([energy_ori, mu_pred]);np.savetxt("opt_res_mu_pred.txt", out.T)
setting_energy=setting_conv_xip#energy_es = energy+es_opt
setting_weight_xval=6592#split the weight obj
setting_Gamma_hole=1.16
setting_EFermi=6539
##############################################
global cart0
cart0 = np.loadtxt(setting_f_cart)  # car_fac0=np.loadtxt("MnO6.fac")
global vec;
vec=cart0[1:6+1]-cart0[0]
for i in range(vec.shape[0]):
    vec[i]=vec[i]/np.linalg.norm(vec[i])# vec[j,0]**2+vec[j,1]**2+vec[j,2]**2
##############################################

global NRUN
NRUN=0
global mu_pred
mu_pred=np.zeros(100)#100 dosent matter
global f,energy,energy_ori,energy_unconv,exp,weight_obj
f=model;
energy=setting_energy#np.linspace(7112-10+1,7112+100,110)
energy_ori=setting_energy_ori
energy_unconv=setting_energy_unconv
exp = np.loadtxt(setting_f_exp,comments="#")
weight_obj=1.5*(exp[:,0]>setting_weight_xval)+1*(exp[:,0]<setting_weight_xval)
global lattice,species
lattice=Lattice.from_parameters(setting_abc[0],setting_abc[1],setting_abc[2],setting_abc[3],setting_abc[4],setting_abc[5])
with open(setting_f_element) as fele:
    ele = fele.readlines()
for i in range(len(ele)):
    ele[i] = ele[i].strip()
species = ele








def opt_sub_conv():
    input_norm = 1
    norm = input_norm
    opt = nlopt.opt(nlopt.LN_NELDERMEAD,6)  # LN_COBYLA  LN_BOBYQA  GN_DIRECT LN_NELDERMEAD
    E0=setting_EFermi
    # opt.set_lower_bounds( np.array( [norm*0.7,-5,E0-5,15-10,30-10,30-10] ) )  # [-float('inf'), 0]
    # opt.set_upper_bounds( np.array( [norm*1.5, 5,E0+5,15+10,30+10,30+10] ) )
    opt.set_lower_bounds( np.array( [norm*0.9,-20,E0-10,15-10,30-10,30-10] ) )  # [-float('inf'), 0]
    opt.set_upper_bounds( np.array( [norm*1.5, 20,E0+10,15+10,30+10,30+10] ) )


    opt.set_min_objective(obj_sub_conv)
    # opt.set_maxtime(300)#2 minuit
    opt.set_maxeval(2000)
    # x0=np.array([norm*1.0,0,7112,15,30,30])
    x0=np.array([norm*1.0,0,E0,15,30,30])
    
    x = opt.optimize(x0)
    minf = opt.last_optimum_value()
    # print("Sub optimization results",x)
    return x

def obj_sub_conv(x,grad=None):
    global mu_pred
    global f, exp
    global energy_unconv


    mu=mu_pred.copy()

    Gamma_hole = setting_Gamma_hole


    norm = x[0]
    es = x[1]
    Efermi = x[2]
    Gamma_max = x[3];
    Ecent = x[4]
    Elarg = x[5]
    ee, mu = smooth_fdmnes(energy_unconv, mu, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    mu=mu*norm
    energy_es = energy_unconv+es#energy_es = energy+es
    res=diff_exp_the(exp[:,0],exp[:,1],energy_es,mu)

    return res

def cal_conv(x,mu):
    global energy_unconv
    Gamma_hole = setting_Gamma_hole
    norm = x[0]
    es = x[1]
    Efermi = x[2]
    Gamma_max = x[3];
    Ecent = x[4]
    Elarg = x[5]
    ee, mu = smooth_fdmnes(energy_unconv, mu, Gamma_hole, Ecent, Elarg, Gamma_max, Efermi)
    mu=mu*norm
    energy_es = energy_unconv+es#energy_es = energy+es

    return energy_es,mu

def obj_structure_unconv_ax_plan(x,grad=None):
    global NRUN
    NRUN=NRUN+1
    global mu_pred
    global f,energy,exp
    global lattice, species, cart0
    global vec
    ipar = x
    ###END input
    # setting_R=5.0
    # setting_npt=210#energy point
    setting_N_abs=setting_Nsite


    iax=x[0]
    iplan=x[1]
    natom = cart0.shape[0] 
    

    vec_change=np.zeros(vec.shape)
    vec_change[0]=vec[0]*iax
    vec_change[1]=vec[1]*iax
    vec_change[2]=vec[2]*iplan
    vec_change[3]=vec[3]*iplan
    vec_change[4]=vec[4]*iplan
    vec_change[5]=vec[5]*iplan
    
    


    ipar = x[2:]#after 1st coord
    ipar = ipar.reshape(natom - 6-1, 3)  # HERE not i id
    # ipar=np.zeros([natom - 2-1, 3])#substitute 2 par fit 
    
    ipar = np.insert(ipar,0,vec_change,axis=0)#insert 6 shell
    ipar = np.insert(ipar, 0, np.zeros(3), axis=0)#insert absorber
    #insert 6 shell


    icart_coords = cart0 + ipar
    istructure = Structure(lattice, species, icart_coords, coords_are_cartesian=True)  # coords_are_cartesian=True


    mu_mat=np.zeros([setting_N_abs,setting_npt])
    for j in range(setting_N_abs):
        id_abs = j  # absorber idx from 0
        iatom = feff.Atoms(istructure, id_abs, setting_R)  # iatom = feff.Atoms(istructure, "Mn", 5.0)
        iz = getz(iatom)
        ipos = iatom.cluster.cart_coords
        iz=torch.tensor(iz)
        ipos=torch.tensor(ipos)
        ta_ml=time.time()
        idata=Data(z=iz,pos=ipos)
        data_list=[idata,idata]
        loader=DataLoader(data_list,batch_size=1)
        it=loader._get_iterator()
        batch=next(it)
        batch=batch.to(device)
        mu=f(batch)
        mu=mu.cpu().detach().numpy().reshape(-1)
        tb_ml=time.time()
        tdiff_ml=tb_ml-ta_ml
        mu_mat[j]=mu.copy()

    #for opt_sub()
    mu_pred=mu_mat.mean(axis=0).reshape(-1).copy()#np.savetxt("tmp_opt_mu.txt", mu)##

    ta_sub=time.time()
    x_conv=opt_sub_conv()#es_opt=opt_sub()#norm_opt=1.0;es_opt=0#

    tb_sub=time.time()
    tdiff_sub=tb_sub-ta_sub


    energy_es,mu=cal_conv(x_conv,mu_pred)

    res=diff_exp_the_obj(exp[:,0],exp[:,1],energy_es,mu)#zhanfei 20230223
    res_save=diff_exp_the_save(exp[:,0],exp[:,1],energy_es,mu)#zhanfei 20230803


    if NRUN%100==0:
        print(NRUN,"th OBJ VAL:",res)

    with open("1optlog.txt","a") as fopt:
        fopt.write(str(res)+" "+str(res_save)+"\n")

    #look structure
    if False:
        iatom10 = feff.Atoms(istructure, id_abs, 7.5)  # iatom = feff.Atoms(istructure, "Mn", 5.0)
        iz10 = getz(iatom10)
        ipos10 = iatom10.cluster.cart_coords
        iz10=torch.tensor(iz10)
        ipos10=torch.tensor(ipos10)
        fdminp(ipos10, iz10, fout="opt_res_fdmnes.inp", nameout="xanes_fit")
        gjf(ipos,iz,fout="opt_res.gjf")
        from pymatgen.io.cif import CifWriter
        cw=CifWriter(istructure)
        cw.write_file("opt_res.cif")
        ##########save opt res
        fthe=interp1d(energy_es,mu,kind="cubic",fill_value='extrapolate')
        fexp=interp1d(exp[:,0],exp[:,1],kind="cubic",fill_value='extrapolate')
        energy_use=exp[:,0]#using exp energy grid
        muthe=fthe(energy_use)
        muexp=fexp(energy_use)
        np.savetxt("opt_res_x.txt", x)
        out = np.vstack([energy_ori, mu_pred])
        np.savetxt("opt_res_mu_pred.txt", out.T)
        out=np.vstack([energy_use,muexp,muthe])
        np.savetxt("opt_res_e_exp_the.txt",out.T)
        global setting_energy_output
        muthe_output=fthe(setting_energy_output)
        muexp_output=fexp(setting_energy_output) 
        out=np.vstack([setting_energy_output,muexp_output,muthe_output])
        np.savetxt("opt_res_e_exp_the_output.txt",out.T)  
        import matplotlib.pyplot as plt
        plt.plot(setting_energy_output,muthe_output,'r-',label="muthe_output")
        plt.plot(setting_energy_output,muexp_output,'k-',label="muexp_output")
        plt.legend()
        plt.savefig("opt_res_e_exp_the_output.png")
        plt.show()




    return res

def obj_structure_unconv_ax_plan_plot(x,grad=None):
    global NRUN
    NRUN=NRUN+1
    global mu_pred
    global f,energy,exp
    global lattice, species, cart0
    global vec
    ipar = x
    ###END input
    # setting_R=5.0
    # setting_npt=210#energy point
    setting_N_abs=setting_Nsite


    iax=x[0]
    iplan=x[1]
    natom = cart0.shape[0] 
    

    vec_change=np.zeros(vec.shape)
    vec_change[0]=vec[0]*iax
    vec_change[1]=vec[1]*iax
    vec_change[2]=vec[2]*iplan
    vec_change[3]=vec[3]*iplan
    vec_change[4]=vec[4]*iplan
    vec_change[5]=vec[5]*iplan
    
    


    ipar = x[2:]#after 1st coord
    ipar = ipar.reshape(natom - 6-1, 3)  # HERE not i id
    # ipar=np.zeros([natom - 2-1, 3])#substitute 2 par fit 
    
    ipar = np.insert(ipar,0,vec_change,axis=0)#insert 6 shell
    ipar = np.insert(ipar, 0, np.zeros(3), axis=0)#insert absorber
    #insert 6 shell


    icart_coords = cart0 + ipar
    istructure = Structure(lattice, species, icart_coords, coords_are_cartesian=True)  # coords_are_cartesian=True


    mu_mat=np.zeros([setting_N_abs,setting_npt])
    for j in range(setting_N_abs):
        id_abs = j  # absorber idx from 0
        iatom = feff.Atoms(istructure, id_abs, setting_R)  # iatom = feff.Atoms(istructure, "Mn", 5.0)
        iz = getz(iatom)
        ipos = iatom.cluster.cart_coords
        iz=torch.tensor(iz)
        ipos=torch.tensor(ipos)
        ta_ml=time.time()
        idata=Data(z=iz,pos=ipos)
        data_list=[idata,idata]
        loader=DataLoader(data_list,batch_size=1)
        it=loader._get_iterator()
        batch=next(it)
        batch=batch.to(device)
        mu=f(batch)
        mu=mu.cpu().detach().numpy().reshape(-1)
        tb_ml=time.time()
        tdiff_ml=tb_ml-ta_ml
        mu_mat[j]=mu.copy()

    #for opt_sub()
    mu_pred=mu_mat.mean(axis=0).reshape(-1).copy()#np.savetxt("tmp_opt_mu.txt", mu)##

    ta_sub=time.time()
    x_conv=opt_sub_conv()#es_opt=opt_sub()#norm_opt=1.0;es_opt=0#

    tb_sub=time.time()
    tdiff_sub=tb_sub-ta_sub


    energy_es,mu=cal_conv(x_conv,mu_pred)

    res=diff_exp_the_obj(exp[:,0],exp[:,1],energy_es,mu)#zhanfei 20230223
    res_save=diff_exp_the_save(exp[:,0],exp[:,1],energy_es,mu)#zhanfei 20230803


    if NRUN%100==0:
        print(NRUN,"th OBJ VAL:",res)

    with open("1optlog.txt","a") as fopt:
        fopt.write(str(res)+" "+str(res_save)+"\n")

    #look structure
    if True:
        iatom10 = feff.Atoms(istructure, id_abs, 7.5)  # iatom = feff.Atoms(istructure, "Mn", 5.0)
        iz10 = getz(iatom10)
        ipos10 = iatom10.cluster.cart_coords
        iz10=torch.tensor(iz10)
        ipos10=torch.tensor(ipos10)
        fdminp(ipos10, iz10, fout="opt_res_fdmnes.inp", nameout="xanes_fit")
        gjf(ipos,iz,fout="opt_res.gjf")
        from pymatgen.io.cif import CifWriter
        cw=CifWriter(istructure)
        cw.write_file("opt_res.cif")
        ##########save opt res
        fthe=interp1d(energy_es,mu,kind="cubic",fill_value='extrapolate')
        fexp=interp1d(exp[:,0],exp[:,1],kind="cubic",fill_value='extrapolate')
        energy_use=exp[:,0]#using exp energy grid
        muthe=fthe(energy_use)
        muexp=fexp(energy_use)
        np.savetxt("opt_res_x.txt", x)
        out = np.vstack([energy_ori, mu_pred])
        np.savetxt("opt_res_mu_pred.txt", out.T)
        out=np.vstack([energy_use,muexp,muthe])
        np.savetxt("opt_res_e_exp_the.txt",out.T)
        global setting_energy_output
        muthe_output=fthe(setting_energy_output)
        muexp_output=fexp(setting_energy_output) 
        out=np.vstack([setting_energy_output,muexp_output,muthe_output])
        np.savetxt("opt_res_e_exp_the_output.txt",out.T)  
        import matplotlib.pyplot as plt
        plt.plot(setting_energy_output,muthe_output,'r-',label="muthe_output")
        plt.plot(setting_energy_output,muexp_output,'k-',label="muexp_output")
        plt.legend()
        plt.savefig("opt_res_e_exp_the_output.png")
        plt.show()




    return res

with open("1optx.txt","w") as foptx:
    foptx.write("#x value of opt steps"+"\n")
with open("1optlog.txt","w") as fopt:
    fopt.write("#obj saved_r2"+"\n")
npar=(28-7)*3+2
opt = nlopt.opt(nlopt.GN_DIRECT_L, npar)  #GN_DIRECT_L LN_NELDERMEAD LN_COBYLA  LN_BOBYQA    GN_ESCH GN_ISRES  GN_CRS2_LM   GN_MLSL  GN_AGS
opt.set_lower_bounds( -0.3*np.ones(npar) )  # [-float('inf'), 0]
opt.set_upper_bounds( 0.3*np.ones(npar)  )
opt.set_min_objective(obj_structure_unconv_ax_plan)

opt.set_maxeval(3)

ta = time.time()
print("Begin structure fitting")
x = opt.optimize(0*np.ones(npar))
tb = time.time()
minf = opt.last_optimum_value()
print("Finish Auto Structure Fitting")
print("Rfactor:",minf)
print("Fitting Results is: ", x)
print("Structure Fitting Spend Time: ", tb - ta)
print("Step:",NRUN)
print("Successful termination (positive return values)",opt.last_optimize_result())
np.savetxt("opt_res_x_nlopt.txt", x)



xx=np.loadtxt("opt_res_x_nlopt.txt")
# x1[0]=-0.03;x1[1]=-0.26
res=obj_structure_unconv_ax_plan_plot(xx,grad=None)
print("Rfactor Validation Obj value:",res)
print("PAUSE")
print("DEBUG")
























