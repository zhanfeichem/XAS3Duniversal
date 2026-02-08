# XAS3Duniversal
## Running
First, copy the dataset from the "dataset_Ni" folder to the "run_MOE" "run_energyrange","run_one_element",folder, or change the dataset file location in the code.</br>
The meaning of Parameter 99 is to call the hyperparameter file name99.txt.</br>
"run_one_element" simple example:  
python valid_XAS3Dabs_Ni_3d.py 99</br>

"run_Ni50FeCoCuZn" Data Experiment to Verify the Enhancement of XAS Predictive Capability for Ni-Deficient Samples by FeCoCuZn-Containing Samples:  
python valid_XAS3Dabs_Ni_3d.py 99</br>

"run_energyrange" prexanesmodel using two XAS3D model for two pre-edge feature and XANES prediction:  
python valid_prexanes.py 99</br>

"run_MOE" MOE multi-expert model using XAS3D as module:  
python valid_moe.py 99</br>

"expalin_3d4dLa" Model interpretability </br>



## Installation Overview
conda create -n pyg_pl python==3.9 </br>
source /scratchfs/heps/zhanf/miniconda3/bin/activate pyg_pl </br>
pip install pytorch-lightning==1.8 -i https://mirrors.ustc.edu.cn/pypi/web/simple </br>
pip install tqdm </br>
### PYG installation </br>
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html  -i https://pypi.tuna.tsinghua.edu.cn/simple  </br>
pip install torch-sparse torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html  </br>
pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html   </br>
pip install torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html  </br>
pip install torchmetrics==0.7 </br>
### Fitting related package </br>
pip install sympy </br>
pip install nlopt </br>
pip install pymatgen </br>
pip install frechetdist </br>
#



