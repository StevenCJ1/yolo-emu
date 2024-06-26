import time
import torch
from torch.serialization import load
from utils.nnetworkutils import load_model_mlp,  index_dict_3, get_computenum
from utils.packetutils import *
from utils.yohobuffer import YOHOBuffer
from utils.mobilenet_part_split import Part_Conv_3, Part_FC_3
from utils.spliter import Spliter
from utils.combiner import Combiner
from utils.spliter_combiner import Spliter_combiner
from thop import profile
model_mlp = load_model_mlp()

print("------------------------------------- MLP ----------------------------------------")
# genarate data
print("*** gearate data for baseline result !")
data_mlp = get_mlp_data_tensors()


print("---------------------------------- start runninng -----------------------------------")
data_baseline = data_mlp
model = model_mlp
# model 0
data_test = data_baseline
flops, params = profile(model[0], inputs=(data_test, ))
print("input model[0] shape: ", data_test.shape)
print(f"flops: {flops/10**9}")

# model 1
data_test = model[0](data_test)
flops, params = profile(model[1], inputs=(data_test, ))
print("input model[1] shape: ", data_test.shape)
print(f"flops: {flops/10**9}")

# model 2
data_test = model[1](data_test)
flops, params = profile(model[2], inputs=(data_test, ))
print("input model[2] shape: ", data_test.shape)
print(f"flops: {flops/10**9}")

data_test = model[2](data_test)


print("---------------------------------- time running  -----------------------------------")
t_part = []
for i in range(3):
    t_start = time.time()
    print(f"*** running model: {i+1} out of {len(model)} !")
    data_baseline = model[i](data_baseline)  
    t_end = time.time()
    t_part.append(t_end - t_start)
    print(f"*** part {i+1} time: {t_end - t_start}")
print(type(data_baseline))
print(f"*** total part time : {sum(t_part)}")
'''
# split mode 
mode = 1
n_split_client = 4
n_combiner_client = 1
compute_num = 3

print(data_mlp.shape)
sc = Spliter_combiner(None, LENGTH, n_split_client, n_combiner_client, -1, -1, n_combiner_client, mode)
# split
sc.split_list = sc.spliter.split(data_mlp)
print("split data shape:", sc.split_list[0].shape)


for j in range(compute_num):
    for i in range(len(sc.split_list)):
        sc.split_list[i] = model_mlp[j](sc.split_list[i])

sc.combine_split_list()
'''