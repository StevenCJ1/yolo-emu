import time
import torch
from torch.serialization import load
from utils.nnetworkutils import load_model, MODEL_PATH_LIST,  index_dict_3, get_computenum
from utils.packetutils import *
from utils.yohobuffer import YOHOBuffer
from utils.mobilenet_part_split import Part_Conv_3, Part_FC_3
from utils.spliter import Spliter
from utils.combiner import Combiner

model = []

for i in range(len(MODEL_PATH_LIST)):
    model.append(load_model(MODEL_PATH_LIST[i]))

part_3_conv = Part_Conv_3(model[2], mode="inference")
part_3_fc = Part_FC_3(model[2], mode="inference")

spliter = None
combiner2 = None
combiner3 = None
combiner_list = []

print("------------------------------- calculate part result------------------------------")
# genarate data
print("*** gearate data for baseline result !")
data_baseline = get_network_data_tensors()

spliter = Spliter(model[0],64000,256,8, 0)
model[0](data_baseline)


start1 = time.time()
model[0](data_baseline)
end1 = time.time()-start1

# split
end2 = 0
data_list = spliter.split(data_baseline)
for data in data_list:
    start2 = time.time()
    model[0](data)
    end2 += time.time() - start2

b = (end2 - end1)/7
print("beta",b) 
print("alpha", 0.22 * 8192 /(end1 - b))



