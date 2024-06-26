import torch
import torch.nn as nn
from models.dual_mobile_net import *
from utils.packetutils import *



MODEL_PATH_LIST = ["./results/distributed_models/part_1.pt",
                   "./results/distributed_models/part_2.pt",
                   "./results/distributed_models/part_3.pt"]

def load_model(model_path):
    model = torch.load(model_path, map_location="cpu")["model"].eval()
    model.mode = "val"
    return model

def combine_model(ensemble_model, model_list):
    ensemble_model.conv1 = model_list[0].conv1
    ensemble_model.bn1 = model_list[0].bn1
    ensemble_model.relu = model_list[0].relu
    ensemble_model.layer1 = model_list[0].layer1
    ensemble_model.layer2 = model_list[0].layer2
    ensemble_model.layer3 = model_list[0].layer3
    ensemble_model.layer4 = model_list[0].layer4

    ensemble_model.layer5 = model_list[1].layer5
    ensemble_model.layer6 = model_list[1].layer6

    ensemble_model.layer7 = model_list[2].layer7
    ensemble_model.conv8 = model_list[2].conv8
    ensemble_model.avgpool = model_list[2].avgpool
    ensemble_model.fc = model_list[2].fc
    ensemble_model.num_emed = model_list[2].num_emed
    ensemble_model.n_spk = model_list[2].n_spk
    return ensemble_model

if __name__ == "__main__":
    model = []
    data_baseline = get_network_data_tensors()
    data_test = data_baseline

    # splited
    for i in range(len(MODEL_PATH_LIST)):
        model.append(load_model(MODEL_PATH_LIST[i]))
    for i in range(3):
        print(f"*** running model: {i + 1} out of {len(model)} !")
        data_baseline = model[i](data_baseline)

    # combined
    ensemble_model = mobilenet_19(mode='valid')
    ensemble_model = combine_model(ensemble_model, model)

    # difference
    fc_data = ensemble_model.forward(data_test)
    ans = []
    for i in range(4):
        d = (data_baseline[i] - fc_data[i]).pow(2).sum(1)
        ans.append(d.item())
    print(f"*** main function: distence: {ans}")
    torch.save(ensemble_model, './results/ensemble_model/mobilenetV2_19.pt')
