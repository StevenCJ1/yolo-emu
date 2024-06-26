import torch
import torch.nn as nn


class MLP1(nn.Module):
    def __init__(self, in_chs) -> None:
        super(MLP1, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(in_chs, 512)
        self.l2 = nn.Linear(512, 256)
        
    def forward(self, x):
        x = x.reshape(-1, 3072)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return x
    
    
class MLP2(nn.Module):
    def __init__(self) -> None:
        super(MLP2, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 64)
        
    def forward(self, x):
        x = x.reshape(-1, 256)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return x    
    
    
class MLP3(nn.Module):
    def __init__(self) -> None:
        super(MLP3, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = x.reshape(-1, 64)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = torch.argmax(x, dim=1)
        return x
    
if __name__ == "__main__":
    model1 = MLP1(3072)
    model2 = MLP2()
    model3 = MLP3()
    infer_in = torch.randint(0, 255, [128, 3072], dtype=torch.float32)
    infer_in = infer_in / 255.0
    out1 = model1(infer_in)
    out2 = model2(out1)
    out3 = model3(out2)
    torch.save(model1.state_dict(), "/home/lukashe/data/projects/yoho-emu-dev-jiakang-v3/yoho-emu-dev-jiakang/cnn_mlp/params/mlp/part1.pth")
    torch.save(model2.state_dict(), "/home/lukashe/data/projects/yoho-emu-dev-jiakang-v3/yoho-emu-dev-jiakang/cnn_mlp/params/mlp/part2.pth")
    torch.save(model3.state_dict(), "/home/lukashe/data/projects/yoho-emu-dev-jiakang-v3/yoho-emu-dev-jiakang/cnn_mlp/params/mlp/part3.pth")