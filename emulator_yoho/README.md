# Emulator_IA_Net_Lite
this folder base on repos 
[ia-net-lite-emu](https://github.com/Huanzhuo/ia-net-lite-emu/tree/dev-wu).

we ported the code, and let it run in the pICA-emu and support `simpleemu`.

## Environment setting
base on this repos `README.md`, we successfully installed docker image `pica_dev:4`, using 
```shell
docker image ls | grep "pica_dev"
```
to check if this image successful builded.

then run this image, open a shell with bash in this docker image.
```shell
docker run -i -t pica_dev:4 /bin/bash
```
for user in china, you should change the pip source to speed up install package
```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
install following package with pip: `cffi progressbar2 museval scapy librosa torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1`
stop this docker container and using this container id, build a new image `pica_torch:1.0`
```shell
docker commit -a "naibaoofficial" -m "build env for ia_net_lite" <container_id> pica_torch:1.0
```
## Start Topo
using following command to start the topology for ia_net_lite, we assumed that you have alreadyed login in `testbed`
```shell
cd \vagrant\emulator_ia_net_lite
sudo python3 topo.py
```
## Runing testcase

### Login server, client, vnf1, and vnf2 inside the corresponded container.

In this repos, the shell has been already opened.

```bash
mininet> xterm client server vnf1 vnf2
```

Then four windows are popped up, you can identify client, server and two VNFs by looking at the host name (e.g. `@client`) in the shell. Then please firstly run `server.py` inside the server's shell and then `client.py` in the clients shell (use `-h` to check the CLI options).

Run server, vnf1, vnf2, and client with traffic filtering. Currently, this order must be kept manually. With the flag `testid`, the number of traffic filters, which are applied in the network, can be defined. Please use the following sequence:

```bash
# on the server
python3 ./server.py --testid n

# on the vnf1
python3 ./vnf.py --switchid 1 --testid n

# on the vnf2
python3 ./vnf.py --switchid 2 --testid n

# on the client, change epochs to modify running rounds (e.g. 60)
python3 ./client.py --epochs 60 --testid n
```

Measurement results are stored in ```measurements/testcase_n```.

### Test cases
   
    | test case id |   client   |   switch 1   |   switch 2   |    server   |
    | -------------|:----------:|:------------:|:------------:|:-----------:|
    |       0      |    none    |     none     |     none     | model 1 2 3 |
    |       1      |  model 1   |     none     |     none     | model 2 3   |
    |       2      |  model 1 2 |     none     |     none     | model 3     |
    |       3      | model 1 2 3|     none     |     none     |     none    |
    |       4      |  model 1   |     none     |     none     | model 2 3   |
    |       5      |  model 1   |    model 2   |     none     | model 3     |
    |       6      |  model 1   |    model 2   |     model 3  |     none    |

## TODO
- 全改成`Yoho`，不要`pica`，`ia_net_lite`
- 删除不要的，整改debug，修改文档。
- 文件夹遵循`pICA`。
