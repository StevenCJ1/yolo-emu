from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo

class CustomTopo(Topo):
    def build(self):
        # 添加主机
        client = self.addHost('client')
        server = self.addHost('server')
        vnf0 = self.addHost('vnf0')
        vnf1 = self.addHost('vnf1')
        vnf2 = self.addHost('vnf2')

        # 添加交换机
        s0 = self.addSwitch('s0')
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')

        # 创建链路
        self.addLink(client, s0, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s0, vnf0, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s0, s1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s1, vnf1, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s1, s2, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s2, vnf2, cls=TCLink, bw=10, delay='5ms')
        self.addLink(s2, server, cls=TCLink, bw=10, delay='5ms')

def run():
    topo = CustomTopo()
    net = Mininet(topo=topo, switch=OVSSwitch, controller=Controller)
    net.start()

    # 显示每个交换机的接口
    for switch in net.switches:
        print(f"Interfaces on {switch.name}:")
        print(switch.cmd('ifconfig'))

    # 测试连接
    net.pingAll()
    net.stop()

if __name__ == '__main__':
    run()
