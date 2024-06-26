import time
from simpleemu.simpleudp import simpleudp
from simpleemu.simplecoin import SimpleCOIN

# Network
serverAddressPort = ("10.0.0.15", 9999)
clientAddressPort = ("10.0.0.12", 9999)
ifce_name, node_ip = simpleudp.get_local_ifce_ip('10.0.')

# Simple coin
app = SimpleCOIN(ifce_name=ifce_name, n_func_process=1, lightweight_mode=True)

@app.main()
def main(simplecoin: SimpleCOIN.IPC, af_packet: bytes):
    
    print(f"main time:{time.time()}")
    simplecoin.submit_func(pid=0,id='test_time',args=())


@app.func('test_time')
def test_time(simplecoin:SimpleCOIN.IPC):
    print(f"func time: {time.time()}")


app.run()