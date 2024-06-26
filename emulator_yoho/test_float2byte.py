import pickle
import struct

list_float = [1.00]

def floatToBytes(f):
    bs = struct.pack("f",f)
    print(f"type bs = {type(bs)}")
    return (bs[3],bs[2],bs[1],bs[0])

bs = [None] * 4
bs[3], bs[2], bs[1], bs[0] = floatToBytes(list_float[0])

def bytesToFloat(h1,h2,h3,h4):
    ba = bytearray()
    ba.append(h1)
    ba.append(h2)
    ba.append(h3)
    ba.append(h4)
    return struct.unpack("!f",ba)[0]

print(bytesToFloat(bs[3],bs[2],bs[1],bs[0]))

list_byte = pickle.dumps(list_float)
print(len(list_byte), list_byte)

print("------------------------------")
a = float(1.00)