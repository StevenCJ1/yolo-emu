from utils.packetutils import chunk_handler, HEADER_DATA
from utils.yohobuffer import YOHOBuffer
import torch

torch_array = torch.tensor([[[1.00, 1.01, 1.02, 1.03, 1.04]]])
print("torch_array:", torch_array)
bytes_arr = chunk_handler.get_serialize_torcharray(HEADER_DATA, 0 , torch_array)

buffer = YOHOBuffer()
for bytes in bytes_arr:
    buffer.put(bytes[2:])

print("len bytes_arr", len(bytes_arr))
print("buffer:", buffer.extract())
torch_array_from_bytes = chunk_handler.derialize_with_index(buffer.extract())
print("torch_array from bytes:", torch_array_from_bytes)