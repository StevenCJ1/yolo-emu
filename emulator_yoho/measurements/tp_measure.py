import pyshark
import pandas as pd
tshark_path = r'E:\SoftWare\Wireshark\tshark.exe'
import os



def get_all_filenames(folder_path):
    files_and_folders = os.listdir(folder_path)
    files = [f for f in files_and_folders if os.path.isfile(os.path.join(folder_path, f))]
    
    return files


def find_target_packet_times(capture):
    length_502_found = False
    target_packet_times = []
    packet_size_list = []
    packet_size = 0
    for packet in capture:
        try:
            packet_length = int(packet.length)

            if packet_length == 502:
                length_502_found = True
                continue

            if length_502_found:
                target_packet_times.append(packet.sniff_time)
                length_502_found = False

            if packet_length > 502:
                packet_size = packet_size + packet_length

            if packet_length == 73:
                packet_size_list.append(packet_size)
                packet_size = 0

        except AttributeError:
            continue

    return target_packet_times,packet_size_list


if '__main__' == __name__:

    test_id = 3
    folder_path = f'testcaseTP_{test_id}/'  # 替换为你的文件夹路径
    filenames = get_all_filenames(folder_path)
    #print(filenames)
    files = {
        "client_s0": f"testcaseTP_{test_id}/tp_client-s0.csv",

        "switch0-client": f'testcaseTP_{test_id}/s0-client.pcap',
        "vnf1_s0": f"testcaseTP_{test_id}/tp_vnf0-s0.csv",
        "switch0-s1": f'testcaseTP_{test_id}/s0-s1.pcap',

        "switch1-s0": f'testcaseTP_{test_id}/s1-s0.pcap',
        "vnf0_s1": f"testcaseTP_{test_id}/tp_vnf1-s1.csv",
        "switch1-s2": f'testcaseTP_{test_id}/s1-s2.pcap',

        "switch2-s1":f'testcaseTP_{test_id}/s2-s1.pcap',
        "vnf2_s2": f"testcaseTP_{test_id}/tp_vnf2-s2.csv",
        "switch2-server":f'testcaseTP_{test_id}/s2-server.pcap',
        "server_s2": f"testcaseTP_{test_id}/tp_server-s2.csv"
    }


    # 创建新的字典
    new_files = {key: value for key, value in files.items() if any(value.endswith(file) for file in filenames)}

    # 初始化一个空的DataFrame
    time_df_list = []
    data_df_list = []
    timepoint = 1
    # 读取CSV文件并存储值
    for key, file_path in new_files.items():
        # switch time point read
        if 'switch' in key:
            cap = pyshark.FileCapture(file_path, tshark_path=tshark_path)
            target_packet_times, packet_size_list = find_target_packet_times(cap)
            cap.close()
            # 将时间戳列表转换为DataFrame
            time_temp_df = pd.DataFrame(target_packet_times, columns=[f"t{timepoint}"])
            data_temp_df = pd.DataFrame(packet_size_list, columns=[f"{key}"])
            data_df_list.append(data_temp_df)
        else:

            time_temp_df = pd.read_csv(file_path)
            # server time point read
            if key in ["server_s2"]:
                time_temp_df = time_temp_df.iloc[:, [0]]
                time_temp_df['t0'] = pd.to_datetime(time_temp_df['t0'], unit='s',utc=True)
                time_temp_df['t0'] = time_temp_df['t0'].dt.tz_convert('Europe/Berlin')
                time_temp_df.columns = [f"t{timepoint}"]

            # client and vnf time point read
            else:
                time_temp_df = time_temp_df.iloc[:, :]
                time_temp_df['t0'] = pd.to_datetime(time_temp_df['t0'], unit='s',utc=True)
                time_temp_df['t1'] = pd.to_datetime(time_temp_df['t1'], unit='s',utc=True)
                time_temp_df['t0'] = time_temp_df['t0'].dt.tz_convert('Europe/Berlin')
                time_temp_df['t1'] = time_temp_df['t1'].dt.tz_convert('Europe/Berlin')
                if key in ["client_s0"]:
                    time_temp_df.columns = [f"t{timepoint}", "t15"]
                else:
                    time_temp_df.columns = [f"t{timepoint}",f"t{timepoint + 1}"]
                    timepoint = timepoint + 1

        timepoint = timepoint + 1

        time_df_list.append(time_temp_df)


    combined_df = pd.concat(time_df_list, axis=1)
    data_df = pd.concat(data_df_list, axis = 1)
    # 将第二列移到最后一列
    cols = list(combined_df.columns)
    cols.append(cols.pop(1))  # 将第二列移动到最后
    combined_df = combined_df[cols]

    for col in combined_df.columns:
        combined_df[col] = pd.to_datetime(combined_df[col], unit='s').dt.tz_localize(None)

    time_diff_data = combined_df.diff(axis=1).drop(columns=['t1'])
    time_diff_data = time_diff_data.apply(lambda x: x.dt.total_seconds())


    # 保存到CSV文件中
    time_file_path = f'testcaseTP_{test_id}/time_diff.csv'
    data_file_path = f'testcaseTP_{test_id}/data_size.csv'
    time_diff_data.to_csv(time_file_path, index=False)
    data_df.to_csv(data_file_path, index=False)


