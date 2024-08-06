import subprocess
import re
from pathlib import Path
def run_val_script(weights, data):
    # 构造命令
    command = [
        'python3', 'val.py',
        '--weights', weights,
        '--data', data,
        '--save-txt',
    ]


    result = subprocess.run(command, capture_output=True, text=True)


    # Regular expression to find the path
    match = re.search(r"(\d+) labels saved to (\S+)", result.stderr)
    saved_path = match.group(2)        # 'runs/val/exp8/labels'
    txt_outputs = []
    labels_dir = Path(saved_path)
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as file:
            txt_outputs.append(file.read())
    print(txt_outputs)

    return result.stdout, result.stderr


if __name__ == "__main__":
    # 定义参数
    weights_path = 'models_trained/best.pt'
    data_path = 'dataset/fruits/data.yaml'


    # 调用函数并获取输出
    #stdout, stderr = run_val_script(weights_path, data_path)

    saved_path = './runs/val/exp/labels'
    txt_outputs = []
    labels_dir = Path(saved_path)
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'rb') as file:
            txt_outputs.append(file.read())

    print(txt_outputs)

