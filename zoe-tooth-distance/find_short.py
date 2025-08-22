import os
from collections import defaultdict

# 替换为你的目录路径
folder_path = 'F:\data_set_tt'

# 获取文件夹中所有文件
all_files = os.listdir(folder_path)

# 用于记录每个前缀（如84-LWT）对应的文件种类
prefix_map = defaultdict(set)

# 遍历所有文件
for filename in all_files:
    if filename.endswith('.json') or filename.endswith('.JPG'):
        parts = filename.split('-')
        if len(parts) < 3:
            continue  # 文件名格式不合法
        prefix = f"{parts[0]}-{parts[1]}"  # 如：84-LWT
        suffix = '-'.join(parts[2:])       # 如：L.json、U.JPG
        prefix_map[prefix].add(suffix)

# 检查缺失项
expected_suffixes = {'L.json', 'L.JPG', 'U.json', 'U.JPG'}
print("以下序号缺少文件：")
for prefix, suffixes in sorted(prefix_map.items()):
    missing = expected_suffixes - suffixes
    if missing:
        print(f"{prefix} 缺少: {', '.join(missing)}")
