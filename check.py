import torch
print(torch.cuda.device_count())  # 查询 GPU 数量
print(torch.cuda.get_device_name(0))  # 获取 GPU 名称（如果存在）
