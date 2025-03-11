import torch

# 生成随机张量
x = torch.randn(3, 3).to("mlu")  # 迁移到 MLU
y = torch.randn(3, 3).to("mlu")  # 迁移到 MLU

# 在 MLU 上执行矩阵运算
z = x @ y  # 矩阵乘法
print(f"Result on MLU:\n{z.cpu()}")  # 需要迁移回 CPU 进行打印
