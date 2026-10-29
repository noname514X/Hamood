# 编程作业：CIFAR-10 KNN 分类器

## 作业背景

K 近邻（KNN）是最基础的机器学习分类算法之一，其核心操作是计算测试样本与所有训练样本之间的距离，再通过多数投票确定类别。本作业以 CIFAR-10 图像数据集为实验对象，要求学生从头实现 KNN 分类器，并通过三种不同的实现方式，深刻理解 NumPy 向量化编程对计算效率的影响。

## 任务要求

在 `cifar10_knn.py` 中完成以下 **3 处填空**，实现 L2 距离矩阵的计算：

| 填空 | 函数 | 方法 |
|------|------|------|
| 填空 1 | `compute_distances_two_loops` | 双层循环，逐对计算 L2 距离 |
| 填空 2 | `compute_distances_one_loop` | 单层循环 + NumPy 广播 |
| 填空 3 | `compute_distances_vectorized` | 矩阵展开公式，不使用任何循环 |

每完成一处填空即可单独运行程序验证，三个版本的准确率应完全一致（seed=42, k=5 时参考值约为 **0.245**）。

## 运行方式

```bash
# 使用默认参数（k=5，seed=42）
python cifar10_knn.py

# 自定义参数
python cifar10_knn.py --k 3 --seed 0 --save outputs
```
