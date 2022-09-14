
<div align="center">   

# 利用transformer 做错号检测
</div>

利用transformer 做错号检测，针对目标区域较小的错号有专门的优化，
- 1,在backbone的第1,2,3阶段提取特诊，(将目标检测头的特征提取前移),并调节anchor 初始框的大小，是的与其对应。
- 2，采用的原始代码为DynamicHead, 但此项目并没有使用其中的模块，主要是因为其速度太慢，并且去掉了检测头的卷积，只留下ATSS最后的分类，框回归，中心点回归的1个卷积层， 位于配置文件 (configs/dyhead_swint_atss_fpn_2x_ms_short.yaml)。
- 3，使用的backbone分别有[Davit](https://arxiv.org/abs/2204.03645), [Facal transformer](https://arxiv.org/pdf/2107.00641.pdf), [Swin transformer](https://arxiv.org/abs/2103.14030)

主要对[DynamicHead](https://github.com/microsoft/DynamicHead)进行优化，前向速度由原来的160ms变为80ms, 并且针对错号检测所关注的准确率不变。


# 使用方法

## 训练

```
    python train_net.py --config configs/dyhead_swint_atss_fpn_2x_ms_short.yaml
```
## 预测
```
    python train_net.py --config configs/dyhead_swint_atss_fpn_2x_ms_short.yaml --eval-only MODEL.WEIGHTS [模型地址]
```
### 若使用新的数据需要配置