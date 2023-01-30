**Step 1 预先准备：**

配置coco数据集，于 https://github.com/WongKinYiu/yolov7/releases/tag/v0.1 处下载预训练权重

**Step 2 剪枝训练：**

```python
# DDP（recommended）
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 D_Resrep_train.py   
--workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml  # 注意模型对齐，不要直接拷贝

!!!在此基础上添加 
--flops_target   剪枝后的模型保留的flops百分比，设置为0.8即意味着剪去20%的flops
--lasso_strength 训练的惩罚力度，数据集越复杂，设置的就越大，通道得分下降的越快（coco设置为0.2，CIFAR-10设置为1e-4）

# Single-GPU
python D_Resrep_train.py  # 额外参数配置参考DDP
```

**Notes：**

1.可以根据MAP的变化与新生成的 **log.txt** 中各通道的得分配合进行调参：

​    20轮后开始在 **log.txt** 中打印通道得分（从小到大排序后最小的5个以及阈值处左右相邻各5个，一共15个），可以每过10轮关注下下降趋势和MAP保持情况

​    **使用DDP训练时，通道得分不能反映真实的得分情况，但是可以反映下降趋势以及彼此间的相对差异！！！**

​    真实的得分情况获取方法详见第三步

2.根据经验，MAP能够在coco数据集上稳定300-400轮左右后出现明显下滑，此时**不用停止训练**，可以先将保存下来的权重进行剪枝

**Step 3 训练后剪枝：**

```python
python D_Resrep_prune.py

!!!在此基础上添加
--original_model 第一步下载的预训练权重
--wrapped_model  剪枝训练后生成的权重
--img_size 输入分辨率，单个数值，416/512/640...
--threshold 默认1e-5，剪枝训练越充分，高MAP保持越久，效果越好；训练不充分时适当提高该值
--factor 默认为8，生成的模型输出通道保持8的倍数，推理加速效果略优于非8倍通道
```

**Notes：**

1.第二步DDP训练途中想要获取真实的得分情况，可以打印 **D_Resrep_prune.py** 文件中的 **metric_vector** 字典变量，最终的训练效果如下图所示：

![Screenshot](Resrep on YOLOv7.assets/Screenshot.png)

2.如果已经出现了较多<1e-5的通道，剪枝后模型的MAP下降较多，说明数据集较为复杂（例如coco），推荐对剪枝后的小模型使用蒸馏手段以恢复精度

   **如果并未出现得分<1e-5的通道，模型精度已经下降明显，请在第二步中调低惩罚系数 或 继续训练直到得分出现1e-5并进行蒸馏**

