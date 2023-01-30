
### 知识蒸馏
知识蒸馏可将强大的教师网络模型的知识转移到紧凑的学生网络模型，从而提升学生模型的精度性能，知识蒸馏可分为logits蒸馏和特征蒸馏。

#### 具体使用步骤
1. 准备教师网络。
    ```python
   modelT = model_arch(teacher_model_selection, num_cls, device)
   modelT.load_state_dict(torch.load(os.path.join('PATH_TO_MODEL')))
    ```
2. 通过配置文件，选择蒸馏算法。
   ```python
   config = {
        'student_model': modelS, #学生网络
        'teacher_model': modelT, #教师网络
        'student_distill_layer': ['linear'], #logits输出层
        'teacher_distill_layer': ['linear'], #logits输出层
        'loss': [dict(type='vanilla_kd', T=4)], #选择loss，以vanillaKD为例。
        'weights': [0.2], #原始模型损失与蒸馏损失权重。
        'student_out_layer': ['linear'], # distiller 返回的输出层。
        'device': device #运行设备
    }
   ```
3. 创建Distiller。
   ```python
   from mslim.algorithms.pytorch.distillation.distiller import Distiller
   distiller_instance = Distiller(config)
    ```
4. 将可训练参数绑定。
   ```python
    optimizer = torch.optim.SGD(distiller_instance.get_learnable_params(), lr=LR, momentum=0.9, weight_decay=5e-4)
    ```
5. 将蒸馏损失与原始模型损失结合。
   ```python
    
    output, kd_loss = distiller.forward(data) #对distiller进行推理，返回学生输出和蒸馏损失。
    loss = criterion(output, target) #原始模型损失 
    losses = kd_loss + loss #蒸馏损失与原始模型结合,可适当对loss乘以系数。
    losses.backward() #整体损失反向传播
    optimizer.step() #优化更新
    ```
- 打印网络结构。(Optional)
    ```python
    distiller_instance.show_student_net()
    distiller_instance.show_teacher_net()
    ```
#### 蒸馏算法配置文件示例
- [Vanila KD](https://arxiv.org/abs/1503.02531)

  ```python
  #配置文件
  config = {
        'student_model': modelS,
        'teacher_model': modelT,
        'student_distill_layer': ['linear'],
        'teacher_distill_layer': ['linear'],
        'loss': [dict(type='vanilla_kd', T=4)],
        'weights': [0.2],
        'student_out_layer': ['linear'],
        'device': device
    }
  ```

- [VID](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf)

  ```python
  #配置文件
  config = {
        'device': device,
        'student_model': modelS,
        'teacher_model': modelT,
        'student_distill_layer': ['layer1', 'layer2', 'layer3'],
        'teacher_distill_layer': ['layer1', 'layer2', 'layer3'],
        'loss': [dict(type='vid', inC=16, midC=16, outC=16, init_pred_var=5.0, eps=1e-5),
                 dict(type='vid', inC=32, midC=32, outC=32, init_pred_var=5.0, eps=1e-5),
                 dict(type='vid', inC=64, midC=64, outC=64, init_pred_var=5.0, eps=1e-5)],
        'weights': [1.0, 1.0, 1.0],
        'student_out_layer': ['linear']
    }
  ```
- [IRG](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Knowledge_Distillation_via_Instance_Relationship_Graph_CVPR_2019_paper.pdf)

  ```python
  #配置文件
  config = {
        'device': device,
        'student_model': modelS,
        'teacher_model': modelT,
        'student_distill_layer': ['layer2', 'layer3', 'linear'],
        'teacher_distill_layer': ['layer2', 'layer3', 'linear'],
        'loss': [dict(type='irg', vert=0.1, edge=5.0, tran=5.0)
                 ],
        'weights': [1.0],
        'student_out_layer': ['linear']
    }
  ```
  
#### YOLOv7 KD with DDP
具体可参考知识蒸馏示例yolov7。
```python
        #添加代码
        from mslim.algorithms.pytorch.distillation.distiller import Distiller
            ...
            # teacher model 
            t_model = torch.load(opt.t_weights, map_location=torch.device('cpu'))
            if t_model.get("model", None) is not None:
            t_model = t_model["model"]
            t_model.to(device)
            t_model.float()
            t_model.eval()
        
            #Build Distiller
            config = {
            'student_model': model,
            'teacher_model': t_model,
            'student_distill_layer': ['model.77'],
            'teacher_distill_layer': ['model.121'],  # 121 7x 105 7
            'loss': [dict(type='YoloKDLoss', nc=model.nc, giou=0.05, dist=1.0, obj=hyp['obj'], cls=hyp['cls'])],
            'weights': [1.0],
            'student_out_layer': ['model.77'],
            'device': device
            }
            distiller_instance = Distiller(config)
            ...
            with amp.autocast(enabled=cuda):
                pred, kd_loss = distiller_instance.forward(imgs) 
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.
                loss = kd_loss + loss
            ...
```
执行训练命令
```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_kd_ddp.py --workers 64 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 512 --data data/data.yaml --img 416 416 --cfg cfg/training/yolov7-tiny.yaml --weights '' --name DDP-train-kd --hyp data/hyp-kd.scratch.tiny.yaml --t_weights teacher.pt  --epochs 300
```
#### Benchmark

|    任务     |     模型      | 类型  | 蒸馏前                 | 蒸馏后                             |    
|:---------:|:-----------:|:---:|:--------------------|:--------------------------------|
|   语义地图    | ResNext-101 | 分割  | bbox:47.9 segm:44.9 | bbox:50.1 segm:64.5 (2.2 1.6up) |      
|   异常检测    |  YOLOv5-S   | 检测  | maP@0.5:0.880       | maP@0.5:0.937 (0.057 up)        |
|   障碍物检测   | YOLOv7-tiny | 检测  | maP@0.5:0.896       | maP@0.5:0.950 (0.054 up)        |
| CIFAR-100 |  ResNet-20  | 分类  | Acc@1:67.35         | Acc@1:69.49 (2.14 up)           |
