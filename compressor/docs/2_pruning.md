### 剪枝
    模型剪枝可将模型中冗余的权重找到并删除。本框架针对提速目的，主要支持结构化/通道剪枝。

#### [ResRep](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.pdf)

##### 算法流程:

1. 读取原始模型

     获取原始模型。

2. ResRep剪枝训练

      向BN层插入经过特殊初始化的Compactor($1\times1$卷积层)，并对所有compactor的梯度施加惩罚项，在初期大约5个epoch的训练之后，筛选模型中Lasso范数最小的通道，直至余下的运算量达到设定的目标值。

3. ResRep剪枝

      移除范数小于给定阈值的通道，自动完成 Conv-BN-Compactor三者的合并，无需进一步精调。

**YOLOv7示例详见： examples/pruning/YOLOv7/Resrep on YOLOv7.md**

##### 以分类网络ResNet56为例:

1. 读取原始模型。
    ```python
    #原始模型获取 best_pretrained.pt
    model = torch.load('best_pretrained.pt')
    #注：使用 baseline 权重继续进行剪枝训练一般能获得更好的效果
    ```
2. Resrep 训练
    ```bash
     #调用 ResRep 类插入 Compactor 并开启 ResRep 训练，可通过 --flops_target 指定目标运算量，--lasso_strength 指定惩罚力度
     python Resrep_train.py --flops_target 0.471 --lasso_strength 1e-4
    ```
    ```python
    #注：实验表明，对残差结构添加 Compactor，效果反而会有一定下降，因此目前的代码仅在非残差结构的 Conv-BN 组合后插入
    from mslim.algorithms.pytorch.prune.Resrep_pruner import ResRep, calculate_model_flops, Resrep_1, Resrep_2
    
    iteration = 0
    ResRep_object = ResRep(copy.deepcopy(model))
    model = ResRep_object.wrapped_model
    dummy_input = torch.randn([1, 3, opt.img_size[0], opt.img_size[0]]).to(device, non_blocking=True)
    ResRep_object.original_flops = calculate_model_flops(model, dummy_input, ResRep_object)
    
    # 需要注释先前的SGD [Optional] 如果原先使用的非SGD，可以参考该API的实现
    optimizer = ResRep_object.sgd_optimizer(hyp['momentum'])
        
    epochs = 480
    iteration = 0
    before_mask_epoch = 5
    num_images = train_loader.sampler.__len__()  # 获取训练集图片个数
    
    for epoch in range(epochs):
        for i, (input, target) in enumerate(train_loader):
            
            model, iteration, compactor_mask_dict = Resrep_1(opt, iteration, model, ResRep_object, epoch, num_images)
            pred = model(input)
            ...
            loss.backward()
            Resrep_2(opt, compactor_mask_dict)
            optimizer.zero_grad()
            # end batch
        # end epoch
    ```
3. ResRep剪枝
    ```python
    #传入第三步训练得到的 best.pt，并指定 threshold 进行通道剪枝，论文使用1e-5
    python Resrep_prune.py --wrapped_model best.pt --threshold 1e-5
    ```
##### Benchmark

| model   (dataset)    | （剪枝前->剪枝后）                       | 
|----------------------|----------------------------------|
| ResNet56  (CIFAR-10) | 94.09% -> 93.89% (flops ↓ 52.9%) |
