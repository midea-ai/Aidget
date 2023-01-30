# pretrain models
python pretrain.py --model resnet20

#distillation
python distill_kd --model_t resnet56 --model_s resnet20