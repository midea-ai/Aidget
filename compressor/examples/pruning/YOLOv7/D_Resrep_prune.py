import sys
import argparse

from thop import profile, clever_format
from mslim.algorithms.pytorch.prune.Resrep_pruner import *
from models.experimental import attempt_load
from utils.torch_utils import revert_sync_batchnorm


def ResRep_prune(original_model, wrapped_model, img_size, threshold=None, factor=None):
    if not threshold:
        threshold = 1e-5
    if not factor:
        factor = 8
    device = torch.device('cuda', 0)
    input = torch.ones([1, 3, img_size, img_size]).cuda()

    model = attempt_load(wrapped_model, map_location=device)
    wrapped_model = revert_sync_batchnorm(model)

    model = torch.load(original_model)['model'].float().cuda()
    flops_b, params_b = profile(model.to(device), inputs=(input,))
    flops_b, params_b = clever_format([flops_b, params_b], '%.3f')
    ResRep_object = ResRep(copy.deepcopy(model))

    compactor_matrix = dict()
    metric_vector = dict()
    filter_ids_below_thresh = dict()
    for name, layer in wrapped_model.named_modules():
        if type(layer).__name__ == 'InsertCompactor':
            compactor_matrix[name] = layer.compactor.pwc.weight.detach().cpu().numpy()
            metric_vector[name] = np.sqrt(np.sum(compactor_matrix[name] ** 2, axis=(1, 2, 3)))

    for name in metric_vector.keys():
        filter_ids_below_thresh[name] = np.where(metric_vector[name] < threshold)[0]
        if len(filter_ids_below_thresh[name]) > 0:
            filter_ids_below_thresh[name] = np.argsort(metric_vector[name][np.where(metric_vector[name] < threshold)[0]])[:(len(filter_ids_below_thresh[name]) - len(filter_ids_below_thresh[name]) % factor)]

    for name in metric_vector.keys():
        if len(filter_ids_below_thresh[name]) > 0:
            compactor_matrix[name] = np.delete(compactor_matrix[name], filter_ids_below_thresh[name], axis=0)

    print("Start pruning...")
    pruned_tail = set()

    for key in ResRep_object.conv_list:
        key = key.replace('module.', '')
        if key not in ResRep_object.connections:
            continue
        if key in ResRep_object.residual_conv or key in ResRep_object.exclude:
            continue
        
        else:
            conv_name = key
            bn_name = ResRep_object.conv_bn_dict['module.' + conv_name].replace('module.', '') + '.module'
            compactor_name = ResRep_object.conv_bn_dict['module.' + conv_name].replace('module.', '') + '.compactor'
            bn_compactor_name = ResRep_object.conv_bn_dict['module.' + conv_name].replace('module.', '')
            conv = visit_module(wrapped_model, conv_name)
            bn = visit_module(wrapped_model, bn_name)
            fused_conv = fuse_conv_bn(conv, bn)

            kernel = F.conv2d(fused_conv.weight.permute(1, 0, 2, 3).cuda(),
                              torch.from_numpy(compactor_matrix[bn_compactor_name]).cuda()).permute(1, 0, 2, 3)
            bias = torch.matmul(torch.from_numpy(compactor_matrix[bn_compactor_name]).reshape(
                compactor_matrix[bn_compactor_name].shape[0],
                compactor_matrix[bn_compactor_name].shape[1]).cuda(), fused_conv.bias.cuda())
            pruned_conv = nn.Conv2d(in_channels=kernel.shape[1], out_channels=kernel.shape[0],
                                    stride=fused_conv.stride, groups=fused_conv.groups,
                                    kernel_size=fused_conv.kernel_size, padding=fused_conv.padding, bias=True)

            pruned_conv.weight.data = kernel
            pruned_conv.bias.data = bias
            ori_module = [conv_name, bn_name, compactor_name]
            new_module = [pruned_conv, nn.Identity(), nn.Identity()]

            for (ori_layer, new_layer) in zip(ori_module, new_module):
                set_module(wrapped_model, ori_layer, new_layer)

            next_conv_list = ResRep_object.connections[key]

            for k in range(len(next_conv_list)):
                next_conv_name = next_conv_list[k]
                next_conv = visit_module(wrapped_model, next_conv_name)
                next_bn_name = ResRep_object.conv_bn_dict['module.' + next_conv_name].replace('module.', '')
                next_bn = visit_module(wrapped_model, next_bn_name)
                if hasattr(next_bn, 'module'):
                    next_bn_name = next_bn_name + '.module'
                    next_bn = visit_module(wrapped_model, next_bn_name)
                fused_next_conv = fuse_conv_bn(next_conv, next_bn)

                if next_conv_name in ResRep_object.concat_relations:
                    if next_conv_name in pruned_tail:
                        continue
                    pruned_tail.add(next_conv_name)
                    in_mask = torch.tensor([])
                    for head in ResRep_object.concat_relations[next_conv_name]:
                        head_out_mask = torch.ones(visit_module(model, head).out_channels)
                        for num in filter_ids_below_thresh[head.replace('conv', 'bn')]:
                            head_out_mask[num] = 0
                        in_mask = torch.cat((head_out_mask, in_mask), 0)
                    pruned_index = np.array(torch.nonzero(1 - in_mask, as_tuple=True)[0])
                    new_next_conv = nn.Conv2d(in_channels=int(torch.sum(in_mask)),
                                              out_channels=fused_next_conv.out_channels,
                                              kernel_size=fused_next_conv.kernel_size,
                                              stride=fused_next_conv.stride,
                                              groups=fused_next_conv.groups,
                                              padding=fused_next_conv.padding, bias=True)
                    new_next_conv.weight.data = nn.Parameter(
                        torch.from_numpy(np.delete(fused_next_conv.weight.detach().cpu().numpy(),
                                                   pruned_index, axis=1)))
                    new_next_conv.bias = nn.Parameter(torch.from_numpy(fused_next_conv.bias.detach().cpu().numpy()))
                else:
                    new_next_conv = nn.Conv2d(in_channels=pruned_conv.out_channels,
                                              out_channels=next_conv.out_channels,
                                              kernel_size=next_conv.kernel_size, 
                                              stride=next_conv.stride,
                                              groups=next_conv.groups,
                                              padding=next_conv.padding, bias=True)

                    if len(filter_ids_below_thresh[bn_compactor_name]) != 0:
                        new_next_conv.weight.data = nn.Parameter(
                            torch.from_numpy(np.delete(fused_next_conv.weight.detach().cpu().numpy(),
                                                       filter_ids_below_thresh[bn_compactor_name], axis=1)))
                        new_next_conv.bias = nn.Parameter(
                            torch.from_numpy(fused_next_conv.bias.detach().cpu().numpy()))
                    else:
                        new_next_conv = fused_next_conv
                set_module(wrapped_model, next_conv_name, new_next_conv)
                set_module(wrapped_model, next_bn_name, nn.Identity())

    ckpt = {
        'model': wrapped_model
    }

    flops_a, params_a = profile(wrapped_model.to(device), inputs=(input,))
    flops_a, params_a = clever_format([flops_a, params_a], '%.3f')
    print(f'==> FLOPs Pruned Percent:',
          f"{100 * (float(flops_b[:-1]) - float(flops_a[:-1])) / float(flops_b[:-1]):.2f}%",
          f'{flops_b} ---> {flops_a}')
    print(f'==> Params Pruned Percent:',
          f"{100 * (float(params_b[:-1]) - float(params_a[:-1])) / float(params_b[:-1]):.2f}%",
          f'{params_b} ---> {params_a}')

    torch.save(ckpt, 'pruned_model.pt')
    torch.onnx.export(wrapped_model.cuda(), input, 'solve.onnx', opset_version=12)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrapped_model', default="", help='Resrep_trained model with 1x1')
    parser.add_argument('--original_model', default='', help='pretrained model without 1x1')
    parser.add_argument('--img_size', default=416)
    parser.add_argument('--factor', default=8)
    parser.add_argument('--threshold', default=1e-5)
    opt = parser.parse_known_args()[0]
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(sys.path)
    ResRep_prune(opt.original_model, opt.wrapped_model, opt.img_size, opt.threshold, opt.factor)
