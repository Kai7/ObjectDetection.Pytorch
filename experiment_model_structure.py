from functools import partial

import argparse
import yaml
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from architectures import retinanet, efficientdet
from ksevendet.architecture import ksevendet
import ksevendet.architecture.backbone.registry as registry
#from datasettool.dataloader import KSevenDataset, CocoDataset, collater, \
#                                   Resizer, AspectRatioBasedSampler, Augmenter, \
#                                   Normalizer
from torch.utils.data import DataLoader
#from datasettool import coco_eval
#from datasettool.flir_eval import evaluate_flir
#from datasettool.coco_eval import evaluate_coco
#from datasettool.cvidata_eval import coco_evaluate_cvidata
#from datasettool.ksevendata_eval import coco_evaluate_ksevendata
#from datasettool import csv_eval

# from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weightso

from PIL import Image

import os
import sys
import logging
# from thop import profile

from distiller import model_summaries
from distiller import create_png
from distiller import model_find_module_name
from distiller import SummaryGraph

import copy

import pdb

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def get_args():
    parser = argparse.ArgumentParser(description='.')
    
    parser.add_argument('--model_config', default=None, type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser.add_argument('--architecture', default='ksevendet', type=str,
                        help='Network Architecture.')
    parser.add_argument('--num_classes', type=int,
                        help='The number of class.')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--input_shape', default='512,512', type=str,
                        help='Input images Shape (height, width)')
    parser.add_argument("--log", default=False, action="store_true" , 
                        help="Write log file.")

    args = parser.parse_args()

    assert args.model_config, 'Model config must be provide.'
    with open(args.model_config, 'r') as f:
        model_cfg = yaml.safe_load(f)
    setattr(args, 'architecture', model_cfg.pop('architecture'))
    setattr(args, 'model_cfg', model_cfg)
        
    return args

def get_logger(name='My Logger', args=None):
    LOGGING_FORMAT = '%(levelname)s:    %(message)s'

    my_logger     = logging.getLogger(name)
    formatter     = logging.Formatter(LOGGING_FORMAT)
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    my_logger.addHandler(streamhandler)
    my_logger.setLevel(logging.INFO)
    if args is not None and args.log:
        filehandler = logging.FileHandler(os.path.join('log', '{}_{}.log'.format(args.dataset, args.network_name)), mode='a')
        filehandler.setFormatter(formatter)
        my_logger.addHandler(filehandler)

    return my_logger

def main():
    args = get_args()
    support_architectures = [
        'ksevendet',
    ]
    support_architectures += [f'efficientdet-d{i}' for i in range(8)]
    support_architectures += [f'retinanet-res{i}' for i in [18, 34, 50, 101, 152]]
    support_architectures.append('retinanet-p45p6')

    print(support_architectures)

    if args.architecture == 'ksevendet':
        ksevendet_cfg = args.model_cfg
        if ksevendet_cfg.get('variant'):
            network_name = f'{args.architecture}-{ksevendet_cfg["variant"]}-{ksevendet_cfg["neck"]}'
        else:
            assert isinstance(ksevendet_cfg, dict)
            network_name = f'{args.architecture}-{ksevendet_cfg["backbone"]}_specifical-{ksevendet_cfg["neck"]}'
    elif args.architecture in support_architectures:
        network_name = args.architecture
    else:
        raise ValueError('Architecture {} is not support.'.format(args.architecture))

    args.network_name = network_name

    net_logger = get_logger(name='Network Logger', args=args)
    net_logger.info('Network Name: {}'.format(network_name))

    height, width = tuple(map(int, args.input_shape.split(',')))
    
    net_logger.info('Number of Classes: {:>3}'.format(args.num_classes))
    
    build_param = {'logger': net_logger}
    if args.architecture == 'ksevendet':
        net_model = ksevendet.KSevenDet(ksevendet_cfg, num_classes=args.num_classes, pretrained=False, **build_param)
    else:
        assert 0, 'architecture error'

    # load last weights
    if args.resume is not None:
        net_logger.info('Loading Weights from Checkpoint : {}'.format(args.resume))
        try:
            ret = net_model.load_state_dict(torch.load(args.resume), strict=False)
        except RuntimeError as e:
            net_logger.warning(f'Ignoring {e}')
            net_logger.warning(f'Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        s_b = args.resume.rindex('_')
        s_e = args.resume.rindex('.')
        start_epoch = int(args.resume[s_b+1:s_e]) + 1
        net_logger.info('Continue on {} Epoch'.format(start_epoch))
    else:
        start_epoch = 1
        
    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            net_model = net_model.cuda()

    #if torch.cuda.is_available():
    #    net_model = torch.nn.DataParallel(net_model).cuda()
    #else:
    #    net_model = torch.nn.DataParallel(net_model)

    sample_image = np.zeros((height, width, 3)).astype(np.float32)
    sample_image = torch.from_numpy(sample_image)
    sample_input = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
    sample_input_shape = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0).shape

    #sample_input = torch.randn(1, 3, height, width).cuda().float()

    #print(sample_input_shape)

    #net_model.eval()
    model_summaries.model_summary(net_model, 'compute', input_shape=sample_input_shape)

    # print(net_model)

    #for parameter in net_model.parameters():
    #    print(type(parameter))

    print('='*120)

    #for n, m in net_model.named_modules():
    #    if isinstance(m, torch.nn.Conv2d):
    #        print(n)

    #for n, p in net_model.named_parameters():
    #    print(n)

    #m_dict = net_model.state_dict()
    #for n in m_dict:
    #    print(n)
    #    # print(type(m_dict[n]))
    
    #def print_child_module_name(parent_m, prefix=''):
    #    for name, sub_module in parent_m.named_children():
    #        name_ = '.'.join([prefix, name])
    #        print(name_)
    #        print(type(sub_module))
    #        print_child_module_name(sub_module, prefix=name_)
    #print_child_module_name(net_model)

    #def print_name(m):
    #    print(m)
    #net_model.apply(print_name)

    #print(net_model.src_device_obj)

    def conv_hook(module, fea_in, fea_out, model):
        module_name.append(module.__class__.__name__)
        features_in_hook.append(fea_in[0].shape)
        features_out_hook.append(fea_out[0].shape)

        conv_weights.append(module.weight)
        mod_name = model_find_module_name(model, module)
        # print('find Conv module: {}'.format(mod_name))
        conv_modules_map[mod_name] = module

        return None

    def bn_hook(module, fea_in, fea_out, model):
        module_name.append(module.__class__.__name__)
        features_in_hook.append(fea_in[0].shape)
        features_out_hook.append(fea_out[0].shape)

        mod_name = model_find_module_name(model, module)
        # print('find BN module: {}'.format(mod_name))
        bn_modules_map[mod_name] = module

        return None

    def module_recorder(m):
        if isinstance(m, torch.nn.Conv2d):
            hook_handles.append(m.register_forward_hook(partial(conv_hook, model=net_model)))
        if isinstance(m, torch.nn.BatchNorm2d):
            hook_handles.append(m.register_forward_hook(partial(bn_hook, model=net_model)))


    module_name = []
    features_in_hook = []
    features_out_hook = []
    conv_weights = []
    conv_modules_map = collections.OrderedDict()
    bn_modules_map = collections.OrderedDict()
    #net_chilren = net_model.children()
    #for child in net_chilren:
    #    child.register_forward_hook(hook=hook)
    hook_handles = []
    with torch.no_grad():
        #for n, p in net_model.named_parameters():
        #    print(p.requires_grad)
        net_model.apply(module_recorder)
        net_model(sample_input, return_head=True)
        #print(len(features_in_hook))
        #for i in range(len(module_name)):
        #    print('{} | {} | {}'.format(module_name[i], features_in_hook[i], features_out_hook[i]))

        # Unregister from the forward hooks
        for handle in hook_handles:
            handle.remove()
    
        #print(type(conv_weights[0]))
        # net_model(sample_input, return_head=True)
        # print(len(features_in_hook))
    
    #print(type(conv_modules_map.keys()))
    #conv_modules = list(conv_modules_map.keys())
    #for i in range(10):
    #    print(f'[{i}] {conv_modules[i]}')
    #print(type(bn_modules_map.keys()))
    #bn_modules = list(bn_modules_map.keys())
    #for i in range(10):
    #    print(f'[{i}] {bn_modules[i]}')
    
    net_model = net_model.cuda()

    # print(torch.equal(conv_modules_map[conv_modules[0]].weight, conv_weights[0]))
    # print(torch.equal(conv_modules_map[conv_modules[1]].weight, conv_weights[1]))

    def printer_hook(module, fea_in, fea_out, model):
        mod_name = model_find_module_name(model, module)
        print(mod_name)

    def module_handles_recorder(m):
        if isinstance(m, torch.nn.Conv2d):
            hook_handles.append(m.register_forward_hook(partial(printer_hook, model=net_model)))
        if isinstance(m, torch.nn.BatchNorm2d):
            hook_handles.append(m.register_forward_hook(partial(printer_hook, model=net_model)))

    #hook_handles = []
    #net_model.apply(module_handles_recorder)

    net_model.set_onnx_convert_info(fixed_size=(height, width))    

    print('new model inference')
    # net_model(sample_input, return_head=True)
    # model_summaries.model_summary(net_model, 'compute', input_shape=sample_input_shape)

    print('done')
    
    # Before summaryGrapy, you must to remove hook functions for model.
    g = SummaryGraph(net_model, sample_input)
    print(type(g))

    SAVE_GRAPH_IMAGE = False 
    if SAVE_GRAPH_IMAGE:
        import io
        png = create_png(g)
        stream = io.BytesIO(png)
        g_image = Image.open(stream).convert('RGB')
        IMAGE_FILE_PATH = 'g_image.jpg'
        print('save graph image to {}'.format(IMAGE_FILE_PATH))
        g_image.save(IMAGE_FILE_PATH, 'jpeg')
        print('done.')
        exit(0)
    
    # pdb.set_trace()


    ops_keys = list(g.ops.keys())
    #for k in ops_keys[:10]:
    #    print('({})  {}'.format(g.ops[k]['type'], g.ops[k]['name']))
    #    print('  In  : {}'.format(str(g.ops[k]['inputs'])))
    #    print('  Out : {}'.format(str(g.ops[k]['outputs'])))

    ops_type_map = collections.defaultdict(list)
    for k in ops_keys:
        ops_type_map[g.ops[k]['type']].append(g.ops[k])
    
    print(ops_type_map.keys())

    ##for _op in ops_type_map['Conv'][:10]:
    #for _op in ops_type_map['Conv']:
    #    print('({})  {}'.format(_op['type'], _op['name']))
    #    print('  In  ({}): {}'.format(len(_op['inputs']), str(_op['inputs'])))
    #    print('  Out ({}): {}'.format(len(_op['outputs']), str(_op['outputs'])))


    ops_types = ['Conv', 'BatchNormalization', 
                 'Relu', 'MaxPool', 'Add', 'Constant', 'Upsample', 'Transpose', 'Reshape', 'Concat', 'Sigmoid']
    ignore_op_type = ['Constant',]

    INPUT_TENSOR_ID = 'input.1'

    weight_nodes_ids = list()
    tensor_nodes = dict()
    for t_id, t_param in g.params.items():
        if t_id == INPUT_TENSOR_ID or data_node_has_parent(g, t_id):
            _attr = {'shape': t_param['shape']}
            _node = SNode(t_id, 'tensor', attr=_attr)
            tensor_nodes[t_id] = _node 
        else:
            weight_nodes_ids.append(t_id)

    op_nodes = dict() 
    for ops_name in ops_keys:
        # print(ops_name)
        op_info = g.ops[ops_name]
        assert op_info['name'] == ops_name
        if op_info['type'] in ignore_op_type:
            continue
        _attr = {
            'type': op_info['type']
        }
        for _a in op_info['attrs']:
            _attr[_a] = op_info['attrs'][_a]
        _node = SNode(ops_name, 'op', attr=_attr)
        for t_id in op_info['inputs']:
            if t_id in weight_nodes_ids:
                continue
            in_node = tensor_nodes.get(t_id, None)
            assert in_node is not None, 'tensor node not found.'
            _node.in_tensor.add(in_node)
            in_node.out_op.add(_node)
        assert len(op_info['outputs']) == 1, 'op output num error'
        for t_id in op_info['outputs']:
            out_node = tensor_nodes.get(t_id, None)
            assert in_node is not None, 'tensor node not found.'
            _node.out_tensor.add(out_node)
            out_node.in_op.add(_node)
        op_nodes[ops_name] = _node

    for _op_name in op_nodes:
        _node = op_nodes[_op_name]
        _node.in_tensor  = list(_node.in_tensor)
        _node.out_tensor = list(_node.out_tensor)
    for _tensor_name in tensor_nodes:
        _tensor = tensor_nodes[_tensor_name]
        _tensor.in_op  = list(_tensor.in_op)
        _tensor.out_op = list(_tensor.out_op)
    
    eq_tensors_map, eq_tensors_map_inverse = generate_equivalence_tensor_map(ops_type_map, op_nodes, tensor_nodes)
    eq_t_ids = list(eq_tensors_map_inverse.keys())
    eq_t_ids.sort()
    #for eq_t_id in eq_t_ids:
    #    print()
    #    t_ids = list(eq_tensors_map_inverse[eq_t_id])
    #    t_ids.sort()
    #    t_ids = ' , '.join(t_ids)
    #    print('{} >> [ {} ]'.format(eq_t_id, t_ids))


    print('============================================================')
    pruning_channel_to_filter_ops_chain = dict()
    for _conv_op in ops_type_map['Conv']:
        _op_name = _conv_op['name']
        conv_node = op_nodes[_op_name]
        # print(conv_node.name)
        assert len(conv_node.in_tensor) == 1, 'len error'
        bottom_tensor_node = conv_node.in_tensor[0]
        eq_tensor_id = eq_tensors_map.get(bottom_tensor_node.name, None)
        search_t_ids = [bottom_tensor_node.name,] if not eq_tensor_id else eq_tensors_map_inverse[eq_tensor_id]
        #if not eq_tensor_id:
        to_prun_filter_convs = set()
        for t_id in search_t_ids:
            search_tensor_node = tensor_nodes[t_id]
            if len(search_tensor_node.in_op) == 0:
                print('input tensor', search_tensor_node.name)
                continue
            assert len(search_tensor_node.in_op) == 1, 'len error'
            bottom_op_node = search_tensor_node.in_op[0]
            if bottom_op_node.attr['type'] == 'Conv':
                to_prun_filter_convs.add(bottom_op_node.name)
        to_prun_filter_convs = list(to_prun_filter_convs)
        to_prun_filter_convs.sort()
        pruning_channel_to_filter_ops_chain[_op_name] = to_prun_filter_convs
                
    pruning_filter_to_channel_ops_chain = dict()
    for _conv_op in ops_type_map['Conv']:
        _op_name = _conv_op['name']
        conv_node = op_nodes[_op_name]
        # print(conv_node.name)
        assert len(conv_node.out_tensor) == 1, 'len error'
        top_tensor_node = conv_node.out_tensor[0]
        eq_tensor_id = eq_tensors_map.get(top_tensor_node.name, None)
        search_t_ids = [top_tensor_node.name,] if not eq_tensor_id else eq_tensors_map_inverse[eq_tensor_id]
        to_prun_channel_convs = set()
        for t_id in search_t_ids:
            search_tensor_node = tensor_nodes[t_id]
            if len(search_tensor_node.out_op) == 0:
                print('output tensor', search_tensor_node.name)
                continue
            # print('{} has {} out ops.'.format(t_id, len(search_tensor_node.out_op)))
            for top_op_node in search_tensor_node.out_op:
                if top_op_node.attr['type'] in ['Conv',]:
                    to_prun_channel_convs.add(top_op_node.name)
        to_prun_channel_convs = list(to_prun_channel_convs)
        to_prun_channel_convs.sort()
        pruning_filter_to_channel_ops_chain[_op_name] = to_prun_channel_convs

    conv_bn_chain = dict()
    for _bn_op in ops_type_map['BatchNormalization']:
        _op_name = _bn_op['name']
        bn_node = op_nodes[_op_name]
        assert len(bn_node.in_tensor) == 1, 'len error'
        bottom_tensor_node = bn_node.in_tensor[0]
        assert len(bottom_tensor_node.in_op) == 1, 'len error'
        bottom_op_node = bottom_tensor_node.in_op[0]
        assert bottom_op_node.attr['type'] == 'Conv', 'Architecture not support. BN must to follow by Conv.'
        bottom_conv_op_name = bottom_op_node.name
        conv_bn_chain[bottom_conv_op_name] = _op_name
    
    # for k in conv_bn_chain:
    #     print('{} -> {}'.format(k, conv_bn_chain[k]))
    # pdb.set_trace()



    def is_conv_chain_valid(conv_module_1, conv_module_2):
        w1_shape = conv_module_1.weight.shape
        w2_shape = conv_module_2.weight.shape
        return w1_shape[0] == w2_shape[1]

    def prune_conv_channel(module_name, conv_modules_map, bn_modules_map, conv_bn_chain, pruning_num=7):
        assert not conv_modules_map[module_name].bias, 'not support'
        print('> Pruning Channels : {}'.format(module_name))
        conv_module = conv_modules_map[module_name]
        conv_weight = conv_module.weight
        w_shape = filter_num, channel_num, kernel_h, kernel_w = conv_weight.shape
        print('    original weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))

        rand_tensor = torch.rand((filter_num, channel_num - pruning_num, kernel_h, kernel_w))
        rand_param  = torch.nn.Parameter(rand_tensor)
        conv_module.weight = rand_param
        w_shape = filter_num, channel_num, kernel_h, kernel_w = rand_param.shape
        print('     pruning weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))
        #conv_bias = conv_module.bias

        for m_name in pruning_channel_to_filter_ops_chain[module_name]:
            if not is_conv_chain_valid(conv_modules_map[m_name], conv_module):
                prune_conv_filter(m_name, conv_modules_map, bn_modules_map, conv_bn_chain, pruning_num=pruning_num)

    def prune_conv_filter(module_name, conv_modules_map, bn_modules_map, conv_bn_chain, pruning_num=7):
        assert not conv_modules_map[module_name].bias, 'not support'
        print('> Pruning Filters : {}'.format(module_name))
        conv_module = conv_modules_map[module_name]
        conv_weight = conv_module.weight
        w_shape = filter_num, channel_num, kernel_h, kernel_w = conv_weight.shape
        print('    original weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))

        rand_tensor = torch.rand((filter_num - pruning_num, channel_num, kernel_h, kernel_w))
        rand_param  = torch.nn.Parameter(rand_tensor)
        conv_module.weight = rand_param
        w_shape = filter_num, channel_num, kernel_h, kernel_w = rand_param.shape
        print('     pruning weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))

        bn_module_name = conv_bn_chain.get(module_name, None)
        if not bn_module_name:
            return
        print(' - Pruning BN Channels : {}'.format(bn_module_name))
        print('     init BN, channels num = {}'.format(filter_num))
        bn_module = bn_modules_map[bn_module_name]
        #print(_num_filters, bn_module.eps, bn_module.momentum, bn_module.affine, bn_module.track_running_stats)
        bn_module.__init__(filter_num, bn_module.eps, bn_module.momentum, bn_module.affine, bn_module.track_running_stats)

        for m_name in pruning_filter_to_channel_ops_chain[module_name]:
            if not is_conv_chain_valid(conv_module, conv_modules_map[m_name]):
                prune_conv_channel(m_name, conv_modules_map, bn_modules_map, conv_bn_chain, pruning_num=pruning_num)


    print('Pruning Channel to Filter Ops Chain')
    for k in pruning_channel_to_filter_ops_chain:
        if k not in ['backbone.layer1.0.conv1',
                     'backbone.layer1.1.conv1',
                     'backbone.layer2.0.conv1',
                     'backbone.layer2.0.downsample.0']:
            continue
        print(f'> {k}')
        for c in  pruning_channel_to_filter_ops_chain[k]:
            print(f'  - {c}')

    print('Pruning Filter to Channel Ops Chain')
    for k in pruning_filter_to_channel_ops_chain:
        if k not in ['backbone.conv1',
                     'backbone.layer1.0.conv2',
                     'backbone.layer1.1.conv2',]:
            continue
        print(f'> {k}')
        for c in  pruning_filter_to_channel_ops_chain[k]:
            print(f'  - {c}')

    # pdb.set_trace()

    all_conv_op_name = [_op['name'] for _op in ops_type_map['Conv']]

    for i, conv_op_name in enumerate(all_conv_op_name):
        conv_op_area_name = conv_op_name.split('.')[0]
        if conv_op_area_name != 'backbone':
            continue
        print(conv_op_name)

        #if i >= 2:
        #    continue
        if op_nodes[conv_op_name].in_tensor[0].name == INPUT_TENSOR_ID:
            print('top conv op, skip')
            continue

        del net_model
        del conv_modules_map, bn_modules_map
        if args.architecture == 'ksevendet':
            #net_model = ksevendet.KSevenDet(ksevendet_cfg, num_classes=args.num_classes, pretrained=False, **build_param)
            net_model = ksevendet.KSevenDet(ksevendet_cfg, num_classes=args.num_classes, pretrained=False)
        else:
            assert 0, 'architecture error'

        # The following statement is important, make sure all weights are in GPU
        net_model = net_model.cuda()

        module_name = []
        features_in_hook = []
        features_out_hook = []
        conv_weights = []
        conv_modules_map = collections.OrderedDict()
        bn_modules_map = collections.OrderedDict()
        hook_handles = []
        with torch.no_grad():
            net_model.apply(module_recorder)
            net_model(sample_input, return_head=True)
            # Unregister from the forward hooks
            for handle in hook_handles:
                handle.remove()

        prune_conv_channel(conv_op_name, conv_modules_map, bn_modules_map, conv_bn_chain) 
        # prune_conv_channel(conv_op_name, conv_modules_map, bn_modules_map) 

        print('Pruning Done.')

        # The following statement is important, make sure all weights are in GPU
        net_model = net_model.cuda()
        net_model.set_onnx_convert_info(fixed_size=(height, width))    

        print('new model inference')
        model_summaries.model_summary(net_model, 'compute', input_shape=sample_input_shape)

        
    # SHOW_TYPE = 'Transpose'
    #for _op in ops_type_map[SHOW_TYPE]:
    #    print('({})  {}'.format(_op['type'], _op['name']))
    #    print('  In  ({}): {}'.format(len(_op['inputs']), str(_op['inputs'])))
    #    print('  Out ({}): {}'.format(len(_op['outputs']), str(_op['outputs'])))


    # from distiller.summary_graph import SummaryGraph
    '''
    equivalence_shape_tensors_map = dict()
    equivalence_shape_tensors_map_inverse = dict()
    # Remove ReLU
    print('Start remove ReLU...')
    refine_ops   = copy.deepcopy(g.ops)
    refine_param = copy.deepcopy(g.params) 
    refine_edges = copy.deepcopy(g.edges)
    for op in refine_ops.values():
        if not op['type'] == 'Relu':
            continue
        print()
        print(op['name'])
        t_in  = op['inputs'][0]
        t_out = op['outputs'][0]

        _tmp_new_e = []
        for enumerate(edge in refine_edges:
            if edge.dst == t_in:
                print(edge)
                new_edge = SummaryGraph.Edge(t_in, edge.dst)
                _tmp_new_e.append(new_edge)
                refine_edges.append(new_edge)
        #print(_tmp_new_e)

        #equivalence_shape_tensors_map[op['inputs'][0]]  = t_merge 
        #equivalence_shape_tensors_map[op['outputs'][0]] = t_merge
        #equivalence_shape_tensors_map_inverse[t_merge] = set() 
        #equivalence_shape_tensors_map_inverse[t_merge].add(op['inputs'][0])
        #equivalence_shape_tensors_map_inverse[t_merge].add(op['outputs'][0])
        #else:
        #    assert 0, 'to check ....'
    exit(0)
    '''

    """Create a PNG object containing a graphiz-dot graph of the network,
    as represented by SummaryGraph 'sgraph'.

    Args:
        sgraph (SummaryGraph): the SummaryGraph instance to draw.
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
    """

    def annotate_op_node(op):
        if op['type'] == 'Conv':
            return ["sh={}".format(distiller.size2str(op['attrs']['kernel_shape'])),
                    "g={}".format(str(op['attrs']['group']))]
        return ''   

    # op_nodes = [op['name'] for op in sgraph.ops.values()]
    # data_nodes = []
    # param_nodes = []
    # for id, param in sgraph.params.items():
    #     n_data = (id, str(distiller.volume(param['shape'])), str(param['shape']))
    #     if data_node_has_parent(sgraph, id):
    #         data_nodes.append(n_data)
    #     else:
    #         param_nodes.append(n_data)
    # edges = sgraph.edges

    # if not display_param_nodes:
    #     # Use only the edges that don't have a parameter source
    #     non_param_ids = op_nodes + [dn[0] for dn in data_nodes]
    #     edges = [edge for edge in sgraph.edges if edge.src in non_param_ids]
    #     param_nodes = None

    # op_nodes_desc = [(op['name'], op['type'], *annotate_op_node(op)) for op in sgraph.ops.values()]
    # pydot_graph = create_pydot_graph(op_nodes_desc, data_nodes, param_nodes, edges, rankdir, styles)
    # png = pydot_graph.create_png()
    # return png


    #print(g.__dict__.keys())
    ##print(g.module_ops_map)
    #print(g.ops)
    #print(g.missing_modules())

class SNode(object):
    def __init__(self, name, node_type, attr=None):
        self.name = name
        self.node_type = node_type

        if attr is not None:
            self.attr = attr
        elif node_type == 'tensor':
            self.attr = {
                'shape' : None,
            }
        elif node_type == 'op':
            self.attr = {
                'type' : None,
            }
        else:
            raise ValueError('node type error')

        if node_type == 'tensor':
            self.in_op = set()
            self.out_op = set()
        else:
            self.in_tensor = set()
            self.out_tensor = set()


    def set_attribute(self, attr_info):
        pass


def generate_equivalence_tensor_map(ops_type_map, op_nodes, tensor_nodes):
    print('generate equivalence tensor map.')
    eq_tensors_map = dict()
    eq_tensors_map_inverse = collections.defaultdict(set) 

    eq_ops_type = ['Add', 'Relu', 'BatchNormalization', 'MaxPool', 'Upsample', 'Conv']
    for _type in eq_ops_type:
        equivalence_tensor_at_op(_type, ops_type_map[_type], op_nodes, tensor_nodes, 
                                 eq_tensors_map, eq_tensors_map_inverse)

    for eq_t_id in eq_tensors_map_inverse:
        t_ids = eq_tensors_map_inverse.pop(eq_t_id)
        min_eq_t_id = min(t_ids)
        for t_id in t_ids:
            eq_tensors_map[t_id] = min_eq_t_id
            eq_tensors_map_inverse[min_eq_t_id].add(t_id)

    return eq_tensors_map, eq_tensors_map_inverse


def equivalence_tensor_at_op(op_type, ops, op_nodes, tensor_nodes, 
                             equivalence_shape_tensors_map, equivalence_shape_tensors_map_inverse):
    print('merge op: {}'.format(op_type))
    assert op_type in ['Add', 'Relu', 'BatchNormalization', 'MaxPool', 'Upsample', 'Conv']
    for _op in ops:
        if op_type == 'Add':
            assert len(_op['outputs']) == 1, 'op output num error'
        elif op_type == 'Relu':
            assert len(_op['inputs']) == 1, 'op input num error'
            assert len(_op['outputs']) == 1, 'op output num error'
        elif op_type == 'BatchNormalization':
            assert len(_op['inputs']) == 5, 'op input num error'
            assert len(_op['outputs']) == 1, 'op output num error'
        elif op_type == 'MaxPool':
            assert len(_op['inputs']) == 1, 'op input num error'
            assert len(_op['outputs']) == 1, 'op output num error'
        elif op_type == 'Upsample':
            assert len(_op['inputs']) == 2, 'op input num error'
            assert len(_op['outputs']) == 1, 'op output num error'
        elif op_type == 'Conv':
            # TODO: What situation lead to 3 inputs.
            assert len(_op['inputs']) in [2, 3], 'op input num error'
            assert len(_op['outputs']) == 1, 'op output num error'
            if op_nodes[_op['name']].attr['group'] == 1:
                continue
        else:
            assert 0, 'Op type error.'
        
        if op_type == 'Add':
            in_t_ids = _op['inputs']
            out_t_ids = [_op['outputs'][0],]
        else:
            in_t_ids = [_op['inputs'][0],]
            out_t_ids = [_op['outputs'][0],]
        connect_t_ids = in_t_ids + out_t_ids
        
        equivalence_t_id = None
        for t_id in connect_t_ids:
            if equivalence_shape_tensors_map.get(t_id, None):
                equivalence_t_id = equivalence_shape_tensors_map[t_id]
                break
        if not equivalence_t_id:
            equivalence_t_id = min(connect_t_ids)
        else:
            # check equivalence tensor is unique, if not, need to merge them
            total_equivalence_t_ids = [equivalence_t_id,]
            for t_id in connect_t_ids:
                eq_t_id = equivalence_shape_tensors_map.get(t_id, None)
                if eq_t_id is not None and eq_t_id != equivalence_t_id:
                    total_equivalence_t_ids.append(eq_t_id)
            # merge start
            if len(total_equivalence_t_ids) != 1:
                min_equivalence_t_id = min(total_equivalence_t_ids)
                for eq_t_id in total_equivalence_t_ids:
                    if eq_t_id == min_equivalence_t_id:
                        continue
                    t_ids = equivalence_shape_tensors_map_inverse.pop(eq_t_id)
                    for t_id in t_ids:
                        equivalence_shape_tensors_map[t_id] = min_equivalence_t_id
                        equivalence_shape_tensors_map_inverse[min_equivalence_t_id].add(t_id)
                equivalence_t_id = min_equivalence_t_id
            # merge end
        for t_id in connect_t_ids:
            equivalence_shape_tensors_map[t_id] = equivalence_t_id
            equivalence_shape_tensors_map_inverse[equivalence_t_id].add(t_id)


def data_node_has_parent(graph, t_id):
    for edge in graph.edges:
        if edge.dst == t_id:
            return True
    return False


if __name__ == '__main__':
    main()
