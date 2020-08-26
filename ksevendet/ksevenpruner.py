from functools import partial

import torch
import numpy as np
import logging
import random
import collections

from PIL import Image

from distiller import model_summaries
from distiller import model_find_module_name
from distiller import SummaryGraph

import pdb


class KSevenPruner(object):
    def __init__(self, net_model, input_shape=(512, 512, 3), tensor_pruning_dependency=None,
                 **kwargs):
        self.logger      = kwargs.get('logger', None)
        self.net_model   = net_model
        self.input_shape = input_shape
        self.tensor_pruning_dependency = tensor_pruning_dependency
        sample_input = torch.from_numpy(np.zeros(input_shape).astype(np.float32))
        sample_input = sample_input.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
        sample_input_shape = sample_input.shape
        summary_table, total_macs = model_summaries.model_summary(net_model, 'compute', 
                                                                  input_shape=sample_input_shape)
        self.summary_table = summary_table
        self.total_macs = total_macs

        if self.logger is not None:
            self.logger.info('\n{}\nTotal MACs: {:,}'.format(summary_table, total_macs))

        def conv_hook(module, fea_in, fea_out, model):
            mod_name = model_find_module_name(model, module)
            # print('find Conv module: {}'.format(mod_name))
            conv_modules_map[mod_name] = module
            return None

        def bn_hook(module, fea_in, fea_out, model):
            mod_name = model_find_module_name(model, module)
            # print('find BN module: {}'.format(mod_name))
            bn_modules_map[mod_name] = module
            return None

        def fc_hook(module, fea_in, fea_out, model):
            mod_name = model_find_module_name(model, module)
            # print('find FC module: {}'.format(mod_name))
            fc_modules_map[mod_name] = module
            return None

        def module_recorder(m):
            if isinstance(m, torch.nn.Conv2d):
                hook_handles.append(m.register_forward_hook(partial(conv_hook, model=net_model)))
            if isinstance(m, torch.nn.BatchNorm2d):
                hook_handles.append(m.register_forward_hook(partial(bn_hook, model=net_model)))
            if isinstance(m, torch.nn.Linear):
                hook_handles.append(m.register_forward_hook(partial(fc_hook, model=net_model)))

        conv_modules_map = collections.OrderedDict()
        bn_modules_map = collections.OrderedDict()
        fc_modules_map = collections.OrderedDict()
        hook_handles = []
        with torch.no_grad():
            net_model.apply(module_recorder)
            # net_model(sample_input, return_head=True)
            net_model(sample_input)

            # Unregister from the forward hooks
            for handle in hook_handles:
                handle.remove()
        
        self.conv_modules_map = conv_modules_map
        self.bn_modules_map = bn_modules_map
        self.fc_modules_map = fc_modules_map

    def gen_summary_graph(self, write_image=False, image_path='g_image.jpg'):
        sample_input = torch.from_numpy(np.zeros(self.input_shape).astype(np.float32))
        sample_input = sample_input.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
        g = SummaryGraph(self.net_model, sample_input)
        if write_image:
            from distiller import create_png
            import io
            png = create_png(g)
            stream = io.BytesIO(png)
            g_image = Image.open(stream).convert('RGB')
            print('Save summary graph image to {}'.format(image_path))
            g_image.save(image_path, 'jpeg')
            print('Write done.')
        return g
    
    def gen_tensor_pruning_dependency(self, dump_json=False, 
                                      dump_json_path='tensor_pruning_dependency.json'):
        # g = self.gen_summary_graph()
        g = self.gen_summary_graph(write_image=True)

        IGNORE_OP_TYPE = ['Constant',]
        INPUT_TENSOR_ID = 'input.1'

        ops_keys = list(g.ops.keys())
        ops_type_map = collections.defaultdict(list)
        for k in ops_keys:
            ops_type_map[g.ops[k]['type']].append(g.ops[k])
        print(ops_type_map.keys())
        # pdb.set_trace()

        # OP Types Example
        # resnet18: ['Conv', 'BatchNormalization', 'Relu', 'MaxPool', 
        #            'Add', 'GlobalAveragePool', 'Flatten', 'Gemm', 'Softmax']
        # mobilenetv2: ['Conv', 'BatchNormalization', 'Clip', 
        #               'Add', 'GlobalAveragePool', 'Flatten', 'Gemm', 'Softmax']
        # resnet18 + bifpn + simple-header:
        #       ['Conv', 'BatchNormalization', 'Relu', 'MaxPool', 
        #        'Add', 'Constant', 'Upsample', 'Transpose', 'Reshape', 'Concat', 'Sigmoid']

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
            if op_info['type'] in IGNORE_OP_TYPE:
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
            if len(op_info['outputs']) != 1:
                pdb.set_trace()
            assert len(op_info['outputs']) == 1, 'op output num error'
            for t_id in op_info['outputs']:
                out_node = tensor_nodes.get(t_id, None)
                assert in_node is not None, 'tensor node not found.'
                _node.out_tensor.add(out_node)
                out_node.in_op.add(_node)
            op_nodes[ops_name] = _node
    
        for t_id in tensor_nodes:
            _tensor_node = tensor_nodes[t_id]
            _tensor_node.in_op = list(_tensor_node.in_op)
            _tensor_node.out_op = list(_tensor_node.out_op)
        for op_name in op_nodes:
            _op_node = op_nodes[op_name]
            _op_node.in_tensor = list(_op_node.in_tensor)
            _op_node.out_tensor = list(_op_node.out_tensor)
    
        eq_tensors_map, eq_tensors_map_inverse = generate_equivalence_tensor_map(ops_type_map, op_nodes, tensor_nodes)
        eq_tensors_ids = list(eq_tensors_map_inverse.keys())
        eq_tensors_ids.sort()

        tensor_pruning_dependency = dict()
        for eq_t_id in eq_tensors_ids:
            tensor_pruning_dependency[eq_t_id] = {
                'filter_ops': set(),
                'channel_ops': set(),
                'equivalence_tensor_id': eq_tensors_map_inverse[eq_t_id],
            }
            print('\n{} >> [ {} ]'.format(eq_t_id, ', '.join(eq_tensors_map_inverse[eq_t_id])))
            for t_id in eq_tensors_map_inverse[eq_t_id]:
                _tensor_node = tensor_nodes[t_id]
                assert len(_tensor_node.in_op) == 1, 'in op len error'
                _in_op_node = _tensor_node.in_op[0]
                if _in_op_node.attr['type'] in ['Conv',]:
                    tensor_pruning_dependency[eq_t_id]['filter_ops'].add((_in_op_node.attr['type'], _in_op_node.name))
                for _out_op_node in _tensor_node.out_op:
                    if _out_op_node.attr['type'] in ['Conv', 'BatchNormalization', 'Gemm',]:
                        if _out_op_node.attr['type'] == 'Conv' and _out_op_node.attr['group'] != 1:
                            continue 
                        tensor_pruning_dependency[eq_t_id]['channel_ops'].add((_out_op_node.attr['type'], _out_op_node.name))

            tensor_pruning_dependency[eq_t_id]['filter_ops'] = list(tensor_pruning_dependency[eq_t_id]['filter_ops'])
            tensor_pruning_dependency[eq_t_id]['channel_ops'] = list(tensor_pruning_dependency[eq_t_id]['channel_ops'])
            print(' - Filter Ops:')
            for _op_name in tensor_pruning_dependency[eq_t_id]['filter_ops']:
                print('   @', _op_name)
            print(' - Clannel Ops:')
            for _op_name in tensor_pruning_dependency[eq_t_id]['channel_ops']:
                print('   @', _op_name)

        if dump_json:
            import json
            #dump_json_path = '{}_tensor_pruning_dependency.json'.format(self.network_name)
            print('dump to file: {}'.format(dump_json_path), end='\r')
            with open(dump_json_path, 'w') as fp:
                json.dump(tensor_pruning_dependency, fp, indent=4)
            print('dump to file: {} [DONE]'.format(dump_json_path), end='\r')

        self.tensor_pruning_dependency = tensor_pruning_dependency
        return tensor_pruning_dependency
    
    def prune(self, pruning_tensor_cfg):
        assert hasattr(self, 'tensor_pruning_dependency'), 'Run gen_tensor_pruning_dependency() first.'

        print('='*60)
        for pruning_cfg in pruning_tensor_cfg:
            pruning_eq_tensor_id, pruning_args = pruning_cfg
            pruning_args['logger'] = self.logger

            if self.logger is not None:
                self.logger.info('Pruning Tensor ID: {}'.format(pruning_eq_tensor_id))
                self.logger.info('Equivalence Tensor IDs: {}'.format(
                                 self.tensor_pruning_dependency[pruning_eq_tensor_id]['equivalence_tensor_id']))
                self.logger.info('>> Filter Ops:')
                for _op_name in self.tensor_pruning_dependency[pruning_eq_tensor_id]['filter_ops']:
                    self.logger.info('   @{}'.format(_op_name))
                self.logger.info('>> Clannel Ops:')
                for _op_name in self.tensor_pruning_dependency[pruning_eq_tensor_id]['channel_ops']:
                    self.logger.info('   @{}'.format(_op_name))
            pruning_filter_ops = self.tensor_pruning_dependency[pruning_eq_tensor_id]['filter_ops']
            for _op in pruning_filter_ops:
                _op_type, _op_name = _op
                if _op_type == 'Conv':
                    prune_conv_filter(_op_name, self.conv_modules_map[_op_name], **pruning_args)
                else:
                    assert 0, 'op type error'
            pruning_channel_ops = self.tensor_pruning_dependency[pruning_eq_tensor_id]['channel_ops']
            for _op in pruning_channel_ops:
                _op_type, _op_name = _op
                if _op_type == 'Conv':
                    prune_conv_channel(_op_name, self.conv_modules_map[_op_name], **pruning_args)
                elif _op_type == 'BatchNormalization':
                    prune_bn_channel(_op_name, self.bn_modules_map[_op_name], **pruning_args)
                elif _op_type == 'Gemm':
                    prune_fc_channel(_op_name, self.fc_modules_map[_op_name], **pruning_args)
                else:
                    assert 0, 'op type error'

        sample_input = torch.from_numpy(np.zeros(self.input_shape).astype(np.float32))
        sample_input = sample_input.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
        sample_input_shape = sample_input.shape

        DEBUG_INFERENCE = False
        if DEBUG_INFERENCE:
            # for name, param in self.net_model.named_parameters():
            #     if 'weight' in name and 'bifpn' in name and 'down_channel' in name:
            #         print(name)
            #         print(param.shape)

            def conv_printer_hook(module, fea_in, fea_out, model):
                mod_name = model_find_module_name(model, module)
                # print('find Conv module: {}'.format(mod_name))
                return None

            def bn_printer_hook(module, fea_in, fea_out, model):
                mod_name = model_find_module_name(model, module)
                # print('find BN module: {}'.format(mod_name))
                return None

            def fc_printer_hook(module, fea_in, fea_out, model):
                mod_name = model_find_module_name(model, module)
                print('find FC module: {}'.format(mod_name))
                return None

            def module_printer(m):
                if isinstance(m, torch.nn.Conv2d):
                    hook_handles.append(m.register_forward_hook(partial(conv_printer_hook, model=self.net_model)))
                if isinstance(m, torch.nn.BatchNorm2d):
                    hook_handles.append(m.register_forward_hook(partial(bn_printer_hook, model=self.net_model)))
                if isinstance(m, torch.nn.Linear):
                    hook_handles.append(m.register_forward_hook(partial(fc_printer_hook, model=self.net_model)))

            hook_handles = []
            with torch.no_grad():
                self.net_model.apply(module_printer)
                self.net_model(sample_input)

                # Unregister from the forward hooks
                for handle in hook_handles:
                    handle.remove()

        summary_table, total_macs = model_summaries.model_summary(self.net_model, 'compute', 
                                                                  input_shape=sample_input_shape)
        if self.logger is not None:
            self.logger.info('\n{}\nTotal MACs: {:,}'.format(summary_table, total_macs))
        self.summary_table_pruned = summary_table
        self.total_macs_pruned = total_macs
        # print('summary skip.')


def prune_conv_filter(module_name, conv_module, pruning_type='random', 
                      pruning_num=None, pruning_rate=None, keep_idxes=None, logger=None):
    # assert conv_module.bias is None, 'not support'
    w_shape = filter_num, channel_num, kernel_h, kernel_w = conv_module.weight.shape
    assert conv_module.groups in [1, filter_num], 'Conv groups not support'
    if pruning_type == 'random':
        assert type(pruning_num) is int or type(pruning_rate) is float, 'pruning num (or rate) error'
        if type(pruning_rate) is float:
            pruning_num = int(pruning_rate * filter_num)
        assert pruning_num < filter_num, 'pruning num need between [0, {})'.format(filter_num)
        keep_idxes = random.sample([idx for idx in range(filter_num)], filter_num-pruning_num)
        keep_idxes.sort()
    else:
        assert 0, 'pruning_type not support now'
    if pruning_num == 0:
        # pdb.set_trace()
        return 
    
    ''' torch.nn.Conv2d.weight
    self.weight = Parameter(torch.Tensor(
                            out_channels, in_channels // groups, *kernel_size))
    '''
    if conv_module.groups == filter_num:
        # Case: depthwise separable convolution
        conv_module.groups = filter_num - pruning_num

    if logger is not None:
        logger.info('Pruning Conv Filters : {}'.format(module_name))
        logger.info('Type: {}, Num: {} (rate:{:.2f})'.format(
                    pruning_type, pruning_num, float(pruning_num)/filter_num))
        logger.info('    original weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))
    _device = conv_module.weight.device
    np_weight = conv_module.weight.cpu().detach().numpy() if _device == torch.device('cuda:0') \
                                                          else conv_module.weight.detach().numpy()
    np_weight = np_weight[keep_idxes,:,:,:]
    pruned_weight_tensor = torch.from_numpy(np_weight).float().to(_device)
    # del conv_module.weight
    conv_module.weight = torch.nn.Parameter(pruned_weight_tensor)

    if conv_module.bias is not None:
        np_bias = conv_module.bias.cpu().detach().numpy() if _device == torch.device('cuda:0') \
                                                        else conv_module.bias.detach().numpy()
        np_bias = np_bias[keep_idxes]
        pruned_bias_tensor = torch.from_numpy(np_bias).float().to(_device)
        conv_module.bias = torch.nn.Parameter(pruned_bias_tensor)

    w_shape = filter_num, channel_num, kernel_h, kernel_w = conv_module.weight.shape
    if logger is not None:
        logger.info('     pruning weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))


def prune_conv_channel(module_name, conv_module, pruning_type='random', 
                       pruning_num=None, pruning_rate=None, keep_idxes=None, logger=None):
    w_shape = filter_num, channel_num, kernel_h, kernel_w = conv_module.weight.shape
    if pruning_type == 'random':
        assert type(pruning_num) is int or type(pruning_rate) is float, 'pruning num (or rate) error'
        if type(pruning_rate) is float:
            pruning_num = int(pruning_rate * channel_num)
        assert pruning_num < channel_num, 'pruning num need between [0, {})'.format(channel_num)
        keep_idxes = random.sample([idx for idx in range(channel_num)], channel_num-pruning_num)
        keep_idxes.sort()
    else:
        assert 0, 'pruning_type not support now'
    if pruning_num == 0:
        # pdb.set_trace()
        return 

    if logger is not None:
        logger.info('Pruning Conv Channels : {}'.format(module_name))
        logger.info('Type: {}, Num: {} (rate:{:.2f})'.format(
                    pruning_type, pruning_num, float(pruning_num)/channel_num))
        logger.info('    original weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))
    _device = conv_module.weight.device
    np_weight = conv_module.weight.cpu().detach().numpy() if _device == torch.device('cuda:0') \
                                                          else conv_module.weight.detach().numpy()
    np_weight = np_weight[:,keep_idxes,:,:]
    pruned_weight_tensor = torch.from_numpy(np_weight).float().to(_device)
    conv_module.weight = torch.nn.Parameter(pruned_weight_tensor)

    w_shape = filter_num, channel_num, kernel_h, kernel_w = conv_module.weight.shape
    if logger is not None:
        logger.info('     pruning weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))

def prune_bn_channel(module_name, bn_module, pruning_type='random', 
                     pruning_num=None, pruning_rate=None, keep_idxes=None, logger=None):
    assert bn_module.affine, 'BN only support affine=True, now'
    assert bn_module.track_running_stats, 'BN only support track_running_stats=True, now'
    assert len(bn_module.weight.shape) == 1, 'BN weight shape error'
    channel_num = bn_module.weight.shape[0]
    if pruning_type == 'random':
        assert type(pruning_num) is int or type(pruning_rate) is float, 'pruning num (or rate) error'
        if type(pruning_rate) is float:
            pruning_num = int(pruning_rate * channel_num)
        assert pruning_num < channel_num, 'pruning num need between [0, {})'.format(channel_num)
        keep_idxes = random.sample([idx for idx in range(channel_num)], channel_num-pruning_num)
        keep_idxes.sort()
    else:
        assert 0, 'pruning_type not support now'
    if pruning_num == 0:
        return 
        
    if logger is not None:
        logger.info('Pruning BN Channels : {}'.format(module_name))
        logger.info('Type: {}, Num: {} (rate:{:.2f})'.format(
                    pruning_type, pruning_num, float(pruning_num)/channel_num))
        logger.info('    original weight shape = [{:>4}]'.format(bn_module.weight.shape[0]))
        logger.info('    original bias shape   = [{:>4}]'.format(bn_module.bias.shape[0]))

    _device = bn_module.weight.device
    np_weight = bn_module.weight.cpu().detach().numpy() if _device == torch.device('cuda:0') \
                                                        else bn_module.weight.detach().numpy()
    np_weight = np_weight[keep_idxes]
    pruned_weight_tensor = torch.from_numpy(np_weight).float().to(_device)
    bn_module.weight = torch.nn.Parameter(pruned_weight_tensor)
    np_bias = bn_module.bias.cpu().detach().numpy() if _device == torch.device('cuda:0') \
                                                    else bn_module.bias.detach().numpy()
    np_bias = np_bias[keep_idxes]
    pruned_bias_tensor = torch.from_numpy(np_bias).float().to(_device)
    bn_module.bias = torch.nn.Parameter(pruned_bias_tensor)

    np_running_mean = bn_module.running_mean.cpu().detach().numpy() if _device == torch.device('cuda:0') \
                                                                    else bn_module.running_mean.detach().numpy()
    np_running_mean = np_running_mean[keep_idxes]
    bn_module.running_mean = torch.from_numpy(np_running_mean).float().to(_device)

    np_running_var = bn_module.running_var.cpu().detach().numpy() if _device == torch.device('cuda:0') \
                                                                    else bn_module.running_var.detach().numpy()
    np_running_var = np_running_var[keep_idxes]
    bn_module.running_var = torch.from_numpy(np_running_var).float().to(_device)
        
    if logger is not None:
        logger.info('    pruning weight shape = [{:>4}]'.format(bn_module.weight.shape[0]))
        logger.info('    pruning bias shape   = [{:>4}]'.format(bn_module.bias.shape[0]))

def prune_fc_channel(module_name, fc_module, pruning_type='random', 
                     pruning_num=None, pruning_rate=None, keep_idxes=None, logger=None):
    assert len(fc_module.weight.shape) == 2, 'FC weight shape error'
    w_shape = filter_num, channel_num = fc_module.weight.shape
    if pruning_type == 'random':
        assert type(pruning_num) is int or type(pruning_rate) is float, 'pruning num (or rate) error'
        if type(pruning_rate) is float:
            pruning_num = int(pruning_rate * channel_num)
        assert pruning_num < channel_num, 'pruning num need between [0, {})'.format(channel_num)
        keep_idxes = random.sample([idx for idx in range(channel_num)], channel_num-pruning_num)
        keep_idxes.sort()
    else:
        assert 0, 'pruning_type not support now'
    if pruning_num == 0:
        return 
        
    if logger is not None:
        logger.info('Pruning FC Channels : {}'.format(module_name))
        logger.info('Type: {}, Num: {} (rate:{:.2f})'.format(
                    pruning_type, pruning_num, float(pruning_num)/channel_num))
        logger.info('    original weight shape = [{:>4}]'.format(channel_num))

    _device = fc_module.weight.device
    np_weight = fc_module.weight.cpu().detach().numpy() if _device == torch.device('cuda:0') \
                                                        else fc_module.weight.detach().numpy()
    np_weight = np_weight[:, keep_idxes]
    pruned_weight_tensor = torch.from_numpy(np_weight).float().to(_device)
    fc_module.weight = torch.nn.Parameter(pruned_weight_tensor)
    if logger is not None:
        logger.info('    pruning weight shape = [{:>4}]'.format(fc_module.weight.shape[0]))


def generate_equivalence_tensor_map(ops_type_map, op_nodes, tensor_nodes):
    print('generate equivalence tensor map.')
    eq_tensors_map = dict()
    eq_tensors_map_inverse = collections.defaultdict(set) 

    eq_ops_type = ['Add', 'Mul', 'Relu', 'Clip', 'Sigmoid', 'BatchNormalization', 
                   'MaxPool', 'GlobalAveragePool', 'Upsample', 'Flatten', 'Conv']
    for _type in eq_ops_type:
        equivalence_tensor_at_op(_type, ops_type_map[_type], op_nodes, tensor_nodes, 
                                 eq_tensors_map, eq_tensors_map_inverse)
    # To check whether we consider all tensors.
    for _conv_op in ops_type_map['Conv']:
        out_t_id = _conv_op['outputs'][0]
        if out_t_id in eq_tensors_map:
            continue
        out_t_node = tensor_nodes[out_t_id]
        for _op_node in out_t_node.out_op:
            if  _op_node.attr['type'] in ['Conv',]:
                assert 0, 'Find alone tensor.'
                pdb.set_trace()
    for _fc_op in ops_type_map['Gemm']:
        out_t_id = _fc_op['outputs'][0]
        if out_t_id in eq_tensors_map:
            continue
        out_t_node = tensor_nodes[out_t_id]
        for _op_node in out_t_node.out_op:
            if  _op_node.attr['type'] in ['Gemm',]:
                assert 0, 'Find alone tensor.'
                pdb.set_trace()

    eq_t_id_list = list(eq_tensors_map_inverse.keys())
    for eq_t_id in eq_t_id_list:
        t_ids = eq_tensors_map_inverse.pop(eq_t_id)
        min_eq_t_id = min(t_ids)
        for t_id in t_ids:
            eq_tensors_map[t_id] = min_eq_t_id
            eq_tensors_map_inverse[min_eq_t_id].add(t_id)
        eq_tensors_map_inverse[min_eq_t_id] = list(eq_tensors_map_inverse[min_eq_t_id])
        eq_tensors_map_inverse[min_eq_t_id].sort()

    return eq_tensors_map, eq_tensors_map_inverse


def equivalence_tensor_at_op(op_type, ops, op_nodes, tensor_nodes, 
                             equivalence_shape_tensors_map, equivalence_shape_tensors_map_inverse):
    print('merge op: {}'.format(op_type))
    # TODO:
    #   check: Mul op (In SEModule, mul is equivalence tensor op)
    assert op_type in ['Add', 'Mul', 'Relu', 'Clip', 'Sigmoid', 'BatchNormalization', 
                       'MaxPool', 'GlobalAveragePool', 'Upsample', 'Flatten', 'Conv']
    for _op in ops:
        if op_type == 'Add':
            assert len(_op['outputs']) == 1, 'op output num error'
        elif op_type == 'Mul':
            assert len(_op['inputs']) == 2, 'op input num error'
            assert len(_op['outputs']) == 1, 'op output num error'
        elif op_type in ['Relu', 'Clip', 'Sigmoid']:
            assert len(_op['inputs']) == 1, 'op input num error'
            assert len(_op['outputs']) == 1, 'op output num error'
        elif op_type == 'BatchNormalization':
            assert len(_op['inputs']) == 5, 'op input num error'
            assert len(_op['outputs']) == 1, 'op output num error'
        elif op_type in ['MaxPool', 'GlobalAveragePool']:
            assert len(_op['inputs']) == 1, 'op input num error'
            assert len(_op['outputs']) == 1, 'op output num error'
        elif op_type == 'Flatten':
            assert len(_op['inputs']) == 1, 'op input num error'
            assert len(_op['outputs']) == 1, 'op output num error'
            assert _op['attrs']['axis'] == 1, 'op attrs error'
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
        
        if op_type in ['Add', 'Mul']:
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