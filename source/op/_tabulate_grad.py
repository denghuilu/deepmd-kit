#!/usr/bin/env python3
"""
Gradients for tabulate.
"""

from tensorflow.python.framework import ops
from deepmd.env import op_module
from deepmd.env import tf
# from deepmd.DescrptSeATabulate import last_layer_size

# refine is needed!
# accurate gradient is needed!
# 'tabulate_one_side' is needed!
@ops.RegisterGradient("TabulateGrad")
def _tabulate_grad_grad_cc (op, dy):    
    return [None, None, dy, None, None]

@ops.RegisterGradient("TabulateFusionGrad")
def _tabulate_fusion_grad_grad_cc (op, dy, dy_):    
    return [None, None, dy, dy_, None, None]

# old implementations here.

@ops.RegisterGradient("Tabulate")
def _tabulate_grad_cc (op, dy):    
    dy = op_module.tabulate_grad(op.inputs[0], op.inputs[1], op.inputs[2], dy, op.inputs[3], last_layer_size = op.get_attr("last_layer_size"))
    return [None, None, dy, None]


@ops.RegisterGradient("TabulateFusion")
def _tabulate_fusion_grad_cc (op, dy):    
    dy_dx, dy_df = op_module.tabulate_fusion_grad(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], dy, op.outputs[0])
    return [None, None, dy_dx, dy_df]