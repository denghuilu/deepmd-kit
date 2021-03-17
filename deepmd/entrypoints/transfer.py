"""Module used for transfering parameters between models."""

from typing import Dict, Optional, Sequence, Tuple
from deepmd.env import tf
import re
import numpy as np
import logging

__all__ = ["transfer"]

log = logging.getLogger(__name__)

@np.vectorize
def convert_number(number: int) -> float:
    binary = bin(number).replace("0b", "").zfill(16)
    sign = int(binary[0]) * -2 + 1
    exp = int(binary[1:6], 2)
    frac = (int(binary[6:], 2) + 2 ** 10) * (2 ** -10)
    return sign * (2 ** (exp - 15)) * frac


def convert_matrix(
    matrix: np.ndarray, shape: Sequence[int], dtype: Optional[type] = None
) -> np.ndarray:
    """Convert matrix of integers to self defined binary format.

    Parameters
    ----------
    matrix : np.ndarray
        array of ints
    shape : Sequence[int]
        shape to cast resulting array to
    dtype : Optional[type]
        type that finall array will be cast to, If None no casting will take place

    Returns
    -------
    np.ndarray
        array cast to required format
    """
    conv = convert_number(matrix.flatten()).reshape(shape)
    if dtype:
        conv = conv.astype(dtype)

    return conv


def transfer(*, old_model: str, raw_model: str, output: str, **kwargs):
    """Transfer operation from old fron graph to new prepared raw graph.

    Parameters
    ----------
    old_model : str
        frozen old graph model
    raw_model : str
        new model that will accept ops from old model
    output : str
        new model with transfered parameters will be saved to this location
    """
    raw_graph = load_graph(raw_model)
    old_graph = load_graph(old_model)
    log.info(f"{len(raw_graph.as_graph_def().node)} ops in the raw graph")
    log.info(f"{len(old_graph.as_graph_def().node)} ops in the old graph")

    new_graph_def = transform_graph(raw_graph, old_graph)
    with tf.gfile.GFile(output, mode="wb") as f:
        f.write(new_graph_def.SerializeToString())
    log.info("the output model is saved in " + output)


def load_graph(graph_name: str) -> tf.Graph:
    """Load graph from passed in path.

    Parameters
    ----------
    graph_name : str
        path to frozen graph on disk

    Returns
    -------
    tf.Graph
        tf graph object
    """
    graph_def = tf.GraphDef()
    with open(graph_name, "rb") as f:
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        return graph


def transform_graph(raw_graph: tf.Graph, old_graph: tf.Graph) -> tf.Graph:
    """Trasform old graph into new.

    Parameters
    ----------
    raw_graph : tf.Graph
        graph receiving parameters from the old one
    old_graph : tf.Graph
        graph providing parameters

    Returns
    -------
    tf.Graph
        new graph with parameters transfered form the old one
    """
    precision_dict = {\
    1:(np.float32, "float32"),\
    2:(np.float64, "float64"),\
    19:(np.float16, "float16")\
    }
    old_graph_def = old_graph.as_graph_def()
    raw_graph_def = raw_graph.as_graph_def()
    raw_graph_node = load_transform_node(raw_graph_def)
    old_graph_node = load_transform_node(old_graph_def)

    for node in raw_graph_def.node:
        if node.name in raw_graph_node.keys():
            check_dim(raw_graph_node, old_graph_node, node.name)
            tensor_shape = [dim.size for dim in raw_graph_node[node.name].tensor_shape.dim]
            old_graph_dtype = precision_dict[old_graph_node[node.name].dtype]
            raw_graph_dtype = precision_dict[raw_graph_node[node.name].dtype]
            log.info("%s is passed from old graph(%s) to raw graph(%s)" % (node.name, old_graph_dtype[1],raw_graph_dtype[1]))
            
            if raw_graph_dtype[1] == "float16":
                if old_graph_dtype[1] == "float64" or old_graph_dtype[1] == "float32":
                    if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
                        tensor_value = np.frombuffer(old_graph_node[node.name].tensor_content, dtype=old_graph_dtype[0])
                        tensor_value = tensor_value.astype(np.float16)
                        node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value, tf.float16, tensor_shape)))

                    else:
                        if old_graph_dtype[1] == "float64":
                            tensor_value = (np.array(old_graph_node[node.name].double_val)).astype(np.float16)
                            node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,tf.float16, [1])))

                        elif old_graph_dtype[1] == "float32":
                            tensor_value = (np.array(old_graph_node[node.name].float_val)).astype(np.float16)
                            node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,tf.float16, [1])))

                elif old_graph_dtype[1] == "float16":
                    tensor_value = convertMatrix(np.array(old_graph_node[node.name].half_val), tensor_shape)
                    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value, tf.float16, tensor_value.shape)))
            
            elif raw_graph_dtype[1] == "float64" or raw_graph_dtype[1] == "float32":
                if old_graph_dtype[1] == "float64" or old_graph_dtype[1] == "float32":
                    if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
                        tensor_value = np.frombuffer(old_graph_node[node.name].tensor_content,dtype = old_graph_dtype[0])
                        tensor_value = tensor_value.astype(dtype=raw_graph_dtype[0])
                        node.attr["value"].tensor.tensor_content = tensor_value.tostring()

                    else:
                        if old_graph_dtype[1] == "float64":
                            tensor_value = (np.array(old_graph_node[node.name].double_val)).astype(raw_graph_dtype[0])
                            node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,raw_graph_dtype[0], [1])))

                        elif old_graph_dtype[1] == "float32": 
                            tensor_value = (np.array(old_graph_node[node.name].float_val)).astype(raw_graph_dtype[0])
                            node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,raw_graph_dtype[0], [1])))
                
                elif old_graph_dtype[1] == "float16":
                    if (len(tensor_shape) != 1) or (tensor_shape[0] != 1):
                        tensor_value = convertMatrix(np.array(old_graph_node[node.name].half_val), tensor_shape)
                        tensor_value = tensor_value.astype(raw_graph_dtype[0])
                        node.attr["value"].tensor.tensor_content = tensor_value.tostring()
                    else:
                        tensor_value = convertMatrix(np.array(old_graph_node[node.name].half_val), tensor_shape)
                        tensor_value = tensor_value.astype(raw_graph_dtype[0])
                        node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(tensor_value,raw_graph_dtype[0], tensor_value.shape)))

    return raw_graph_def


class CopyNodeAttr:
    def __init__(self, node) -> None:
        self.node = node

    def from_array(
        self, tensor: np.ndarray, dtype: type, shape: Optional[Sequence[int]] = None
    ):
        if shape is None:
            shape = tensor.shape
        self.node.attr["value"].CopyFrom(
            tf.AttrValue(tensor=tf.make_tensor_proto(tensor, dtype, shape))
        )

    def from_str(self, tensor: np.ndarray):
        self.node.attr["value"].tensor.tensor_content = tensor.tostring()


def load_tensor(node: tf.Tensor, dtype_old: type, dtype_new: type) -> np.ndarray:
    if dtype_old == np.float64:
        tensor = np.array(node.double_val, dtype=dtype_new)
    elif dtype_old == np.float32:
        tensor = np.array(node.float_val, dtype=dtype_new)

    return tensor


def check_dim(raw_graph_node: tf.Tensor, old_graph_node: tf.Tensor, node_name: str):
    """Check if dimensions of tensor in old and new graph is equal.

    Parameters
    ----------
    raw_graph_node : tf.Tensor
        node of the receiving graph
    old_graph_node : tf.Tensor
        node of the graph from which will node be extracted
    node_name : str
        name of the node

    Raises
    ------
    RuntimeError
        if node dimension do not match
    """
    raw_graph_dim = raw_graph_node[node_name].tensor_shape
    old_graph_dim = old_graph_node[node_name].tensor_shape
    if raw_graph_dim != old_graph_dim:
        raise RuntimeError(
            f"old graph {old_graph_dim} and raw graph {raw_graph_dim} "
            f"has different {node_name} dim"
        )


def load_transform_node(graph: tf.Graph) -> Dict[str, tf.Tensor]:
    """Load nodes and their names from graph to dict.

    Parameters
    ----------
    graph : tf.Graph
        tensforflow graph

    Returns
    -------
    Dict[str, tf.Tensor]
        mapping on graph node names and corresponding tensors
    """
    transform_node_pattern = re.compile(
        r"filter_type_\d+/matrix_\d+_\d+|"
        r"filter_type_\d+/bias_\d+_\d+|"
        r"filter_type_\d+/idt_\d+_\d+|"
        r"layer_\d+_type_\d+/matrix|"
        r"layer_\d+_type_\d+/bias|"
        r"layer_\d+_type_\d+/idt|"
        r"final_layer_type_\d+/matrix|"
        r"descrpt_attr/t_avg|"
        r"descrpt_attr/t_std|"
        r"final_layer_type_\d+/bias|"
        r"fitting_attr/t_fparam_avg|"
        r"fitting_attr/t_fparam_istd|"
        r"fitting_attr/t_aparam_avg|"
        r"fitting_attr/t_aparam_istd|"
        r"model_attr/t_tab_info|"
        r"model_attr/t_tab_data|"
    )

    transform_node = {}
    for node in graph.node:
        if transform_node_pattern.fullmatch(node.name) is not None:
            transform_node[node.name] = node.attr["value"].tensor
    return transform_node
