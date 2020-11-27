#time: 2020-11-28
#author: Yongjie

import tensorflow as tf
from tensorflow.python.platform import gfile

def get_ckpt_nodes(input_checkpoint, _nodes_path):
    """
    get ckpt model nodes
    :param input_checkpoint:    ckpt model 
    :param _nodes_path: output nodes path
    :return: 
    """
    print("load graph")
    tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    file = open(_nodes_path, 'a+')
    for n in tf.get_default_graph().as_graph_def().node:
        file.write(n.name + '\n')
    file.close()
    print("Success###")

def get_pb_nodes(input_checkpoint, _nodes_path):
    """
    
    get pb model nodes
    :param input_checkpoint: pb model
    :param _nodes_path: output nodes path
    :return: 
    """
    print("load graph")
    with gfile.FastGFile(input_checkpoint, 'rb') as f:

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        file = open(_nodes_path, 'a+')
        for i, n in enumerate(graph_def.node):
            # print("Name of the node - %s" % n.name)
            file.write(n.name + '\n')
        file.close()
    print("Success###")


if __name__=="__main__":

    #ckpt model
    input_path = "./ckpt/model_120000-120001"
    nodes_path = "nodes/nodes.txt"
    get_ckpt_nodes(input_path)

    #pb model
    pb_path = './pb/mobile_layer_24.pb'
    nodes_path = "./nodes/nodes_txt"
    get_pb_nodes(pb_path, nodes_path)