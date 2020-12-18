#time: 2020-12-18
#author: Yongjie

import tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(ckpt, output_graph, output_node_names):
    """
    
    :param ckpt: ckpt_path 
    :param output_graph: export pb Path
    :param output_node_names: export nodes name
    :return: 
    """
    # saver = tf.train.import_meta_graph(ckpt+'.meta', clear_devices=True)
    saver = tf.compat.v1.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(',')
        )
        # print(output_graph_def)
        with tf.gfile.GFile(output_graph, 'wb') as fw:
            fw.write(output_graph_def.SerializeToString())
        print('{} ops in the final graph.'.format(len(output_graph_def.node)))

def restore_and_save(checkpoint_file, export_path):
    """
    
    :param checkpoint_file: ckpt path 
    :param export_path: export pb path
    :return: 
    """

    # checkpoint_file = tf.train.latest_checkpoint(input_checkpoint)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():

            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            print(graph.get_name_scope())
            # for node in graph.as_graph_def().node:
            #     print(node.name)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            input_ids = tf.saved_model.utils.build_tensor_info(
                graph.get_operation_by_name("input_ids").outputs[0])
            input_mask = tf.saved_model.utils.build_tensor_info(
                graph.get_operation_by_name("input_mask").outputs[0])
            segment_ids = tf.saved_model.utils.build_tensor_info(
                graph.get_operation_by_name("segment_ids").outputs[0])
            dropout = tf.saved_model.utils.build_tensor_info(
                graph.get_operation_by_name("dropout").outputs[0])
            # is_training = tf.saved_model.utils.build_tensor_info(
            #     graph.get_operation_by_name("is_training").outputs[0])
            predict = tf.saved_model.utils.build_tensor_info(
                graph.get_operation_by_name("predict/pre").outputs[0])

            # 定义模型的输入输出，建立调用接口与tensor签名之间的映射
            labeling_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        'input_ids:0': input_ids,
                        'input_mask:0': input_mask,
                        'segment_ids:0': segment_ids,
                        # 'is_training:0': is_training,
                        'dropout:0': dropout,
                    },
                    outputs={
                        "predict": predict,

                    },
                    method_name="tensorflow/serving/predict"
                ))

            # 建立模型名称与模型签名之间的映射
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        labeling_signature
                })

            builder.save()
            print("Build Done")


if __name__=="__main__":

    #for inference
    ckpt_path = "./ckpt/ner.ckpt-3000"
    pb_path = './pb/lstm_crf.pb'
    output_node_name = 'predict/pre'  # output nodes
    freeze_graph(ckpt_path, pb_path, output_node_name)

    #for tensorflow-serving deployment
    # ckpt_path = "./ckpt/ner.ckpt-3000"
    # export_p = "./pb/serving"
    # restore_and_save(ckpt_path, export_p)
