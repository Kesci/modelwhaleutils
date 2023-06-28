class MlFlowRunNotFould(Exception):
    def __init__(self, m):
        self.message = m

    def __str__(self):
        return self.message


def save_tf_ckpt(sess, directory, filename):
    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename + '.ckpt')
    saver = tf.train.Saver()
    saver.save(sess, filepath)
    return filepath


def save_as_pb(sess, directory, filename, output_node):
    import tensorflow as tf
    from tensorflow.python.tools import freeze_graph
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save check point for graph frozen later
    ckpt_filepath = save_tf_ckpt(sess, directory=directory, filename=filename)
    pbtxt_filename = filename + '.pbtxt'
    pbtxt_filepath = os.path.join(directory, pbtxt_filename)
    pb_filepath = os.path.join(directory, filename + '.pb')
    # This will only save the graph but the variables will not be saved.
    # You have to freeze your model first.
    tf.train.write_graph(
        graph_or_graph_def=sess.graph_def,
        logdir=directory,
        name=pbtxt_filename,
        as_text=True)

    freeze_graph.freeze_graph(
        input_graph=pbtxt_filepath,
        input_saver='',
        input_binary=False,
        input_checkpoint=ckpt_filepath,
        output_node_names=output_node,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=pb_filepath,
        clear_devices=True,
        initializer_nodes='')

    return pb_filepath
