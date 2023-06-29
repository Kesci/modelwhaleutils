import threading
import time
import os
import requests
import warnings
import json
import atexit
import jwt
from .logs import *


def create_run(payload, post_addr):
    print("request address", post_addr)
    json_struct = {"metadata": payload['metadata']}
    if payload['is_debug'] == True:
        json_struct['mlflow_run_id'] = payload['mlflow_run']['info']['run_uuid']
    else:
        if payload['use_mlflow'] == True:
            json_struct['mlflow_run_id'] = payload['mlflow_run'].info.run_uuid
        else:
            json_struct['init'] = True
    for _ in range(3):
        r = requests.post(post_addr, json=json_struct, headers={"Authorization": jwt.encode(
            {"whatever": "1"}, "857851b2-c28c-4d94-83c8-f607b50ccd03")})
        if r.status_code >= 400:
            # something wrong
            errorMsg = ''
            try:
                errorMsg = r.json()
            except:
                pass
            print("resp:", r)
            warnings.warn("code: {}, resp.json: {}, resp.text: {}".format(
                r.status_code, errorMsg, r.text))
        else:
            print("modelwhale run 生成成功")
            return True
    return False


class MLLoger(Logger):
    def log(self, step=None, epoch=None, batch=None, loss=None, acc=None, custom_logs=None):
        val = dict()
        val['_TIMESTAMP'] = int(time.time())
        if step is not None:
            val['_STEP'] = step + 1
        if epoch is not None:
            val['_EPOCH'] = epoch + 1
        if batch is not None:
            val['_BATCH'] = batch + 1
        if loss is not None:
            val['_LOSS'] = loss
        if acc is not None:
            val['_ACC'] = acc

        if acc:
            if '_MAX_ACC' not in self.memoize and acc:
                self.memoize['_MAX_ACC'] = val
            elif self.memoize['_MAX_ACC']['_ACC'] < val['_ACC']:
                self.memoize['_MAX_ACC'] = val

        if loss:
            best = False
            if '_MIN_LOSS' not in self.memoize and loss:
                self.memoize['_MIN_LOSS'] = val
                best = True
            elif self.memoize['_MIN_LOSS']['_LOSS'] > val['_LOSS']:
                self.memoize['_MIN_LOSS'] = val
                best = True
            if best:
                if step is not None:
                    self.memoize["{}_{}".format('_BEST', '_STEP')] = step+1
                elif epoch is not None:
                    self.memoize["{}_{}".format('_BEST', '_EPOCH')] = epoch+1
                elif batch is not None:
                    self.memoize["{}_{}".format('_BEST', '_BATCH')] = batch+1
        if custom_logs:
            if isinstance(custom_logs, dict):
                for k, v in custom_logs.items():
                    if k not in ['loss', 'acc', 'accuracy', 'val_loss', 'val_acc', 'val_accuracy']:
                        if 'custom_keys' not in self.metadata['annotations']:
                            self.metadata['annotations']['custom_keys'] = []
                        if k not in self.metadata['annotations']['custom_keys']:
                            self.metadata['annotations']['custom_keys'].append(
                                k)
                        val[k] = v
        for k, _ in val.items():
            if k not in self.metadata['annotations']['keys']:
                self.metadata['annotations']['keys'].append(k)
        super().log(val)


class CustomLogger(Logger):
    pass


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
