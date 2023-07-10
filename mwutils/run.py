from botocore.exceptions import ClientError
from random import randrange
from os import path, getpid
import numpy as np
import logging
import botocore
import boto3
import json
import signal
import traceback
import time
import jwt
import warnings
import requests
import os
from mwutils.sys_stat import SystemStats
from mwutils.logs import Logger, mili_time
import mwutils
import mlflow
from .utils import *


_STEP = 'step'
_EPOCH = 'epoch'
_BATCH = 'batch'
_LOSS = 'loss'
_ACC = 'acc'
_TIMESTAMP = 'timestamp'

_MAX_ACC = "max_accuracy"
_MIN_LOSS = "min_loss"
_BEST = "best"

MODEL_TYPE_TF = "tf"
MODEL_TYPE_KERAS = "keras"
MODEL_TYPE_TORCH = "torch"
MODEL_TYPE_CUSTOM = "custom"

run_names = {}


class Run():
    def __init__(self, name="", user_id="", lab_id="", org_id="", user_token="", use_mlflow=False, is_debug=False, debug_uid="", flush_interval_seconds=5,
                 sys_stat_sample_size=1, sys_stat_sample_interval=2, local_path='', write_logs_to_local=False,
                 remote_path='', buffer_all_logs=False):
        if use_mlflow == True:
            active_run = mlflow.active_run()
            if active_run is None:
                raise MlFlowRunNotFould("没有找到已创建的 mlflow run")
            else:
                self.use_mlflow = True
                self.mlflow_run = active_run
        else:
            self.use_mlflow = False
        if name == '':
            if self.use_mlflow == True and self.mlflow_run is not None:
                name = self.mlflow_run.info.run_name
            else:
                name = '数据科学实验@' + str(randrange(999))
        if name in run_names:
            s = "name {} is already used in current session.".format(name)
            raise Exception(s)
        p = os.path.expanduser('~')
        _path = p + '/.ide/config.json'
        _data = None
        config_user_id = None
        config_lab_id = None
        config_org_id = None
        config_token = None
        if os.path.exists(_path):
            f = open(_path)
            _data = json.load(f)
            f.close()
        if _data:
            _remote_path = _data['website']['siteUrl']
            config_user_id = _data['website']['user']['_id']
            config_lab_id = _data['website']['lab']['_id']
            config_org_id = _data['website']['org']['_id']
            config_token = _data['website']['token']
        run_names[name] = self
        self._loggers = {}
        self.custom_loggers = {}
        env_user_id = os.getenv("ENV_USER_ID")
        env_lab_id = os.getenv("ENV_LAB_ID")
        env_org_id = os.getenv("ENV_ORG_ID")
        env_token = os.getenv("ENV_TOKEN")

        if config_user_id:
            self.user_id = config_user_id
        elif env_user_id:
            self.user_id = env_user_id
        else:
            self.user_id = user_id
        if config_lab_id:
            self.lab_id = config_lab_id
        elif env_lab_id:
            self.lab_id = env_lab_id
        else:
            self.lab_id = lab_id

        if config_org_id:
            self.org_id = config_org_id
        elif env_org_id:
            self.org_id = env_org_id
        else:
            self.org_id = org_id

        # self.user_token = user_token
        if config_token:
            print('using config_token')
            self.user_token = config_token
        elif env_token:
            print('using env_token')
            self.user_token = env_token
        else:
            self.user_token = user_token

        if remote_path:
            self.remote_path = remote_path
        elif _remote_path:
            self.remote_path = _remote_path + '/api/runs'
        else:
            self.remote_path = 'https://www.heywhale.com/api/runs'

        print('api 地址: ', self.remote_path)
        timestr = str(mili_time())
        if not (self.user_id and self.lab_id and self.org_id):
            s = "At least one of required fields is empty:\nuser_id: {}\norg_id: {}\nlab_id: {}\n".format(
                user_id, org_id, lab_id)
            raise Exception(s)
        self.run_id = name + '_' + timestr
        self.flush_interval_seconds = max(5, flush_interval_seconds)
        self._sys_stat_sample_size = sys_stat_sample_size
        self._sys_stat_sample_interval_seconds = sys_stat_sample_interval
        self.local_path = local_path
        self.write_logs_to_local = write_logs_to_local
        self.logs_remote_path = self.remote_path + '/logs'
        self.conclude_remote_path = self.remote_path + '/conclude'
        self.abort_remote_path = self.remote_path + "/abort"
        self.buffer_all_logs = buffer_all_logs
        self.model_path = ""
        self.metadata = {"name": name, "user_id": self.user_id,
                         "lab_id": self.lab_id, "run_id": self.run_id, "org_id": self.org_id, "annotations": {"custom_keys": [], "keys": []}}
        self.pid = None
        self.started = False

        # INIT ML
        self.pid = getpid()
        train_path = path.join(
            self.local_path, "train.json") if self.write_logs_to_local else ''
        test_path = path.join(
            self.local_path, "test.json") if self.write_logs_to_local else ''
        val_path = path.join(
            self.local_path, "val.json") if self.write_logs_to_local else ''
        sys_path = path.join(
            self.local_path, "sys.json") if self.write_logs_to_local else ''
        self._loggers['train'] = MLLoger("train", sample_time_interval_seconds=self.flush_interval_seconds,
                                         metadata=self.metadata, local_path=train_path, post_addr=self.logs_remote_path,
                                         buffer_all=self.buffer_all_logs)
        self._loggers['test'] = MLLoger("test", sample_time_interval_seconds=self.flush_interval_seconds,
                                        metadata=self.metadata, local_path=test_path, post_addr=self.logs_remote_path,
                                        buffer_all=self.buffer_all_logs)
        self._loggers['val'] = MLLoger("val", sample_time_interval_seconds=self.flush_interval_seconds,
                                       metadata=self.metadata, local_path=val_path, post_addr=self.logs_remote_path,
                                       buffer_all=self.buffer_all_logs)
        self._loggers['system'] = CustomLogger("system", sample_time_interval_seconds=self.flush_interval_seconds,
                                               metadata=self.metadata, local_path=sys_path, post_addr=self.logs_remote_path,
                                               buffer_all=self.buffer_all_logs)
        self._loggers['meta'] = CustomLogger("meta", sample_time_interval_seconds=self.flush_interval_seconds,
                                             metadata=self.metadata, local_path=sys_path, post_addr=self.logs_remote_path,
                                             buffer_all=self.buffer_all_logs)
        print('logger class registered')
        # START ML
        self.started = True
        self.__register_signal_handlers()
        for _, logger in self._loggers.items():
            logger.start()
        for _, clogger in self.custom_loggers.items():
            clogger.start()
        self.sys_stat = SystemStats(self)
        self.sys_stat.start()

        print('logger started')
        # 创建一个 RUN
        _request_meta = {
            'metadata': {
                'name': name,
                'user_id': self.user_id,
                'run_id': self.run_id,
                'lab_id': self.lab_id,
                'org_id': self.org_id
            }
        }
        if is_debug == True:
            _addr = self.remote_path + '/linkMLFlow'
            _request_meta['use_mlflow'] = True
            _request_meta['is_debug'] = True
            _request_meta['mlflow_run'] = {'info': {'run_uuid': debug_uid}}
            create_run(_request_meta, _addr)
        else:
            if self.use_mlflow:
                _request_meta['is_debug'] = False
                _addr = self.remote_path + '/linkMLFlow'
                _request_meta['use_mlflow'] = True
                _request_meta['mlflow_run'] = self.mlflow_run
                create_run(_request_meta, _addr)
            else:
                _request_meta['use_mlflow'] = False
                _request_meta['is_debug'] = False
                create_run(_request_meta, self.logs_remote_path)

    def init_ml(self):
        # if self.pid:
        #     return
        self.pid = getpid()
        train_path = path.join(
            self.local_path, "train.json") if self.write_logs_to_local else ''
        test_path = path.join(
            self.local_path, "test.json") if self.write_logs_to_local else ''
        val_path = path.join(
            self.local_path, "val.json") if self.write_logs_to_local else ''
        sys_path = path.join(
            self.local_path, "sys.json") if self.write_logs_to_local else ''
        self._loggers['train'] = MLLoger("train", sample_time_interval_seconds=self.flush_interval_seconds,
                                         metadata=self.metadata, local_path=train_path, post_addr=self.logs_remote_path,
                                         buffer_all=self.buffer_all_logs)
        self._loggers['test'] = MLLoger("test", sample_time_interval_seconds=self.flush_interval_seconds,
                                        metadata=self.metadata, local_path=test_path, post_addr=self.logs_remote_path,
                                        buffer_all=self.buffer_all_logs)
        self._loggers['val'] = MLLoger("val", sample_time_interval_seconds=self.flush_interval_seconds,
                                       metadata=self.metadata, local_path=val_path, post_addr=self.logs_remote_path,
                                       buffer_all=self.buffer_all_logs)
        self._loggers['system'] = CustomLogger("system", sample_time_interval_seconds=self.flush_interval_seconds,
                                               metadata=self.metadata, local_path=sys_path, post_addr=self.logs_remote_path,
                                               buffer_all=self.buffer_all_logs)
        self._loggers['meta'] = CustomLogger("meta", sample_time_interval_seconds=self.flush_interval_seconds,
                                             metadata=self.metadata, local_path=sys_path, post_addr=self.logs_remote_path,
                                             buffer_all=self.buffer_all_logs)

    def start_ml(self):
        if self.started:
            return
        self.started = True
        self.__register_signal_handlers()
        for _, logger in self._loggers.items():
            logger.start()
        for _, clogger in self.custom_loggers.items():
            clogger.start()
        self.sys_stat = SystemStats(self)
        self.sys_stat.start()

    def log_meta(self, data):
        self._loggers['meta'].log(data)

    def log_ml(self, step=None, epoch=None, batch=None, loss=None, acc=None, phase="train", custom_logs=None):
        # phase is the same thing with namea
        if acc is not None:
            try:
                acc = float(acc)
            except:
                raise TypeError('acc cannot be transferred to float!')

        if loss is not None:
            try:
                loss = float(loss)
            except:
                raise TypeError('loss cannot be transferred to float!')

        self._loggers[phase].log(step=step, epoch=epoch,
                                 batch=batch, loss=loss, acc=acc, custom_logs=custom_logs)

    def new_custom_logger(self, name, local_path=''):
        self.custom_loggers[name] = CustomLogger(name, sample_time_interval_seconds=self.flush_interval_seconds,
                                                 metadata=self.metadata, local_path=local_path, post_addr=self.logs_remote_path,
                                                 buffer_all=self.buffer_all_logs)

    def add_memoize_funcs_to_logger(self, name, funcs):
        self._loggers[name].add_memoize_funcs(funcs)

    def set_tf_model(self, model):
        self.model = model
        self.model_type = MODEL_TYPE_TF

    def _save_tf_model(self, model_path):
        # SavedModel
        # tf2
        import tensorflow as tf
        tf.saved_model.save(self.model, model_path)
        pass

    def set_keras_model(self, model):
        self.model = model
        self.model_type = MODEL_TYPE_KERAS

    def _save_keras_model(self, model_path):
        # SavedModel
        # tf2
        import tensorflow as tf
        tf.keras.models.save_model(self.model, model_path)
        self.model_path = model_path

    def set_torch_model(self, model):
        self.model = model
        self.model_type = MODEL_TYPE_TORCH

    def _save_torch_model(self, model_path):
        # torch version >= 1.6
        import torch
        torch.save(self.model, model_path)
        self.model_path = model_path
        pass

    def set_custom_model(self, path):
        self.model_type = MODEL_TYPE_CUSTOM
        self.model_path = path

    def _save_model(self, model_path):
        if hasattr(self, "model_type"):
            if self.model_type == MODEL_TYPE_TORCH:
                self._save_torch_model(model_path)
            elif self.model_type == MODEL_TYPE_KERAS:
                self._save_keras_model(model_path)
            elif self.model_type == MODEL_TYPE_TF:
                self._save_tf_model(model_path)

    def __upload_model(self):
        pass

    def __register_signal_handlers(self):
        signal.signal(signal.SIGINT, self.__sigint_handler)
        signal.signal(signal.SIGTERM, self.__sigterm_handler)

    def __sigint_handler(self, signum, frame):
        self.__abort_run("SIGINT", "[SIGINT]Terminated by system")
        traceback.print_stack(f=frame)
        raise RuntimeError("terminated by system")

    def __sigterm_handler(self, signum, frame):
        self.__abort_run("SIGTERM", "[SIGTERM]Terminated by user")
        traceback.print_stack(f=frame)
        raise KeyboardInterrupt("termniated by user")

    def __abort_run(self, sig, reason):
        if self.remote_path:
            tp = int(time.time())
            json_struct = {"metadata": self.metadata,
                           "timestamp": tp, "signal": sig, "reason": reason}
            for _ in range(3):
                r = requests.post(self.abort_remote_path, json=json_struct, headers={"Authorization": jwt.encode(
                    {"whatever": "1"}, "857851b2-c28c-4d94-83c8-f607b50ccd03")})
                if r.status_code >= 400:
                    # something wrong
                    jb = ''
                    try:
                        jb = r.json()
                    except:
                        pass
                    print("resp:", r)
                    msg = "code: {}, resp.json: {}, resp.text: {}".format(
                        r.status_code, jb, r.text)
                    print(msg)
                    warnings.warn(msg)
                else:
                    print("abort remote call succeed. resp:", r)
                    break
        self.started = False
        self.run_id = "aborted"

    def conclude(self, show_memoize=True, save_model=False, model_path="./saved_model", target=None, output_node=None, use_jit=False):
        if not self.started:
            pass
        for _, logger in self._loggers.items():
            logger.cancel()
            if show_memoize and logger.memoize:
                print(logger.name, logger.memoize)
        for _, clogger in self.custom_loggers.items():
            clogger.cancel()
            if show_memoize and clogger.memoize:
                print(clogger.name, clogger.memoize)
        prefixes = []
        if save_model == True:
            if self.user_token == '':
                print('token not specified, skipping')
            else:
                class_type = str(type(target))
                epoch_time = int(time.time())
                os.mkdir(str(epoch_time))
                _path = str(epoch_time)
                if target == None:
                    print('no model specified, skipping')
                    pass
                else:
                    if 'keras' in class_type:
                        _save_path = _path + '/saved_model.pb'
                        print('Keras Model detected, saving to ' + _save_path)
                        target.save(_save_path)
                        pass
                    if 'tensorflow' in class_type and 'keras' not in class_type:
                        import tensorflow as tf
                        if target._closed:
                            print(
                                'session closed, please run conclude() function in session')
                            return
                        else:
                            # tf 1
                            _save_path = _path + '/saved_model'
                            print(
                                'Tensorflow Model detected, saving to ' + _save_path)
                            # saver = tf.train.Saver()
                            # saver.save(target, _save_path)
                            save_as_pb(target, _path,
                                       'saved_model', output_node)
                            pass
                    elif 'tensorflow' not in class_type and 'keras' not in class_type:
                        try:
                            # torch
                            _save_path = _path + '/saved_model.pth'
                            if use_jit:
                                print(
                                    'TorchScript Model detected, saving to ' + _save_path)
                                target.save(_save_path)
                            else:
                                print(
                                    'Torch Model detected, saving to ' + _save_path)
                                import torch
                                torch.save(target.state_dict(), _save_path)
                                pass
                        except:
                            print('模型文件无法保存，请检查格式。')
                            pass

                path_artifact = '/api/dataset-upload-token?subType=artifact'
                endpoint_get_token = self.remote_path.replace(
                    '/api/runs', path_artifact) + '&token=' + self.user_token
                r = requests.get(endpoint_get_token)
                oss_config = json.loads(r.text)
                AK = oss_config['accessKeyId']
                SK = oss_config['secretAccessKey']
                region = oss_config['region']
                Session = oss_config['sessionToken']
                bucket = oss_config['bucket']
                epoch_time = int(time.time())

                s3_client = boto3.client('s3',
                                         region_name=region,
                                         aws_access_key_id=AK,
                                         aws_secret_access_key=SK,
                                         aws_session_token=Session
                                         )
                if oss_config["host"] != "":
                    s3_client = boto3.client('s3',
                                             endpoint_url=oss_config["host"],
                                             region_name=region,
                                             aws_access_key_id=AK,
                                             aws_secret_access_key=SK,
                                             aws_session_token=Session
                                             )

                upload_dir = _path
                for subdir, dirs, files in os.walk(upload_dir):
                    for file in files:
                        fullpath = os.path.join(subdir, file)
                        try:
                            object_name = oss_config['prefixToSave'] + \
                                str(epoch_time) + '/' + file
                            print('uploading file: ', fullpath)
                            response = s3_client.upload_file(
                                fullpath, bucket, object_name)
                            prefixes.append(object_name)
                        except ClientError as e:
                            logging.error(e)
                            print('Error uploading file ', file)
                            pass
        if self.remote_path:
            tp = int(time.time())
            json_struct = {
                "metadata": self.metadata,
                "best": [{"phase": name, "val": logger.memoize, _TIMESTAMP: tp} for name, logger in self._loggers.items()],
            }
            if len(prefixes) > 0:
                json_struct['files'] = prefixes
            for _ in range(3):
                r = requests.post(self.conclude_remote_path, json=json_struct, headers={"Authorization": jwt.encode(
                    {"whatever": "1"}, "857851b2-c28c-4d94-83c8-f607b50ccd03")})
                if r.status_code >= 400:
                    # something wrong
                    jb = ''
                    try:
                        jb = r.json()
                    except:
                        pass
                    print("resp:", r)
                    msg = "code: {}, resp.json: {}, resp.text: {}".format(
                        r.status_code, jb, r.text)
                    print(msg)
                    warnings.warn(msg)
                else:
                    print("conclude remote call succeed. resp:", r)
                    break
        # if upload_model:
        #     self.__upload_model()
        if self.use_mlflow:
            mlflow.end_run()
        self.started = False
        self.run_id = "concluded"
        print('记录已结束')
