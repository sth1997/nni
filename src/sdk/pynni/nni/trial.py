# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================


import json_tricks

from .env_vars import trial_env_vars
from . import platform

import zmq
from nni.networkmorphism_tuner.bayesian import BayesianOptimizer
from nni.networkmorphism_tuner.networkmorphism_tuner import NetworkMorphismTuner

import time

__all__ = [
    'get_next_parameter',
    'get_current_parameter',
    'report_intermediate_result',
    'report_final_result',
    'get_experiment_id',
    'get_trial_id',
    'get_sequence_id'
]


_params = None
_experiment_id = platform.get_experiment_id()
_trial_id = platform.get_trial_id()
_sequence_id = platform.get_sequence_id()


def get_next_parameter(socket):
    """Returns a set of (hyper-)paremeters generated by Tuner."""
    global _params
    father_id = -1

    _params = platform.get_next_parameter()
    if _params is None:
        return None

    socket.send_pyobj({"type": "get_next_parameter"})
    message = socket.recv_pyobj()
    tuner = message["tuner"]
    
    if tuner.history:
        parameter_id = int(get_sequence_id())
        t1 = time.time()
        json_out, father_id = tuner.generate_parameters(parameter_id)
        t2 = time.time()
        print("Generated time = " + str(t2 - t1))
        _params['parameters'] = json_out
        socket.send_pyobj({"type": "generated_parameter", "parameters": _params['parameters'], "father_id": father_id, "parameter_id": parameter_id})
        message = socket.recv_pyobj()
    else:
        socket.send_pyobj({"type": "generated_parameter"})
        message = socket.recv_pyobj()
        
    return _params['parameters']

def get_current_parameter(tag=None):
    global _params
    if _params is None:
        return None
    if tag is None:
        return _params['parameters']
    return _params['parameters'][tag]

def get_experiment_id():
    return _experiment_id

def get_trial_id():
    return _trial_id

def get_sequence_id():
    return _sequence_id

_intermediate_seq = 0

def report_intermediate_result(metric):
    """Reports intermediate result to Assessor.
    metric: serializable object.
    """
    global _intermediate_seq
    assert _params is not None, 'nni.get_next_parameter() needs to be called before report_intermediate_result'
    metric = json_tricks.dumps({
        'parameter_id': _params['parameter_id'],
        'trial_job_id': trial_env_vars.NNI_TRIAL_JOB_ID,
        'type': 'PERIODICAL',
        'sequence': _intermediate_seq,
        'value': metric
    })
    _intermediate_seq += 1
    platform.send_metric(metric)


def report_final_result(metric):
    """Reports final result to tuner.
    metric: serializable object.
    """
    assert _params is not None, 'nni.get_next_parameter() needs to be called before report_final_result'
    metric = json_tricks.dumps({
        'parameter_id': _params['parameter_id'],
        'trial_job_id': trial_env_vars.NNI_TRIAL_JOB_ID,
        'type': 'FINAL',
        'sequence': 0,  # TODO: may be unnecessary
        'value': metric
    })
    platform.send_metric(metric)
