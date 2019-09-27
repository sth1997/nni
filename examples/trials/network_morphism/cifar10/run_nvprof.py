import argparse
import os
import subprocess
import torch

from nni.networkmorphism_tuner.graph import json_to_graph

def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    graph = json_to_graph(ir_model_json)
    model = graph.produce_torch_model()
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="Experiment directory. Like \"~/nni/experiments/ABC\".")
    parser.add_argument("output_log", type=str, help="Path to save the result log from this script.")
    parser.add_argument("batch_size", type=int, help="Batch size for training and testing.")
    return parser.parse_args()

def analysis_result(result):
    global args
    FLOPs = 0.0
    for line in result:
        l = bytes.decode(line).split()
        if l[1] == "flop_count_sp":
            FLOPs += int(l[0]) * float(l[-1]) # Invocations * Avg
    return (FLOPs / args.batch_size)

if __name__ == "__main__":
    train_set_num = 50000
    test_set_num = 10000
    args = get_args()
    id_epoch_dict = {}
    id_duration_dict = {}
    id_flops_dict = {}
    id_parameters_dict = {}
    success_trials = 0
    error_trials = 0
    result_file = open(args.output_log, "w")
    for trials in os.listdir(args.experiment_path + "/trials"):
        id = -1
        with open(args.experiment_path + "/trials/" + trials + "/run.sh", "r") as file_read:
            for line in file_read.readlines():
                line = line.strip('\n')
                l = line.split()
                if l[0] == "export":
                    tmp_l = l[1].split("=")
                    if tmp_l[0] == "NNI_TRIAL_SEQ_ID":
                        id = int(tmp_l[1])
                        break
        if id == -1:
            result_file.write("ERROR: trial " + trials + " has no id.")
            result_file.close()
            exit()

        epoch = -1
        duration = -1.0
        with open(args.experiment_path + "/trials/" + trials + "/trial.log", "r") as file_read:
            for line in file_read.readlines():
                line = line.strip('\n')
                l = line.split()
                if "ERROR" in l:
                    epoch = 0 # If one model caused an error when training, we don't calculate it's FLOPs by setting its epoch to 0.
                tmp_l = l[-1].split("=")
                if tmp_l[0] == "epoch":
                    epoch = int(tmp_l[1])
                if tmp_l[0] == "duration":
                    duration = float(tmp_l[1])
        if epoch == -1:
            #result_file.write("ERROR: trial " + trials + " has no \"epoch\" in trial.log.")
            #result_file.close()
            #exit()
            epoch = 0
        if epoch == 0 or duration < 0:
            error_trials += 1
        if id in id_epoch_dict or id in id_duration_dict:
            result_file.write("ERROR: trial " + trials + "'s id has already in dict.")
            result_file.close()
            exit()
        id_epoch_dict[id] = epoch
        id_duration_dict[id] = duration

        if epoch > 0 and duration >= 0:
            success_trials += 1

    result_file.write("Number of success trials = " + str(success_trials) + "\n")
    result_file.write("Number of error trials = " + str(error_trials) + "\n")
    total_trials = success_trials + error_trials


    total_train_FLOPs = 0.0
    total_test_FLOPs = 0.0
    calc_trials = 0
    for model_json_file_name in os.listdir(args.experiment_path + "/log/model_path"):
        model_json_file_path = os.path.join(args.experiment_path + "/log/model_path", model_json_file_name)
        if model_json_file_name.endswith(".json") and os.path.isfile(model_json_file_path):
            id = int(model_json_file_name.split(".")[0])
            with open(model_json_file_path) as json_file:
                json_net = json_file.readline()
                pytorch_net = build_graph_from_json(json_net)
                id_parameters_dict[id] = sum(param.numel() for param in pytorch_net.parameters())
            continue

            '''if (not id in id_epoch_dict) or (id_epoch_dict[id] == 0):
                continue
            print(model_json_file_name + "  begin")
            epoch = id_epoch_dict[id]
            p = subprocess.Popen("CUDA_VISIBLE_DEVICES=0  nvprof --metrics flop_count_sp python3 cifar10_pytorch_for_nvprof.py train " + model_json_file_path, shell=True, stderr=subprocess.PIPE) #the output of nvprof is in stderr, and the output of python file is in stdout
            p.wait()
            result = p.stderr
            train_FLOPs = analysis_result(result) * train_set_num * epoch
            total_train_FLOPs += train_FLOPs
            p = subprocess.Popen("CUDA_VISIBLE_DEVICES=0  nvprof --metrics flop_count_sp python3 cifar10_pytorch_for_nvprof.py test " + model_json_file_path, shell=True, stderr=subprocess.PIPE)
            p.wait()
            result = p.stderr
            test_FLOPs = analysis_result(result) * test_set_num * epoch
            total_test_FLOPs += test_FLOPs
            result_file.write(model_json_file_name + " train_FLOPs=" + str(train_FLOPs) + " test_FLOPs=" + str(test_FLOPs) + "\n")
            result_file.flush()
            calc_trials += 1
            id_flops_dict[id] = (train_FLOPs + test_FLOPs) / id_duration_dict[id]
			'''
				
        
    result_file.write("Number of trials with FLOPs = " + str(calc_trials) + "\n")
    result_file.write("total_train_FLOPs=" + str(total_train_FLOPs) + " total_test_FLOPs=" + str(total_test_FLOPs) + "\n")
    result_file.write("total_FLOPs=" + str(total_train_FLOPs + total_test_FLOPs) + "\n")
    result_file.close()

    experiment_name = args.experiment_path.strip("/").split("/")[-1]
    #duration_file =  open(experiment_name + "_duration.csv", "w")
    #flops_file = open(experiment_name + "_FLOPS.csv", "w")
    parameters_file = open(experiment_name + "_parameters.csv", "w")
    for i in range(total_trials):
        parameters_file.write(str(id_parameters_dict[i]) + ",")
        '''if id_duration_dict[i] >= 0:
            duration_file.write(str(id_duration_dict[i]) + ",")
            flops_file.write(str(id_flops_dict[i]) + ",")
        else:
            duration_file.write(",")
            flops_file.write(",")
    duration_file.close()
    flops_file.close()'''
    parameters_file.close()
	



