# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging
import sys

import nni
from nni.networkmorphism_tuner.graph import json_to_graph

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torchvision
from torch.utils.data.distributed import DistributedSampler

from distributed_utils import dist_init, average_gradients, DistModule

import utils
import time

import zmq
from nni.env_vars import trial_env_vars
import json
import os

def get_args():
    """ get args from command line
    """
    parser = argparse.ArgumentParser("cifar10")
    parser.add_argument("--batch_size_per_gpu", type=int, default=128, help="batch size per gpu")
    parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer")
    parser.add_argument("--epochs", type=int, default=200, help="epoch limit")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
    parser.add_argument("--cutout_length", type=int, default=8, help="cutout length")
    parser.add_argument(
        "--model_path", type=str, default="./", help="Path to save the destination model"
    )
    parser.add_argument('--port', default='23456', type=str)
    parser.add_argument('-j', '--workers', default=2, type=int)
    return parser.parse_args()


def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    graph = json_to_graph(ir_model_json)
    logging.debug(graph.operation_history)
    model = graph.produce_torch_model()
    return model


def parse_rev_args(receive_msg):
    """ parse reveive msgs to global variable
    """
    global trainloader
    global testloader
    global trainsampler
    global testsampler
    global net
    global criterion
    global optimizer
    global rank, world_size

    # Loading Data
    if rank == 0:
        logger.debug("Preparing data..")

    transform_train, transform_test = utils.data_transforms_cifar10(args)

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    trainsampler = DistributedSampler(trainset)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=args.workers,
        pin_memory=False, sampler=trainsampler
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    testsampler = DistributedSampler(testset)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=0,
        pin_memory=False, sampler=testsampler
    )
    if rank == 0:
        print("len(trainset)=" + str(len(trainset)))
        print("len(testset)=" + str(len(testset)))

    # Model
    if rank == 0:
        logger.debug("Building model..")
    net = build_graph_from_json(receive_msg)

    net.cuda()
    net = DistModule(net)
    criterion = nn.CrossEntropyLoss()
    #nni.networkmorphism_tuner.graph.TorchModel does not support DataParallel
    #if device == "cuda" and torch.cuda.device_count() > 1:
    #    net = torch.nn.DataParallel(net)

    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
        )
    if args.optimizer == "Adadelta":
        optimizer = optim.Adadelta(net.parameters(), lr=args.learning_rate)
    if args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=args.learning_rate)
    if args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    if args.optimizer == "Adamax":
        optimizer = optim.Adamax(net.parameters(), lr=args.learning_rate)
    if args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate)

    cudnn.benchmark = True

    return 0


# Training
def train(epoch):
    """ train model on each epoch in trainset
    """

    global trainloader
    global net
    global criterion
    global optimizer
    global rank, world_size

    if rank == 0:
        logger.debug("Epoch: %d", epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = targets.cuda(async=True)
        #inputs, targets = inputs.to(device), targets.to(device)
        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(targets)

        optimizer.zero_grad()
        outputs = net(input_var)
        loss = criterion(outputs, target_var) / world_size

        loss.backward()
        average_gradients(net)
        optimizer.step()

        
        train_loss += loss.item()
        _, predicted = outputs.data.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #As the cost of all_reduce, we don't use all_reduce every batch to calculate acc."
        """
        if rank == 0:
            logger.debug(
                "Loss: %.3f | Acc: %.3f%% (%d/%d)",
                train_loss / (batch_idx + 1),
                100.0 * tmp_correct / tmp_total,
                tmp_correct,
                tmp_total,
            )
        """

    reduced_total = torch.Tensor([total])
    reduced_correct = torch.Tensor([correct])
    reduced_total = reduced_total.cuda()
    reduced_correct = reduced_correct.cuda()
    dist.all_reduce(reduced_total)
    dist.all_reduce(reduced_correct)

    tmp_total = int(reduced_total[0])
    tmp_correct = int(reduced_correct[0])
    acc = 100.0 * tmp_correct / tmp_total

    return acc


def test(epoch):
    """ eval model on each epoch in testset
    """
    global best_acc
    global testloader
    global net
    global criterion
    global optimizer
    global rank, world_size

    if rank == 0:
        logger.debug("Eval on epoch: %d", epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #targets = targets.cuda(async=True)
            #inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.cuda(), targets.cuda()
            #input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
            #target_var = torch.autograd.Variable(targets, volatile=True)
            #outputs = net(input_var)
            outputs = net(inputs)
            loss = criterion(outputs, targets) / world_size

            test_loss += loss.item()
            _, predicted = outputs.data.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #As the cost of all_reduce, we don't use all_reduce every batch to calculate acc."
            """
            if rank == 0:
                logger.debug(
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)",
                    test_loss / (batch_idx + 1),
                    100.0 * tmp_correct / tmp_total,
                    tmp_correct,
                    tmp_total,
                )"""

    reduced_total = torch.Tensor([total])
    reduced_correct = torch.Tensor([correct])
    reduced_total = reduced_total.cuda()
    reduced_correct = reduced_correct.cuda()
    dist.all_reduce(reduced_total)
    dist.all_reduce(reduced_correct)

    tmp_total = int(reduced_total[0])
    tmp_correct = int(reduced_correct[0])
    acc = 100.0 * tmp_correct / tmp_total
    if acc > best_acc:
        best_acc = acc
    return acc, best_acc

if __name__ == "__main__":
    # pylint: disable=W0603
    # set the logger format
    nni.init_trial()
    global logger
    logger = None

    global args
    args = get_args()

    global rank, world_size
    rank, world_size = dist_init(args.port)
    if rank == 0:
        # set the logger format
        log_format = "%(asctime)s %(message)s"
        logging.basicConfig(
            filename="networkmorphism.log",
            filemode="a",
            level=logging.INFO,
            format=log_format,
            datefmt="%m/%d %I:%M:%S %p",
        )
        logger = logging.getLogger("cifar10-network-morphism-pytorch")

    global trainloader, testloader, trainsampler, testsampler, net, criterion, optimizer, best_acc
    trainloader = None
    testloader = None
    trainsampler = None
    testsampler = None
    net = None
    criterion = None
    optimizer = None
    #device = "cuda:"+str(rank) if torch.cuda.is_available() else "cpu"
    best_acc = 0.0

    try:
        assert(args.workers % world_size == 0)
        args.workers = args.workers // world_size
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        RCV_CONFIG = ""
        real_model_file = os.path.join(trial_env_vars.NNI_SYS_DIR, "real_model.json")
        if rank == 0:    # only works for single node
            socket.connect("tcp://172.23.33.30:8081")
            # trial get next parameter from network morphism tuner
            RCV_CONFIG = nni.get_next_parameter(socket)
            with open(real_model_file, "w") as f:
                json.dump(RCV_CONFIG, f)
            logger.info(RCV_CONFIG)
        else:
            while not os.path.isfile(real_model_file):
                time.sleep(5)
            with open(real_model_file, "r") as f:
                RCV_CONFIG = json.load(f)

        parse_rev_args(RCV_CONFIG)
        train_acc = 0.0
        best_acc = 0.0
        early_stop = utils.EarlyStopping(mode="max")
        start_time = time.time()
        tmp_ep = 0
        for ep in range(args.epochs):
            trainsampler.set_epoch(ep)
            train_acc = train(ep)
            test_acc, best_acc = test(ep)
            if rank == 0:
                nni.report_intermediate_result(test_acc)
                logger.debug(test_acc)
            tmp_ep = ep
            if early_stop.step(test_acc):
                break
        if rank == 0:
            print("duration=" + str(time.time() - start_time))
            print("epoch=" + str(tmp_ep))
            print("best_acc=" + str(best_acc))

            # trial report best_acc to tuner
            nni.report_final_result(best_acc)
        dist.barrier()
    except Exception as exception:
        logger.exception(exception)
        raise
