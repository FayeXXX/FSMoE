#! /usr/bin/python3
import time
import random
import subprocess
from manager import GPUManager


params = "--loss Softmax \
--use_default_parameters False \
--num_workers 32 \
--MAX_EPOCH 10 \
--transform cocoop \
--backbone ViT-B-32 \
--CTX_INIT '' \
--N_CTX 4 \
--method lowlayer2_ablation "


def get_mission_queue(cmd_command):

    mission_queue = []
    # layers = [[0, 1, 2, 3],
    #           [4, 5, 6, 7],
    #           [7, 8, 9, 10],
    #           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #           [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    LR = [0.002]  # [0.002, 0.0025, 0.0028, 0.003]
    batch_size = [64]  # [16, 32, 64, 128]
    split_idx = [0]
    dataset = ['cifar-10-10']
    for ds in dataset:
        for s in split_idx:
            for l in layers:
                for lr in LR:
                    for bs in batch_size:
                        cmd_params = (f" {params} "
                                      f"--dataset {ds} "
                                      f"--split_idx {s} "
                                      f"--layers {l} "
                                      f"--LR {lr} "
                                      f"--batch_size {bs} ")
                        # cmd = f"{cmd_command} {cmd_params} > /dev/null"
                        cmd = f"{cmd_command} {cmd_params} "
                        mission_queue.append(cmd)

    return mission_queue


gm = GPUManager()
cmd_command = 'python osr_lowlayer_ablation.py '
mission_queue = get_mission_queue(cmd_command)
total = len(mission_queue)
finished = 0
running = 0
p = []
min_gpu_number = 1  # 最小GPU数量，多于这个数值才会开始执行训练任务。
time_interval = 120  # 监控GPU状态的频率，单位秒。

while finished + running < total:
    localtime = time.asctime(time.localtime(time.time()))
    gpu_av = gm.choose_no_task_gpu()
    # gpu_av = [x for x in gpu_av if x != 1]
    # 在每轮epoch当中仅提交1个GPU计算任务
    if len(gpu_av) >= min_gpu_number:
        # 为了保证服务器上所有GPU负载均衡，从所有空闲GPU当中随机选择一个执行本轮次的计算任务
        gpu_index = random.sample(gpu_av, min_gpu_number)[:min_gpu_number]
        gpu_index_str = ','.join(map(str, gpu_index))

        cmd_ = 'CUDA_VISIBLE_DEVICES=' + gpu_index_str + ' ' + mission_queue.pop(0)  # mission_queue当中的任务采用先进先出优先级策略
        print(f'Mission : {cmd_}\nRUN ON GPU : {gpu_index_str}\nStarted @ {localtime}\n')
        # subprocess.call(cmd_, shell=True)
        p.append(subprocess.Popen(cmd_, shell=True))
        running += 1
        time.sleep(time_interval)  # 等待NVIDIA CUDA代码库初始化并启动
    else:  # 如果服务器上所有GPU都已经满载则不提交GPU计算任务
        pass
        # print(f'Keep Looking @ {localtime} \r')

    new_p = []  # 用来存储已经提交到GPU但是还没结束计算的进程
    for i in range(len(p)):
        if p[i].poll() != None:
            running -= 1
            finished += 1
        else:
            new_p.append(p[i])
    # if len(new_p) == len(p):  # 此时说明已提交GPU的进程队列当中没有进程被执行完
    #     time.sleep(time_interval)
    #     1
    p = new_p

for i in range(len(p)):  # mission_queue队列当中的所有GPU计算任务均已提交，等待GPU计算完毕结束主进程
    p[i].wait()

print('Mission Complete ! Checking GPU Process Over ! ')

