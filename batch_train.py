#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, psutil, time, pathlib, contextlib
import numpy as np
import torch
import torch.multiprocessing as mp
import train

#mp.set_start_method('spawn')

def gpu_worker(itask, iworker, igpu, fileprefix, train_kwargs, result_queue):

    if fileprefix:
        filename = '{}-task{:03d}.out'.format(fileprefix,itask)
        filepath = pathlib.Path(filename).resolve()
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
        stdout_target = filepath.open('wt')
    else:
        stdout_target = sys.__stdout__

    with contextlib.redirect_stdout(stdout_target):

        currproc = psutil.Process()
        createtime = currproc.create_time()
        if torch.cuda.is_available():
            train_kwargs['device'] = 'cuda:{}'.format(igpu)
        else:
            train_kwargs['device'] = 'cpu'
        print('Process {} task {} on worker {} on GPU {} on CPU {}'.
            format(currproc.pid, itask, iworker, igpu, currproc.cpu_num()))
        result = train.train(**train_kwargs)
        delta_seconds = time.time() - createtime
        full_result = (itask, iworker, igpu, delta_seconds, result['accuracy'],
            result['final_epoch'], train_kwargs)
        result_queue.put(full_result)


def batch_training(fileprefix='', tasks=[]):

    if fileprefix:
        filename = '{}-main.out'.format(fileprefix)
        filepath = pathlib.Path(filename).resolve()
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
        stdout_target = filepath.open('wt')
    else:
        stdout_target = sys.__stdout__

    with contextlib.redirect_stdout(stdout_target):

        print('System-wide logical CPUs:', psutil.cpu_count())
        print('System-wide physical CPUs:', psutil.cpu_count(logical=False))
        if torch.cuda.is_available():
            oversubscribe = 2
            ngpus = torch.cuda.device_count()
            nworkers = ngpus * oversubscribe
        else:
            ngpus = 0
            nworkers = psutil.cpu_count() // 4
        curproc = psutil.Process()
        createtime = curproc.create_time()
        print('Main process {} on CPU {} with {} threads'.
            format(curproc.pid, curproc.cpu_num(), curproc.num_threads()))
        print('Presently available CPUs:', len(curproc.cpu_affinity()))
        print('Presently available GPUs:', ngpus)
        print('Worker processes:', nworkers)
        # load input tasks into queue
        task_queue = mp.SimpleQueue()
        for i,task in enumerate(tasks):
            print('Task',i+1,task)
            task_queue.put(task)
        # worker locks
        locks = []
        active_processes = []
        for i in range(nworkers):
            locks.append(mp.Lock())
            active_processes.append(None)
        # results queue
        result_queue = mp.SimpleQueue()
        itask = 0
        while not task_queue.empty():
            for ilock,lock in enumerate(locks):
                if lock.acquire(timeout=1):
                    # acquire lock and expect process == None
                    assert(active_processes[ilock] is None)
                    if task_queue.empty():
                        lock.release()
                        continue
                    train_kwargs = task_queue.get()
                    if ngpus:
                        igpu = ilock%ngpus
                    else:
                        igpu=0
                    args = (itask, ilock, igpu, fileprefix,
                            train_kwargs, result_queue)
                    p = mp.Process(target=gpu_worker, args=args)
                    #p = mp.Process()
                    print('  Launching task {}/{} on worker {} on GPU {}'.
                        format(itask, len(tasks), ilock, igpu))
                    itask += 1
                    p.start()
                    active_processes[ilock] = p
                else:
                    # locked and expect process != None
                    existing_process = active_processes[ilock]
                    assert(existing_process is not None)
                    if existing_process.exitcode is not None:
                        # process is complete; close and release
                        print('  Process {} finished'.format(existing_process.pid))
                        active_processes[ilock] = None
                        lock.release()
        print('Finished task loop')
        still_running = True
        while still_running:
            still_running = False
            for i,process in enumerate(active_processes):
                if process is None: continue
                if process.exitcode is None:
                    still_running = True
                    break
                else:
                    print('  Process {} finished'.format(process.pid))
                    active_processes[i] = None
            time.sleep(1)
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        print('Tasks:', len(tasks), 'results:', len(results))
        def sort_func(element):
            return element[0]
        results = sorted(results, key=sort_func)
        for i,result in enumerate(results):
            print('Task {:3d} worker/GPU {:2d}/{:1d}  dt {:5.1f}s  max/med acc {:5.1f}%/{:5.1f}%  kw: {}'.
                format(*result[0:4], result[4].max(), np.median(result[4]), result[6]))
        delta_seconds = time.time() - createtime
        print('Main execution: {:.1f} s'.format(delta_seconds))


def test_batch_training():
    tasks = [{'data':1,'epochs':1,'repeat':1} for _ in range(32)]
    batch_training(tasks=tasks)


def prepare_tasks():
    data = 1
    epochs = 16
    repeat = 10
    optims = [
              {'optim_name':'Adamax', 'optim_kwargs':{'lr':1e-3}},
              {'optim_name':'SGD', 'optim_kwargs':{'lr':1e-2}},
              {'optim_name':'ASGD', 'optim_kwargs':{'lr':1e-2}},
             ]
    activs = [
              {'activation':'relu'},
              {'activation':'leaky_relu', 'activ_kwargs':{'negative_slope':0.02}},
              {'activation':'tanh'},
             ]
    convs = [
             [3,9],
             [4,12],
             [5,15],
             [6,18],
            ]
    fcs = [
           [40,20],
           [40,30],
           [60,30],
           [60,40],
           [80,40],
           [80,60],
           [100,50],
           [100,80],
           [120,60],
           [120,80],
          ]
    tasks = []
    for optim in optims:
        for activ in activs:
            for conv_layers in convs:
                for fc_layers in fcs:
                    modelkw = {
                               'conv_layers':conv_layers,
                               'fc_layers':fc_layers,
                              }
                    modelkw.update(activ)
                    task_kw = {}
                    task_kw.update(optim)
                    task_kw.update ({
                                     'model_kwargs':modelkw,
                                     'data':data,
                                     'epochs':epochs,
                                     'repeat':repeat,
                                    })
                    tasks.append(task_kw)
    return tasks

if __name__=='__main__':
    mp.set_start_method('spawn')
    test_batch_training()
