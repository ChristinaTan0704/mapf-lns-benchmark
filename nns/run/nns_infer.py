import os
import datetime
import argparse
import collections
import re
from multiprocessing import Pool
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import subprocess
import time
import shutil
from collections import deque
from collections import OrderedDict
from utils import *
from nns.model import NNS
import torch 

def run_pbs_NNS_inference_with_openCpp(args):
    
    def gen_nns_input(state, removal_sets, get_subset_only=False):
        nonlocal args
        nonlocal map_w
        nonlocal map_h
        
        if args.replan_solver == "pbs":
            pad_nb_size = args.neighborSize
        else:
            pad_nb_size = 32
        
        subagents_idx = np.zeros((args.num_subset, pad_nb_size)) - 1
        
        valid_subset_idx = 0
        nns_input_subsets = []
        for one_removal_set in removal_sets:
            if len(one_removal_set)  > pad_nb_size or len(one_removal_set) <= 0:
                print("subset size", len(one_removal_set))
            else: 
                subagents_idx[valid_subset_idx] = np.pad(one_removal_set, (0, pad_nb_size - len(one_removal_set)), constant_values=-1) # np.array(one_removal_set)
                valid_subset_idx += 1
                nns_input_subsets.append(one_removal_set)
                
        remain_num = args.num_subset - valid_subset_idx
        upsample_idx = np.random.choice(valid_subset_idx, remain_num, replace=True)
        
        
        subagents_idx[valid_subset_idx:] = subagents_idx[upsample_idx]
        nns_input_subsets += [nns_input_subsets[i] for i in upsample_idx]
        subset_mask = subagents_idx == -1
        
        if get_subset_only:
            return nns_input_subsets
        else:
            curr_paths, agent_paths, agent_paths_mask = get_idx_from_agentPaths(state, spatial_pool=spatial_pool, temporal_pool=temporal_pool, cutoff=cutoff, map_w=map_w, map_h=map_h)
            return curr_paths, agent_paths, agent_paths_mask, subagents_idx, subset_mask, nns_input_subsets
        
    
    def nns_prediction(model, device, curr_paths, agent_paths, agent_paths_mask, shortest_path, obstacle, subagents_idx, subset_mask):
        curr_paths = torch.tensor(curr_paths, dtype=torch.long).unsqueeze(0)
        agent_paths = torch.tensor(agent_paths, dtype=torch.long).unsqueeze(0)
        agent_paths_mask = torch.tensor(agent_paths_mask, dtype=torch.bool).unsqueeze(0)
        shortest_path = torch.tensor(shortest_path, dtype=torch.long).unsqueeze(0)
        obstacle = torch.tensor(obstacle, dtype=torch.long).unsqueeze(0)
        subagents_idx = torch.tensor(subagents_idx, dtype=torch.long).unsqueeze(0)
        subset_mask = torch.tensor(subset_mask, dtype=torch.bool).unsqueeze(0)
        
        batch_data = (curr_paths, agent_paths, agent_paths_mask, shortest_path, obstacle, subagents_idx, subset_mask)
        batch_data = [item.to(device) for item in batch_data]
        infer_start = datetime.datetime.now()
        prediction = model(batch_data)
        infer_time = (datetime.datetime.now() - infer_start).total_seconds()
        
        return prediction, infer_time
    
    def update_state_based_on_nns_prediction(prediction, curr_state, removal_sets, agent_selection_counts, pbs_exe_path, map_path, agent_num, curr_state_path, count_decay, it, last_accept_it, pbs_replan_time_limit=20):
        sub_selection_counts = np.sum(np.take(agent_selection_counts, np.array(removal_sets)), axis=1)
        keep_count_threshold = np.random.choice(sub_selection_counts)
        keep_mask = sub_selection_counts < keep_count_threshold
        
        
        prediction[~keep_mask] = -np.inf
        selected_subset_index = np.argmax(prediction)
        
        improvement = 0
        succ, _, replan_time, improvement, _, _, agnet_paths = pbs_replan(pbs_exe_path, removal_sets[selected_subset_index], map_path, agent_num, curr_state_path)
        
        if improvement > 0 and succ and replan_time <= pbs_replan_time_limit:
            for agent_id in agnet_paths:
                curr_state[agent_id] = agnet_paths[agent_id].copy()
            if last_accept_it != -1:
                count_decay = min(count_decay, 1 / (it - last_accept_it))
            last_accept_it = it
            agent_selection_counts[np.array(removal_sets[selected_subset_index])] += 1
        if replan_time > pbs_replan_time_limit:
            replan_time = pbs_replan_time_limit
        
        return curr_state, replan_time, last_accept_it, agent_selection_counts, count_decay
            
    map_setting = {"empty-32-32": {"spatial_pool":1, "temporal_pool":1, "cutoff":48},
                   "random-32-32-20": {"spatial_pool":1, "temporal_pool":1, "cutoff":48},
                   "ost003d": {"spatial_pool":4, "temporal_pool":4, "cutoff":192},
                   "warehouse-10-20-10-2-1": {"spatial_pool":1, "temporal_pool":2, "cutoff":96},
                   "den520d": {"spatial_pool":4, "temporal_pool":4, "cutoff":216},
                   "Paris_1_256": {"spatial_pool":4, "temporal_pool":4, "cutoff":216}}
    
    # check the input parameters
    assert args.output_folder != "", "output_folder should be provided"
    assert os.path.exists(args.pbs_replan_exe), "pbs_replan_exe {} does not exist".format(args.pbs_replan_exe)
    
    # make the output folder
    initial_state_path = args.initial_state
    pbs_log_path = None
    
    # process one state at a time
    if args.log_path != "" and ".log" in args.log_path:
        pbs_log_path = args.log_path
        log_folder = os.path.dirname(pbs_log_path)
        os.makedirs(log_folder, exist_ok=True)
        if os.path.exists(pbs_log_path):
            os.remove(pbs_log_path)

    # initialization
    iteration = 0
    file_name = os.path.basename(initial_state_path)
    map_name = re.findall(r'map-(.*?)-scene', file_name)[0]
    scene_num = re.findall(r'scene-(.*?)-agent', file_name)[0]
    agent_num = re.findall(r'agent-(.*?).json', file_name)[0]
    map_path = os.path.join(args.map_folder, map_name + ".map")
    curr_state_path = initial_state_path
    iterInfo = {"tabu_list": []}
    curr_state = read_json(curr_state_path)
    curr_state = {int(k): v for k, v in curr_state.items()}
    runtime_limit = args.infer_time
    shortest_path_info = read_json(os.path.join(args.shortest_path_folder, file_name))
    sum_shortest_path = sum([shortest_path_info[i][0] for i in shortest_path_info])
    
    state_data_output_folder = os.path.join(args.output_folder, "nns_running_state_temp", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(state_data_output_folder, exist_ok=True)

    # set up the formatted log file
    if pbs_log_path is not None:
        with open(pbs_log_path, "a") as f:
            f.write("### init soc {}\n".format(sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)))
    
    # Start the subprocess
    cpp_command = "{} --map {} --agentNum {} --state {} --destroyStrategy {} --pprun 1 --num_subset {} --uniform_neighbor 0 --neighborSize {} --replanTime {} --replan false".format(args.gen_subset_exe, map_path, agent_num, curr_state_path,  args.destroyStrategy, args.num_subset,  args.neighborSize, args.replan_time_limit)
    print("cpp_command: ", cpp_command)
    open_exe = subprocess.Popen(cpp_command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    while True:
        line = open_exe.stdout.readline() 
        if line != "":
            print(line)
        if "CPP program" in line:
            break
    
    # Start the inference process
    iteration = 0
    sum_removal_time = 0
    sum_replan_time = 0
    sum_run_time = 0
    sum_program_time = 0
    
    # Get the shortest path
    boolean_map = get_validLoc_bool_map(map_path)[2]
    map_w, map_h = boolean_map.shape
    agent_shortestPath = dict()
    for agent_id, path in curr_state.items():
        agent_shortestPath[agent_id] = bfs_shortest_path(boolean_map, tuple(path[0]), tuple(path[-1]))

    spatial_pool = map_setting[map_name]["spatial_pool"]
    temporal_pool = map_setting[map_name]["temporal_pool"]
    cutoff = map_setting[map_name]["cutoff"]
    shortest_path, _, _ = get_idx_from_agentPaths(agent_shortestPath, spatial_pool=spatial_pool, temporal_pool=temporal_pool, cutoff=cutoff, map_w=map_w, map_h=map_h)
    obstacle_map = downsample_obstacle_map(~boolean_map, spatial_pool=spatial_pool) # shape (feature_w * feature_h
    
    ##### load training model
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model = NNS(config=config)
    if device == 'cpu':
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.checkpoint)
    # model.load_state_dict(checkpoint)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("loaded checkpoint from ", args.checkpoint)
    model.to(device)
    model.eval()  # Set the model to training mode
        
        
    agent_selection_counts = np.zeros(int(agent_num))
    count_decay = 1
    last_accept_it = -1 
    while sum_run_time < runtime_limit or iteration < args.max_iter:
        program_start_time = datetime.datetime.now()
        print("### Iteration: {} init soc {}".format(iteration, sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)))
        
        ##### Generate removal set #####
        _, iterInfo, open_exe = cpp_removal_Open(exe=open_exe, state=curr_state_path)

        ##### Generate NNS input #####
        curr_paths, agent_paths, agent_paths_mask, subagents_idx, subset_mask, nns_input_subsets = gen_nns_input(curr_state, list(iterInfo["removalSet_info"].keys()))
        
        ##### NNS inference #####
        prediction, infer_time = nns_prediction(model, device, curr_paths, agent_paths, agent_paths_mask, shortest_path, obstacle_map, subagents_idx, subset_mask)
        prediction = prediction.squeeze().cpu().detach().numpy()
            
        ##### Update state #####
        curr_state, replan_time, last_accept_it, agent_selection_counts, count_decay = update_state_based_on_nns_prediction(prediction, curr_state, nns_input_subsets, agent_selection_counts, args.pbs_replan_exe, map_path, agent_num, curr_state_path, count_decay, iteration, last_accept_it, args.pbs_replan_time_limit)
        curr_state_path = os.path.join(state_data_output_folder, "map-{}-scene-{}-agent-{}-state.json".format(map_name, scene_num, agent_num))
        save_json(curr_state_path, curr_state)
        
        ##### Update time #####
        sets_removal_time = sum([iterInfo["removalSet_info"][i]["removal_time"] for i in iterInfo["removalSet_info"]])
        sum_replan_time += replan_time

        sum_removal_time += (sets_removal_time + infer_time)
            
        program_run_time = (datetime.datetime.now() - program_start_time).total_seconds()
        sum_program_time += program_run_time
        sum_run_time = sum_removal_time + sum_replan_time
        current_soc = sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)
        delay =  current_soc - sum_shortest_path
        print("### Iteration: {} soc {} sum_run_time {:.6f} sum_program_time {:.6f} replan_time {:.6f} removal_time {:.6f} infer_time {:.6f} delay {}".format(iteration, current_soc, sum_run_time, sum_program_time, replan_time, sets_removal_time, infer_time, delay))
        if pbs_log_path is not None and iteration % args.log_iter == 0:
            with open(pbs_log_path, "a") as f:
                f.write("### Iteration: {} soc {} sum_run_time {:.6f} sum_program_time {:.6f} replan_time {:.6f} removal_time {:.6f} infer_time {:.6f} delay {}\n".format(iteration, current_soc, sum_run_time, sum_program_time, replan_time, sets_removal_time, infer_time, delay))
            print("update state and log: ", pbs_log_path)
        iteration += 1
        


def run_pp_NNS_inference_with_openCpp_resume(args):
    
    def gen_nns_input(state, removal_sets, get_subset_only=False):
        nonlocal args
        nonlocal map_w
        nonlocal map_h
        
        if args.neighborSize == 50 or args.neighborSize == 25 or args.neighborSize == 10:
            pad_nb_size = args.neighborSize
        else:
            pad_nb_size = 32
        
        subagents_idx = np.zeros((args.num_subset, pad_nb_size)) - 1
        
        valid_subset_idx = 0
        nns_input_subsets = []
        for one_removal_set in removal_sets:
            if len(one_removal_set)  > pad_nb_size or len(one_removal_set) <= 0:
                print("subset size", len(one_removal_set))
            else: 
                subagents_idx[valid_subset_idx] = np.pad(one_removal_set, (0, pad_nb_size - len(one_removal_set)), constant_values=-1) # np.array(one_removal_set)
                valid_subset_idx += 1
                nns_input_subsets.append(one_removal_set)
                
        remain_num = args.num_subset - valid_subset_idx
        upsample_idx = np.random.choice(valid_subset_idx, remain_num, replace=True)
        
        
        subagents_idx[valid_subset_idx:] = subagents_idx[upsample_idx]
        nns_input_subsets += [nns_input_subsets[i] for i in upsample_idx]
        subset_mask = subagents_idx == -1
        
        if get_subset_only:
            return nns_input_subsets
        else:
            curr_paths, agent_paths, agent_paths_mask = get_idx_from_agentPaths(state, spatial_pool=spatial_pool, temporal_pool=temporal_pool, cutoff=cutoff, map_w=map_w, map_h=map_h)
            return curr_paths, agent_paths, agent_paths_mask, subagents_idx, subset_mask, nns_input_subsets
        
    
    def nns_prediction(model, device, curr_paths, agent_paths, agent_paths_mask, shortest_path, obstacle, subagents_idx, subset_mask):
        curr_paths = torch.tensor(curr_paths, dtype=torch.long).unsqueeze(0)
        agent_paths = torch.tensor(agent_paths, dtype=torch.long).unsqueeze(0)
        agent_paths_mask = torch.tensor(agent_paths_mask, dtype=torch.bool).unsqueeze(0)
        shortest_path = torch.tensor(shortest_path, dtype=torch.long).unsqueeze(0)
        obstacle = torch.tensor(obstacle, dtype=torch.long).unsqueeze(0)
        subagents_idx = torch.tensor(subagents_idx, dtype=torch.long).unsqueeze(0)
        subset_mask = torch.tensor(subset_mask, dtype=torch.bool).unsqueeze(0)
        
        batch_data = (curr_paths, agent_paths, agent_paths_mask, shortest_path, obstacle, subagents_idx, subset_mask)
        batch_data = [item.to(device) for item in batch_data]
        infer_start = datetime.datetime.now()
        prediction = model(batch_data)
        infer_time = (datetime.datetime.now() - infer_start).total_seconds()
        
        return prediction, infer_time
    
    def update_state_based_on_nns_prediction_with_openPPreplan(prediction, adaptive_weight, curr_state, removal_strategy_list, removal_sets, agent_selection_counts, pp_open_exe, curr_state_path, count_decay, it, last_accept_it, pp_replan_time_limit=20):
        gamma = 0.01
        sub_selection_counts = [(agent_selection_counts[np.array(i)]).sum() for i in removal_sets]
        keep_count_threshold = np.random.choice(sub_selection_counts)
        keep_mask = sub_selection_counts < keep_count_threshold
        
        prediction[~keep_mask] = -np.inf
        selected_subset_index = np.argmax(prediction)
        
        improvement = 0
        strategy = removal_strategy_list[selected_subset_index]
        agnet_paths, replan_time, improvement = cpp_pp_replanOpen(pp_open_exe, curr_state_path, removal_sets[selected_subset_index])
        if improvement > 0 and replan_time <= pp_replan_time_limit:
            adaptive_weight[strategy] = (1-gamma) * adaptive_weight[strategy] + gamma * improvement
            for agent_id in agnet_paths:
                curr_state[agent_id] = agnet_paths[agent_id].copy()
            if last_accept_it != -1:
                count_decay = min(count_decay, 1 / (it - last_accept_it))
            last_accept_it = it
            agent_selection_counts[np.array(removal_sets[selected_subset_index])] += 1
        else:
            adaptive_weight[strategy] = (1-gamma) * adaptive_weight[strategy]
        if replan_time > pp_replan_time_limit:
            replan_time = pp_replan_time_limit
        return curr_state, replan_time, last_accept_it, agent_selection_counts, count_decay, adaptive_weight
            
    map_setting = {"empty-32-32": {"spatial_pool":1, "temporal_pool":1, "cutoff":48},
                   "random-32-32-20": {"spatial_pool":1, "temporal_pool":1, "cutoff":48},
                   "ost003d": {"spatial_pool":4, "temporal_pool":4, "cutoff":192},
                   "warehouse-10-20-10-2-1": {"spatial_pool":1, "temporal_pool":2, "cutoff":96},
                   "den520d": {"spatial_pool":4, "temporal_pool":4, "cutoff":216},
                   "Paris_1_256": {"spatial_pool":4, "temporal_pool":4, "cutoff":216}}
    
    # check the input parameters
    assert args.output_folder != "", "output_folder should be provided"
    assert os.path.exists(args.pbs_replan_exe), "pbs_replan_exe {} does not exist".format(args.pbs_replan_exe)
    
    # make the output folder
    initial_state_path = args.initial_state

    # initialization
    iteration = 0
    file_name = os.path.basename(initial_state_path)
    map_name = re.findall(r'map-(.*?)-scene', file_name)[0]
    scene_num = re.findall(r'scene-(.*?)-agent', file_name)[0]
    if "method" in file_name:
        agent_num = re.findall(r'agent-(.*?)-method', file_name)[0]
    else:
        agent_num = re.findall(r'agent-(.*?).json', file_name)[0]
    map_path = os.path.join(args.map_folder, map_name + ".map")
    curr_state_path = initial_state_path
    iterInfo = {"tabu_list": []}
    curr_state = read_json(curr_state_path)
    curr_state = {int(k): v for k, v in curr_state.items()}
    runtime_limit = args.infer_time
    shortest_path_info = read_json(os.path.join(args.shortest_path_folder,  "map-{}-scene-{}-agent-{}.json".format(map_name, scene_num, agent_num)))
    sum_shortest_path = sum([shortest_path_info[i][0] for i in shortest_path_info])
    
    # Start the inference process
    iteration = 0
    sum_run_time = 0
    sum_program_time = 0
    
    
    result_log_path = args.log_path
    log_folder = os.path.dirname(result_log_path)
    os.makedirs(log_folder, exist_ok=True)
    if os.path.exists(result_log_path) and not args.overwrite:
        print("exist log file, skip : ", result_log_path)
        return 
    elif os.path.exists(result_log_path) and args.overwrite:
        os.remove(result_log_path)
        print("exist log file, remove and overwrite : ", result_log_path)
        
    # Start the subprocess
    cpp_command = "{} --map {} --agentNum {} --state {} --destroyStrategy {} --pprun 1 --num_subset {} --uniform_neighbor 0 --neighborSize {} --replanTime {} --replan false".format(args.gen_subset_exe, map_path, agent_num, curr_state_path,  args.destroyStrategy, args.num_subset, args.neighborSize, args.replan_time_limit)
    print("cpp_command: ", cpp_command)
    open_exe = subprocess.Popen(cpp_command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    while True:
        line = open_exe.stdout.readline() 
        if line != "":
            print(line)
        if "CPP program" in line:
            break
        
    cpp_command = "{} --map {} --agentNum {} --state {} --destroyStrategy {} --pprun 1 --num_subset {} --uniform_neighbor 0 --neighborSize {} --replanTime {} --replan false".format(args.pp_exe, map_path, agent_num, curr_state_path,  args.destroyStrategy, args.num_subset, args.neighborSize, args.replan_time_limit)
    print("cpp_command: ", cpp_command)
    open_pp_exe = subprocess.Popen(cpp_command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    while True:
        line = open_pp_exe.stdout.readline() 
        if line != "":
            print(line)
        if "CPP program" in line:
            break

    # Get the shortest path
    boolean_map = get_validLoc_bool_map(map_path)[2]
    map_w, map_h = boolean_map.shape
    agent_shortestPath = dict()
    for agent_id, path in curr_state.items():
        agent_shortestPath[agent_id] = bfs_shortest_path(boolean_map, tuple(path[0]), tuple(path[-1]))

    spatial_pool = map_setting[map_name]["spatial_pool"]
    temporal_pool = map_setting[map_name]["temporal_pool"]
    cutoff = map_setting[map_name]["cutoff"]
    shortest_path, _, _ = get_idx_from_agentPaths(agent_shortestPath, spatial_pool=spatial_pool, temporal_pool=temporal_pool, cutoff=cutoff, map_w=map_w, map_h=map_h)
    obstacle_map = downsample_obstacle_map(~boolean_map, spatial_pool=spatial_pool) # shape (feature_w * feature_h
    
    ##### load training model
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model = NNS(config=config)
    if device == 'cpu':
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.checkpoint)
    # model.load_state_dict(checkpoint)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("loaded checkpoint from ", args.checkpoint)
    model.to(device)
    model.eval()  # Set the model to training mode
        
    agent_selection_counts = np.zeros(int(agent_num))
    count_decay = 1
    last_accept_it = -1 
    adaptive_weight = [1,1,1]
    while sum_run_time < runtime_limit or iteration < args.max_iter:
        program_start_time = datetime.datetime.now()
        print("### Iteration: {} init soc {}".format(iteration, sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)))
        
        ##### Generate removal set #####
        _, iterInfo, open_exe = cpp_removal_Open(exe=open_exe, state=curr_state_path, adaptive_weight=adaptive_weight)

        ##### Generate NNS input #####
        curr_paths, agent_paths, agent_paths_mask, subagents_idx, subset_mask, nns_input_subsets = gen_nns_input(curr_state, list(iterInfo["removalSet_info"].keys()))
        
        ##### NNS inference #####
        prediction, infer_time = nns_prediction(model, device, curr_paths, agent_paths, agent_paths_mask, shortest_path, obstacle_map, subagents_idx, subset_mask)
        prediction = prediction.squeeze().cpu().detach().numpy()
        
        ##### Update state #####
        removal_strategy_list = [iterInfo["removalSet_info"][i]["destroy_strategy"] for i in iterInfo["removalSet_info"].keys()]
        curr_state, replan_time, last_accept_it, agent_selection_counts, count_decay, adaptive_weight = update_state_based_on_nns_prediction_with_openPPreplan(prediction, adaptive_weight, curr_state, removal_strategy_list, nns_input_subsets, agent_selection_counts, open_pp_exe, curr_state_path, count_decay, iteration, last_accept_it, args.replan_time_limit)
        save_json(curr_state_path, curr_state)
        
        ##### Update time #####
        sets_removal_time = sum([iterInfo["removalSet_info"][i]["removal_time"] for i in iterInfo["removalSet_info"]])
        sum_run_time += replan_time

        sum_run_time += (sets_removal_time + infer_time)
            
        program_run_time = (datetime.datetime.now() - program_start_time).total_seconds()
        sum_program_time += program_run_time
        current_soc = sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)
        delay =  current_soc - sum_shortest_path
        print("### Iteration: {} soc {} sum_run_time {:.6f} sum_program_time {:.6f} replan_time {:.6f} removal_time {:.6f} infer_time {:.6f} delay {}".format(iteration, current_soc, sum_run_time, sum_program_time, replan_time, sets_removal_time, infer_time, delay))
        if result_log_path is not None and iteration % args.log_iter == 0:
            with open(result_log_path, "a") as f:
                f.write("### Iteration: {} soc {} sum_run_time {:.6f} sum_program_time {:.6f} replan_time {:.6f} removal_time {:.6f} infer_time {:.6f} delay {}\n".format(iteration, current_soc, sum_run_time, sum_program_time, replan_time, sets_removal_time, infer_time, delay))
            print("update state and log: ", result_log_path)
        iteration += 1


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--replan_solver", type=str, default="pp", help="pbs / pp; collection nns with pp OR pbs")
    # input 
    parser.add_argument("--initial_state", type=str, default="")
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--map_folder", type=str, default="data/map")
    parser.add_argument("--scene_folder", type=str, default="data/scene")
    parser.add_argument("--shortest_path_folder", type=str, default="data/example/shortest_path")
    # NNS model
    parser.add_argument('--checkpoint', type=str, default="", help='')
    parser.add_argument('--config', type=str, default="", help='')
    # LNS parameters
    parser.add_argument("--destroyStrategy", type=str, default=None, help="LNS destroy strategy")
    parser.add_argument("--neighborSize", type=int, default=25, help="neighbor size")
    parser.add_argument("--pprun", type=int, default=6, help="the number of runs for PP when collecting data with PP")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--num_subset", type=int, default=20)
    parser.add_argument("--runtime_limit", type=float, default=1)
    parser.add_argument("--replan_time_limit", type=float, default=0.6)
    parser.add_argument("--gen_subset_exe", type=str, default="exe/lns-removal-replan/lns-removal-replan")
    parser.add_argument("--pbs_replan_exe", type=str, default="exe/pbs-replan/pbs-replan")
    parser.add_argument("--pp_exe", type=str, default="exe/pp-replan/pp_open")
    # optional
    parser.add_argument("--pbs_replan_time_limit", type=int, default=100, help="replan time limit for pbs")
    parser.add_argument("--infer_time", type=float, default=300, help="run time limit for inference, only counts the removal and replan time without all other program overhead")
    parser.add_argument("--resume", type=int, default=0, help="resume the data collection process or not")
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--num_cores", type=int, default=1, help="number of cores to use to run PBS in parallel, the number of cores should be the same as the number of CPUs in the allocated node")
    parser.add_argument("--log_path", type=str, default="output/data_collection.log")
    parser.add_argument("--log_iter", type=int, default=1, help="gap for iteration to log")
    parser.add_argument("--save_iter_theshold", type=int, default=100, help="save the nns data every save_iter_theshold iterations incase the json file is too large")

    args = parser.parse_args()
    
    if args.replan_solver == "pbs":
        run_pbs_NNS_inference_with_openCpp(args)
    elif args.replan_solver == "pp":
        run_pp_NNS_inference_with_openCpp_resume(args)
        
    
