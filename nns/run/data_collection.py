import os
import datetime
import argparse
import re
from multiprocessing import Pool
import numpy as np
import subprocess
from utils import *

def tranfrom_json_to_nns_npz(nns_data, nb_size, map_path, shortest_path, obstacle_map, final_output_path, args):
    assert shortest_path is not None, "shortest_path is None"
    assert obstacle_map is not None, "obstacle_map is None"

    map_setting = {"empty-32-32": {"spatial_pool":1, "temporal_pool":1, "cutoff":48},
                   "random-32-32-20": {"spatial_pool":1, "temporal_pool":1, "cutoff":48},
                   "ost003d": {"spatial_pool":4, "temporal_pool":4, "cutoff":192},
                   "warehouse-10-20-10-2-1": {"spatial_pool":1, "temporal_pool":2, "cutoff":96},
                   "den520d": {"spatial_pool":4, "temporal_pool":4, "cutoff":216},
                   "Paris_1_256": {"spatial_pool":4, "temporal_pool":4, "cutoff":216}}

    process_start_time = datetime.datetime.now()
    map_tag = os.path.basename(map_path).split(".")[0]
    spatial_pool, temporal_pool, cutoff = map_setting[map_tag]["spatial_pool"], map_setting[map_tag]["temporal_pool"], map_setting[map_tag]["cutoff"]
    print("spatial_pool: {}, temporal_pool: {}, cutoff: {}".format(spatial_pool, temporal_pool, cutoff))
    boolean_map = get_validLoc_bool_map(map_path)[2]
    map_w, map_h = boolean_map.shape
    
    exact_percentages = []
    
    curr_paths_list = []
    agent_paths_list = []
    agent_paths_mask_list = []
    subagents_idx_list = []
    subset_mask_list = []
    gt_imps_list = []
    
    for _, nns_info in nns_data.items():
        
        curr_paths, agent_paths, agent_paths_mask = get_idx_from_agentPaths(nns_info["info"]["init_state"], spatial_pool=spatial_pool, temporal_pool=temporal_pool, cutoff=cutoff, map_w=map_w, map_h=map_h)
        if nb_size == 50 or nb_size == 25 or nb_size == 10:
            pad_nb_size = nb_size
        else:
            pad_nb_size = 32
        subagents_idx = np.zeros((args.num_subset, pad_nb_size)) - 1
        gt_imps = np.zeros((args.num_subset, 1))
        valid_subset_idx = 0

        for one_subset_info in nns_info["info"]["removal_sets"]:
            if len(one_subset_info["replanAgents"]) <= 0 or len(one_subset_info["replanAgents"]) > pad_nb_size:
                print("wrong subset size", len(one_subset_info["replanAgents"]))
            else:
                subagents_idx[valid_subset_idx] = np.pad(one_subset_info["replanAgents"], (0, pad_nb_size - len(one_subset_info["replanAgents"])), constant_values=-1) # np.array(one_subset_info["replanAgents"])
                gt_imps[valid_subset_idx] = one_subset_info["improvement"]
                valid_subset_idx += 1
        exact_percentages.append(valid_subset_idx / args.num_subset)
        remain_num = args.num_subset - valid_subset_idx
        upsample_idx = np.random.choice(valid_subset_idx, remain_num, replace=True)
        
        subagents_idx[valid_subset_idx:] = subagents_idx[upsample_idx]
        gt_imps[valid_subset_idx:] = gt_imps[upsample_idx]
        subset_mask = subagents_idx == -1
        
        curr_paths_list.append(curr_paths)
        agent_paths_list.append(agent_paths)
        agent_paths_mask_list.append(agent_paths_mask)
        subagents_idx_list.append(subagents_idx)
        subset_mask_list.append(subset_mask)
        gt_imps_list.append(gt_imps)
        
        print("exact_percentages: ", exact_percentages[-1])
        print("avg exact_percentages: ", np.mean(exact_percentages))
    
    curr_paths_list = np.concatenate([np.expand_dims(arr, axis=0) for arr in curr_paths_list], axis=0)
    agent_paths_list = np.concatenate([np.expand_dims(arr, axis=0) for arr in agent_paths_list], axis=0)
    agent_paths_mask_list = np.concatenate([np.expand_dims(arr, axis=0) for arr in agent_paths_mask_list], axis=0)
    subagents_idx_list = np.concatenate([np.expand_dims(arr, axis=0) for arr in subagents_idx_list], axis=0)
    subset_mask_list = np.concatenate([np.expand_dims(arr, axis=0) for arr in subset_mask_list], axis=0)
    gt_imps_list = np.concatenate([np.expand_dims(arr, axis=0) for arr in gt_imps_list], axis=0)
    
    
    np.savez(final_output_path, curr_paths=curr_paths_list, agent_paths=agent_paths_list, agent_paths_mask=agent_paths_mask_list, shortest_path=shortest_path, obstacle=obstacle_map, subagents_idx=subagents_idx_list, subset_mask=subset_mask_list, gt_imps=gt_imps_list)
    print("save to: ", final_output_path)
    print("process time: ", (datetime.datetime.now() - process_start_time).total_seconds())


def collect_nns_data_with_pp(args):
    # check the input parameters
    assert args.output_folder != "", "output_folder should be provided"
    
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

    nb_tag = "Size{}".format(args.neighborSize)
    nns_data_output_folder = os.path.join(args.output_folder, "nns_data", "map-{}-agent-{}-method-{}-nb-{}".format(map_name, agent_num, args.destroyStrategy, nb_tag))
    os.makedirs(nns_data_output_folder, exist_ok=True)

    state_data_output_folder = os.path.join(args.output_folder, "nns_running_state_temp", "nns_data_collection")
    os.makedirs(state_data_output_folder, exist_ok=True)
    
    # set up the formatted log file
    if pbs_log_path is not None:
        with open(pbs_log_path, "a") as f:
            f.write("### init soc {}\n".format(sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)))
    
    # Start the subprocess
    cpp_command = "{} --map {} --agentNum {} --state {} --destroyStrategy {} --pprun {} --num_subset {} --uniform_neighbor 0 --neighborSize {} --replanTime {} --replan true".format(args.gen_subset_exe, map_path, agent_num, curr_state_path, args.pprun,  args.destroyStrategy, args.num_subset, args.neighborSize, args.replan_time_limit)
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
    nns_iter_info = dict()
    
    just_resume = False
    curr_state_path_resume = os.path.join(state_data_output_folder, "map-{}-scene-{}-agent-{}-state-resume.json".format(map_name, scene_num, agent_num))
    if os.path.exists(curr_state_path_resume)  and args.resume:
        resume_nns_iter_info = read_json(curr_state_path_resume)
        iteration = max([int(i) for i in resume_nns_iter_info.keys()])
        curr_state = resume_nns_iter_info[str(iteration)]["info"]["init_state"]
        curr_state = {int(k): v for k, v in curr_state.items()}
        curr_state_path = os.path.join(state_data_output_folder, "map-{}-scene-{}-agent-{}-state.json".format(map_name, scene_num, agent_num))
        save_json(curr_state_path, curr_state)
        print("resume from the last iteration: ", iteration, curr_state_path_resume, "saved current state to ", curr_state_path)
        just_resume = True
    
    shortest_path = None
    obstacle_map = None
    map_setting = {"empty-32-32": {"spatial_pool":1, "temporal_pool":1, "cutoff":48},
                "random-32-32-20": {"spatial_pool":1, "temporal_pool":1, "cutoff":48},
                "ost003d": {"spatial_pool":4, "temporal_pool":4, "cutoff":192},
                "warehouse-10-20-10-2-1": {"spatial_pool":1, "temporal_pool":2, "cutoff":96},
                "den520d": {"spatial_pool":4, "temporal_pool":4, "cutoff":216},
                "Paris_1_256": {"spatial_pool":4, "temporal_pool":4, "cutoff":216}}
    boolean_map = get_validLoc_bool_map(map_path)[2]
    
    map_tag = os.path.basename(map_path).split(".")[0]
    spatial_pool, temporal_pool, cutoff = map_setting[map_tag]["spatial_pool"], map_setting[map_tag]["temporal_pool"], map_setting[map_tag]["cutoff"]
    
    agent_shortestPath = dict()
    for agent_id, path in curr_state.items():
        agent_shortestPath[agent_id] = bfs_shortest_path(boolean_map, tuple(path[0]), tuple(path[-1]))
    map_w, map_h = boolean_map.shape
    shortest_path, _, _ = get_idx_from_agentPaths(agent_shortestPath, spatial_pool=spatial_pool, temporal_pool=temporal_pool, cutoff=cutoff, map_w=map_w, map_h=map_h)
    obstacle_map = downsample_obstacle_map(~boolean_map, spatial_pool=spatial_pool) # shape (feature_w * feature_h
    batch_end = 0
        
    while  iteration < args.max_iter:
        program_start_time = datetime.datetime.now()
        print("### Iteration: {} init soc {}".format(iteration, sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)))
        
        ##### Generate removal set #####
        _, iterInfo, open_exe = cpp_removal_replanOpen(exe=open_exe, state=curr_state_path, adaptive_weight=[1,1,1])

        ##### Generate improvement using PBS in parallel #####
        removal_info = dict()
        removal_info["removal_sets"] = []
        best_replan_time = None
        best_imp = None
        best_replan_paths = None
        for removal_set, info in iterInfo["removalSet_info"].items():
            if best_replan_time is None:
                best_replan_time = info["average_replan_time"]
                best_imp = info["avg_improvement"]
                best_replan_paths = info["agent_paths"]
            else:
                if info["avg_improvement"] > best_imp:
                    best_replan_time = info["average_replan_time"]
                    best_imp = info["avg_improvement"]
                    best_replan_paths = info["agent_paths"]
            removal_info["removal_sets"].append({"replanAgents": list(removal_set), "improvement": info["avg_improvement"], "replan_time": info["average_replan_time"]})
            

        nns_iter_info[iteration] = dict()
        nns_iter_info[iteration]["info"] = removal_info
        nns_iter_info[iteration]["info"]["init_state"] = curr_state.copy()
        
        ##### Update state #####
        if len(best_replan_paths) > 0 and best_imp > 0:
            for agent_id in best_replan_paths:
                curr_state[agent_id] = best_replan_paths[agent_id]
        curr_state_path = os.path.join(state_data_output_folder, "map-{}-scene-{}-agent-{}-state.json".format(map_name, scene_num, agent_num))
        save_json(curr_state_path, curr_state)

        
        ##### Update time #####
        sets_removal_time = sum([iterInfo["removalSet_info"][i]["removal_time"] for i in iterInfo["removalSet_info"]])
        avg_replan_time = np.mean([i["replan_time"] for i in removal_info["removal_sets"]])
        sum_replan_time += avg_replan_time
        sum_removal_time += sets_removal_time
        program_run_time = (datetime.datetime.now() - program_start_time).total_seconds()
        sum_program_time += program_run_time
        sum_run_time += (sets_removal_time + best_replan_time) # best_replan_time is the replan time of the best subset, sum_removal_time runs in parallel
        print("### Iteration: {} soc {} sum_run_time {:.6f} sum_program_time {:.6f} (best) replan_time {:.6f} avg_replan_time {:.6f} (sum) removal_time {:.6f}".format(iteration, sum([len(curr_state[i]) for i in curr_state]) - len(curr_state), sum_run_time, sum_program_time, best_replan_time, avg_replan_time, sum_removal_time))
        if pbs_log_path is not None and iteration % args.log_iter == 0:
            current_soc = sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)
            with open(pbs_log_path, "a") as f:
                f.write("### Iteration: {} soc {} sum_run_time {:.6f} sum_program_time {:.6f} (best) replan_time {:.6f} avg_replan_time {:.6f} (sum) removal_time {:.6f}\n".format(iteration, current_soc, sum_run_time, sum_program_time, best_replan_time, avg_replan_time, sum_removal_time))
            print("update state and log: ", pbs_log_path)
        
        if args.max_iter > args.save_iter_theshold: # if over 100 iteration save first
            if iteration % args.save_iter_theshold == 0 and iteration > 0 and not just_resume:
                batch_start = iteration - args.save_iter_theshold
                batch_end = iteration - 1
                new_batch_info_save_path = os.path.join(nns_data_output_folder, "map-{}-scene-{}-agent-{}-method-{}-nb-{}-s{}-e{}.json".format(map_name, scene_num, agent_num, args.destroyStrategy,nb_tag, batch_start, batch_end))
                new_batch_to_save = {k: v for k, v in nns_iter_info.items() if k < iteration}
                tranfrom_json_to_nns_npz(new_batch_to_save, args.neighborSize, map_path, shortest_path, obstacle_map, new_batch_info_save_path.replace(".json", ".npz"), args)
                nns_iter_info = {iteration : nns_iter_info[iteration]} # only save the current iteration info
                curr_state_path_resume = os.path.join(state_data_output_folder, "map-{}-scene-{}-agent-{}-state-resume.json".format(map_name, scene_num, agent_num))
                save_json(curr_state_path_resume, nns_iter_info)
                
        iteration += 1
        just_resume = False
        
        if iteration == args.max_iter :
            new_batch_info_save_path = os.path.join(nns_data_output_folder, "map-{}-scene-{}-agent-{}-method-{}-nb-{}-s{}-e{}.json".format(map_name, scene_num, agent_num, args.destroyStrategy,nb_tag, batch_end, iteration))
            new_batch_to_save = nns_iter_info
            tranfrom_json_to_nns_npz(new_batch_to_save, args.neighborSize, map_path, shortest_path, obstacle_map, new_batch_info_save_path.replace(".json", ".npz"), args)


def collect_nns_data_with_pbs(args):
    
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

    state_data_output_folder = os.path.join(args.output_folder, "nns_running_state_temp", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(state_data_output_folder, exist_ok=True)

    
    # set up the formatted log file
    if pbs_log_path is not None:
        with open(pbs_log_path, "a") as f:
            f.write("### init soc {}\n".format(sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)))
    
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
        
    # Start the inference process
    iteration = 0
    sum_removal_time = 0
    sum_replan_time = 0
    sum_run_time = 0
    sum_program_time = 0
    
    nns_iter_info = dict()
    nns_data_output_folder = os.path.join(args.output_folder, "nns_data", "map-{}-agent-{}".format(map_name, agent_num))
    os.makedirs(nns_data_output_folder, exist_ok=True)
    npz_info_save_path = os.path.join(nns_data_output_folder, "map-{}-scene-{}-agent-{}.npz".format(map_name, scene_num, agent_num))
    if os.path.exists(npz_info_save_path) and not args.overwrite:
        print("npz_info_save_path already exists, skip: ", npz_info_save_path)
        return

    while iteration < args.max_iter:
        program_start_time = datetime.datetime.now()
        print("### Iteration: {} init soc {}".format(iteration, sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)))
        
        ##### Generate removal set #####
        _, iterInfo, open_exe = cpp_removal_Open(exe=open_exe, state=curr_state_path, adaptive_weight=[1,1,1])

        ##### Generate improvement using PBS in parallel #####
        with Pool(args.num_cores) as pool:
            args_for_func = [(args.pbs_replan_exe, i, map_path, agent_num, curr_state_path) for i in iterInfo["removalSet_info"]]
            results = pool.starmap(pbs_replan, args_for_func)
        removal_info = dict()
        removal_info["removal_sets"] = []
        _, _, best_replan_time, best_imp, _, _, best_replan_paths = results[0]
        for _, replanAgents, replan_time, improvement, initial_cost, replaned_cost, agnet_paths in results:
            removal_info["removal_sets"].append({"replanAgents": list(replanAgents), "improvement": improvement, "initial_cost": initial_cost, "replaned_cost": replaned_cost, "replan_time": replan_time})
            if improvement > best_imp:
                best_imp = improvement
                best_replan_paths = agnet_paths
                best_replan_time = replan_time
        nns_iter_info[iteration] = dict()
        nns_iter_info[iteration]["info"] = removal_info
        nns_iter_info[iteration]["info"]["init_state"] = curr_state.copy()
        
        ##### Update state #####
        if len(best_replan_paths) > 0 and best_imp > 0:
            for agent_id in best_replan_paths:
                curr_state[agent_id] = best_replan_paths[agent_id]
        curr_state_path = os.path.join(state_data_output_folder, "map-{}-scene-{}-agent-{}-state.json".format(map_name, scene_num, agent_num))
        save_json(curr_state_path, curr_state)

        
        ##### Update time #####
        sets_removal_time = sum([iterInfo["removalSet_info"][i]["removal_time"] for i in iterInfo["removalSet_info"]])
        avg_replan_time = np.mean([i["replan_time"] for i in removal_info["removal_sets"]])
        sum_replan_time += avg_replan_time
        sum_removal_time += sets_removal_time
        program_run_time = (datetime.datetime.now() - program_start_time).total_seconds()
        sum_program_time += program_run_time
        sum_run_time += (sets_removal_time + best_replan_time) # best_replan_time is the replan time of the best subset, sum_removal_time runs in parallel
        print("### Iteration: {} soc {} sum_run_time {:.6f} sum_program_time {:.6f} (best) replan_time {:.6f} avg_replan_time {:.6f} (sum) removal_time {:.6f}".format(iteration, sum([len(curr_state[i]) for i in curr_state]) - len(curr_state), sum_run_time, sum_program_time, best_replan_time, avg_replan_time, sum_removal_time))
        if pbs_log_path is not None and iteration % args.log_iter == 0:
            current_soc = sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)
            with open(pbs_log_path, "a") as f:
                f.write("### Iteration: {} soc {} sum_run_time {:.6f} sum_program_time {:.6f} (best) replan_time {:.6f} avg_replan_time {:.6f} (sum) removal_time {:.6f}\n".format(iteration, current_soc, sum_run_time, sum_program_time, best_replan_time, avg_replan_time, sum_removal_time))
            print("update state and log: ", pbs_log_path)
        iteration += 1
        
    ##### Save the nns data #####
    map_setting = {"empty-32-32": {"spatial_pool":1, "temporal_pool":1, "cutoff":48},
            "random-32-32-20": {"spatial_pool":1, "temporal_pool":1, "cutoff":48},
            "ost003d": {"spatial_pool":4, "temporal_pool":4, "cutoff":192},
            "warehouse-10-20-10-2-1": {"spatial_pool":1, "temporal_pool":2, "cutoff":96},
            "den520d": {"spatial_pool":4, "temporal_pool":4, "cutoff":216},
            "Paris_1_256": {"spatial_pool":4, "temporal_pool":4, "cutoff":216}}
    boolean_map = get_validLoc_bool_map(map_path)[2]
    
    map_tag = os.path.basename(map_path).split(".")[0]
    spatial_pool, temporal_pool, cutoff = map_setting[map_tag]["spatial_pool"], map_setting[map_tag]["temporal_pool"], map_setting[map_tag]["cutoff"]
        
    agent_shortestPath = dict()
    for agent_id, path in curr_state.items():
        agent_shortestPath[agent_id] = bfs_shortest_path(boolean_map, tuple(path[0]), tuple(path[-1]))
    map_w, map_h = boolean_map.shape
    shortest_path, _, _ = get_idx_from_agentPaths(agent_shortestPath, spatial_pool=spatial_pool, temporal_pool=temporal_pool, cutoff=cutoff, map_w=map_w, map_h=map_h)
    obstacle_map = downsample_obstacle_map(~boolean_map, spatial_pool=spatial_pool) # shape (feature_w * feature_h
    
    tranfrom_json_to_nns_npz(nns_iter_info, args.neighborSize, map_path, shortest_path, obstacle_map, npz_info_save_path, args)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--replan_solver", type=str, default="pp", help="pbs / pp; collection nns with pp OR pbs")
    # input 
    parser.add_argument("--initial_state", type=str, default="")
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--map_folder", type=str, default="data/map")
    parser.add_argument("--scene_folder", type=str, default="data/scene")
    # data collection
    parser.add_argument("--destroyStrategy", type=str, default=None, help="LNS destroy strategy")
    parser.add_argument("--neighborSize", type=int, default=25, help="neighbor size")
    parser.add_argument("--pprun", type=int, default=6, help="the number of runs for PP when collecting data with PP")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--num_subset", type=int, default=20)
    parser.add_argument("--runtime_limit", type=float, default=1)
    parser.add_argument("--replan_time_limit", type=float, default=0.6)
    parser.add_argument("--gen_subset_exe", type=str, default="exe/lns-removal-replan/lns-removal-replan")
    parser.add_argument("--pbs_replan_exe", type=str, default="exe/pbs-replan/pbs-replan")
    # optional
    parser.add_argument("--resume", type=int, default=0, help="resume the data collection process or not")
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--num_cores", type=int, default=1, help="number of cores to use to run PBS in parallel, the number of cores should be the same as the number of CPUs in the allocated node")
    parser.add_argument("--log_path", type=str, default="output/data_collection.log")
    parser.add_argument("--log_iter", type=int, default=1, help="gap for iteration to log")
    parser.add_argument("--save_iter_theshold", type=int, default=100, help="save the nns data every save_iter_theshold iterations incase the json file is too large")
    
    
    args = parser.parse_args()
    
    if args.replan_solver == "pbs":
        collect_nns_data_with_pbs(args)
    elif args.replan_solver == "pp":
        collect_nns_data_with_pp(args)
  

        
    
