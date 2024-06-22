from utils import *
import os
import datetime
import argparse
import re
from multiprocessing import Pool
import numpy as np
import subprocess

def update_state_weight_based_on_ranking_score(removal_strategy_list, score_list, curr_state, subset_info_list, subset_list, adaptive_weight=[1,1,0], topk=1, destroyStrategy=None, replan_time_limit=0.6):
    assert len(score_list) == len(subset_info_list) == len(subset_list) == len(removal_strategy_list), "score_list, subset_info_list, and subset_list should have the same length, len(score_list): {}, len(subset_info_list): {}, len(subset_list): {}, len(removal_strategy_list): {}".format(len(score_list), len(subset_info_list), len(subset_list), len(removal_strategy_list))
    gamma = 0.01
    # get the index of score_list
    score_idx = np.argsort(score_list)[::-1]
    pp_replan_time_list = []
    pp_removal_time = sum([item["removal_time"] for item in subset_info_list.values()])
    trial = 0
    for idx in score_idx:
        if trial >= topk:
            break
        
        trial += 1
        replan_time = subset_info_list[tuple(subset_list[idx])]["average_replan_time"]
        pp_replan_time_list.append(min(replan_time, replan_time_limit))
        
        if destroyStrategy is not None and destroyStrategy == "Adaptive":
            strategy = removal_strategy_list[idx]
            print("strategy: ", strategy, "score ", score_list[idx], "gt_imp ", subset_info_list[tuple(subset_list[idx])]["avg_improvement"],  "ori adaptive_weight: ", adaptive_weight[strategy], "removal set: " , subset_list[idx] , "improved ", subset_info_list[tuple(subset_list[idx])]["agent_paths"] != [])
        if subset_info_list[tuple(subset_list[idx])]["agent_paths"] != [] and replan_time <= 0.6: 
            improvement = subset_info_list[tuple(subset_list[idx])]["avg_improvement"] / len(tuple(subset_list[idx]))
            if destroyStrategy is not None and destroyStrategy == "Adaptive":
                adaptive_weight[strategy] = (1-gamma) * adaptive_weight[strategy] + gamma * improvement
            curr_state.update(subset_info_list[tuple(subset_list[idx])]["agent_paths"])
            break
        elif destroyStrategy is not None and destroyStrategy == "Adaptive":
            adaptive_weight[strategy] = (1-gamma) * adaptive_weight[strategy]
            
    print("### adaptive_weight: ", adaptive_weight)
    return curr_state, pp_replan_time_list, pp_removal_time, adaptive_weight, trial


def update_state_based_on_ranking_score(score_list, curr_state, subset_info_list, subset_list, topk=1):
    
    assert len(score_list) == len(subset_info_list) == len(subset_list), "score_list, subset_info_list, and subset_list should have the same length, len(score_list): {}, len(subset_info_list): {}, len(subset_list): {}".format(len(score_list), len(subset_info_list), len(subset_list))

    # get the index of score_list
    score_idx = np.argsort(score_list)[::-1]
    pp_replan_time = 0
    pp_removal_time = sum([item["removal_time"] for item in subset_info_list.values()])
    
    trial = 0
    for idx in score_idx:
        if trial >= topk:
            break
        trial += 1
        pp_replan_time += subset_info_list[tuple(subset_list[idx])]["average_replan_time"]
        if subset_info_list[tuple(subset_list[idx])]["agent_paths"] != []:
            curr_state.update(subset_info_list[tuple(subset_list[idx])]["agent_paths"])
            break
        
    return curr_state, pp_replan_time, pp_removal_time


def save_svm_feature(label_list=[], feature_list=[], qid_list=[], comment_list=[], save_path=""):
    """_summary_

    Args:
        label_list (list): list of labels
        feature_list (list): list of NORMALIZED features
        qid_list (list): list of ids for each subset
        save_path (str): path to save the svm data
        comment_list (list): list of comments
        save_path (str): path to save the svm data
    """
    
    assert len(label_list) == len(feature_list) == len(qid_list), "label, feature, and id should have the same length, len(label_list): {}, len(feature_list): {}, len(qid_list): {}".format(len(label_list), len(feature_list), len(qid_list))
    if comment_list!=[]:
        assert len(label_list) == len(comment_list), "label, feature, and comment should have the same length, len(label_list): {}, len(feature_list): {}, len(comment_list): {}".format(len(label_list), len(feature_list), len(comment_list))
    
    svm_lines = []
    for i in range(len(label_list)):
        # 3 qid:1 1:1 2:1 3:0 4:0.2 5:0 #
        one_line = "{} qid:{} ".format(label_list[i], qid_list[i]) + " ".join(["{}:{:.5f}".format(j+1, feature_list[i][j]) for j in range(len(feature_list[i]))]) + " # {:.5f}".format(comment_list[i]) if comment_list!=[] else ""
        svm_lines.append(one_line)
    save_txt(save_path, svm_lines)


def svm_get_avg_ranking(val_input_data_path, val_pred_path):
    """_summary_

    Args:
        val_input_data_path (str) : svm validation input data path, including the qid and ground truth score as comment
        val_pred_path (str) : svm validation prediction path, one score per line

    Returns:
        avgRank (float) : average of ranking for each instance
    """
    
    pred_score_list = read_txt(val_pred_path)
    val_input_data = read_txt(val_input_data_path)
    assert len(pred_score_list) == len(val_input_data), "pred_score_list and val_input_data should have the same length, len(pred_score_list): {}, len(val_input_data): {} from val_pred_path: {}, val_input_data_path: {}".format(len(pred_score_list), len(val_input_data), val_pred_path, val_input_data_path)
    
    instace_rank_list = []
    gt_score_list = [] 
    pre_idx = 0
    for idx, line in enumerate(val_input_data):
        gt_score_list.append(float(re.findall(r'# (.*?)$', line)[0].split()[0]))
        if idx == len(val_input_data) - 1 or int(re.findall(r'qid:(.*?) ', val_input_data[idx+1])[0]) != int(re.findall(r'qid:(.*?) ', val_input_data[pre_idx])[0]):
            instance_score_list = pred_score_list[pre_idx:idx + 1]
            max_score_idx = np.argmax(instance_score_list)
            instance_gt_rank = rank_list(gt_score_list[pre_idx:idx+1])
            instace_rank_list.append(instance_gt_rank[max_score_idx])
            pre_idx = idx+1
    return np.average(instace_rank_list, axis=0)


def update_one_state(index, curr_state_list, predict_scores, files_subsetInfo, iter_state_output_folder, iteration, subsetLen_list, topk=1):
    """_summary_
    Args:
        index (int): index of the current state in curr_state_list (used to locate the score_list)
        curr_state_list (list): list of current state
        num_subset (int): number of subset
        predict_scores (list): list of predict scores
        files_subsetInfo (dict): dict of subset info, key is file name, value is subset info
        iter_state_output_folder (str): folder to save the iteration state
        iteration (int): current iteration, used to name the output file
        subsetLen_list (list): list of subset length
    """


    one_state_path = curr_state_list[index]
    # score_list = predict_scores[index*num_subset:(index+1)*num_subset]
    start_idx = 0 if index == 0 else sum(subsetLen_list[:index])
    end_idx = sum(subsetLen_list[:index+1])
    score_list = predict_scores[start_idx:end_idx]
    subset_list = [i for i in files_subsetInfo[one_state_path]["removalSet_info"]]
    
    one_state = read_json(one_state_path)
    one_state, _, _ = update_state_based_on_ranking_score(score_list, one_state, files_subsetInfo[one_state_path]["removalSet_info"], subset_list, topk=topk)
    
    file_name = os.path.basename(one_state_path).strip(".json").split("_iter")[0]
    one_state_path = os.path.join(iter_state_output_folder, f"{file_name}_iter{iteration}.json")
    save_json(one_state_path, one_state)
    return index, one_state_path


def gen_label_from_cost(cost_list):
    sorted_agnet_cost = sorted(cost_list, reverse=True)

    num_of_sample = len(sorted_agnet_cost)
    score_margin = sorted_agnet_cost[int(num_of_sample*0.25)]
    score_margin2 = sorted_agnet_cost[int(num_of_sample*0.5)]
    
    if score_margin == score_margin2:
        print("score_margin == score_margin2 : ", score_margin, score_margin2)
     
    label = [] 
    for cost in cost_list:
        if cost >= score_margin:
            label.append(2)
        elif cost >= score_margin2:
            label.append(1)
        else:
            label.append(0)
            
    return label


def gen_svm_feature(selected_agent_set, instance, shortest_path_data, map_degree):
    paths = []
    starts = []
    ends = []
    unselected_agent_set = [i for i in range(len(instance)) if i not in selected_agent_set]
    for one_info in instance.values():
        paths.append(one_info)
        starts.append(one_info[0])
        ends.append(one_info[-1])
    
    # static feature
    # Row and column numbers of ai’s start and goal vertices.
    selected_starts = []
    for agent in selected_agent_set:
        selected_starts.append(starts[agent]) 
    
    unselected_starts = []
    for agent in unselected_agent_set:
        unselected_starts.append(starts[agent])
        
    selected_starts = np.array(selected_starts)
    unselected_starts = np.array(unselected_starts)
    
    selected_ends = []
    for agent in selected_agent_set:
        selected_ends.append(ends[agent])
        
    unselected_ends = []
    for agent in unselected_agent_set:
        unselected_ends.append(ends[agent])
    
    selected_ends = np.array(selected_ends)
    unselected_ends = np.array(unselected_ends)
    
    # Distance between ai’s start and goal vertices.
    selected_dis = np.array([shortest_path_data[str(i)][0] for i in selected_agent_set])
    unselected_dis = np.array([shortest_path_data[str(i)][0] for i in unselected_agent_set])

    
    selected_goal_degree = np.array([map_degree[end[0], end[1]] for end in selected_ends])
    unselected_goal_degree = np.array([map_degree[end[0], end[1]] for end in unselected_ends])
    
    # dynamic feature
    # agent delay time
    agent_num = len(starts)
    agent_delay_ratio = np.zeros(agent_num)
    agent_delay = np.zeros(agent_num)
    for i in range(agent_num):
        if shortest_path_data[str(i)][0] == 0:
            import pdb; pdb.set_trace()
        agent_delay[i] =  len(paths[i]) - shortest_path_data[str(i)][0]
        agent_delay_ratio[i] = len(paths[i]) / shortest_path_data[str(i)][0]
    
    map_loc_heat_value = np.zeros(map_degree.shape)
    for one_path in paths:
        for one_loc in one_path:
            map_loc_heat_value[tuple(one_loc)] += 1
    
    # The minimum, maximum, sum and average of the heat values of the vertices on ai’s path pi
    agent_min_max_sum_avg_heat_value = np.zeros((agent_num, 4))
    for i in range(agent_num):
        path_heat_values = map_loc_heat_value[np.array(paths[i])[:, 0], np.array(paths[i])[:, 1]]
        agent_min_max_sum_avg_heat_value[i] = np.array([np.min(path_heat_values), np.max(path_heat_values), np.sum(path_heat_values), np.mean(path_heat_values)])
        
    # The number of time steps that ai is on a vertex with degree j (1 ≤ j ≤ 4).
    agent_degree_time = np.zeros((agent_num, 4))
    for i in range(agent_num):
        path_degrees = map_degree[np.array(paths[i])[:, 0], np.array(paths[i])[:, 1]]
        agent_degree_time[i] = np.array([np.sum(path_degrees==1), np.sum(path_degrees==2), np.sum(path_degrees==3), np.sum(path_degrees==4)])
    
    selected_delay = np.array(agent_delay[selected_agent_set])
    selected_delay_ratio = np.array(agent_delay_ratio[selected_agent_set])
    selected_agent_min_max_sum_avg_heat_value = np.array(agent_min_max_sum_avg_heat_value[selected_agent_set])
    selected_agent_degree_time = np.array(agent_degree_time[selected_agent_set])
    
    unselected_delay = np.array(agent_delay[unselected_agent_set])
    unselected_delay_ratio = np.array(agent_delay_ratio[unselected_agent_set])
    unselected_agent_min_max_sum_avg_heat_value = np.array(agent_min_max_sum_avg_heat_value[unselected_agent_set])
    unselected_agent_degree_time = np.array(agent_degree_time[unselected_agent_set])
    

    # Calculate features for selected and unselected data
    selected_array = np.concatenate([selected_dis.reshape(-1, 1), selected_starts, selected_ends, selected_goal_degree.reshape(-1, 1), selected_delay.reshape(-1, 1), selected_delay_ratio.reshape(-1, 1), selected_agent_min_max_sum_avg_heat_value, selected_agent_degree_time], axis=-1)
    selected_features = np.concatenate([np.min(selected_array, axis=0), np.max(selected_array, axis=0), np.sum(selected_array, axis=0), np.mean(selected_array, axis=0)], axis=-1)
    
    unselected_arrays = np.concatenate([unselected_dis.reshape(-1, 1), unselected_starts, unselected_ends, unselected_goal_degree.reshape(-1, 1), unselected_delay.reshape(-1, 1), unselected_delay_ratio.reshape(-1, 1), unselected_agent_min_max_sum_avg_heat_value, unselected_agent_degree_time], axis=-1)
    unselected_features = np.concatenate([np.min(unselected_arrays, axis=0), np.max(unselected_arrays, axis=0), np.sum(unselected_arrays, axis=0), np.mean(unselected_arrays, axis=0)], axis=-1)

    features = np.concatenate([selected_features, unselected_features], axis=-1)
    
    return tuple(selected_agent_set), list(features)


def svm_train(args):
    assert os.path.exists(args.gen_subset_exe), "gen_subset_exe does not exist: {}".format(args.gen_subset_exe)
    iteration = 0
    curr_state_list = args.initial_state
    svm_data_output_folder = os.path.join(args.svm_data_output_folder, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(svm_data_output_folder, exist_ok=True)
    iter_state_output_folder = os.path.join(args.iter_state_output_folder, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(iter_state_output_folder, exist_ok=True)
    
    map_name_list = [re.findall(r'map-(.*?)-scene', os.path.basename(curr_state_list[i]))[0] for i in range(len(curr_state_list))]
    assert len(set(map_name_list)) == 1, "All initial states should be from the same map, map_name_list: {}".format(map_name_list)
    
    file_name = os.path.basename(curr_state_list[0])
    map_name = re.findall(r'map-(.*?)-scene', file_name)[0]
    agent_num = re.findall(r'agent-(.*?).json', file_name)[0]
    map_path = os.path.join(args.map_folder, map_name + ".map")
    _, _, boolean_map = get_validLoc_bool_map(map_path)
    map_degree = get_map_degree(boolean_map)
    svm_train_data = []
    svm_train_data_path = os.path.join(svm_data_output_folder, "svm_{}_agent{}_train_temp.txt".format(map_name, agent_num))
    
    file_tabu = dict()
    for curr_state_path in curr_state_list:
        file_tabu[curr_state_path] = []
    
    train_qid = 1 
    
    # Start the subprocess
    if args.fix_adaptive_weight == []:
        adaptive_weight = [1,1,0] # will not be used if --destroyStrategy is not Adaptive
    else:
        adaptive_weight = args.fix_adaptive_weight
    cpp_command = "{} --map {} --agentNum {} --state {} --adaptive_weight {} --pprun {} --num_subset {} --uniform_neighbor 2 --replanTime {} --destroyStrategy {}".format(args.gen_subset_exe, map_path, agent_num, curr_state_path, " ".join([str(i) for i in adaptive_weight]), args.pprun,  args.num_subset, args.replan_time_limit, args.destroyStrategy)
    print("cpp_command: ", cpp_command)
    open_exe = subprocess.Popen(cpp_command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    while True:
        line = open_exe.stdout.readline() 
        if "CPP program" in line:
            break
        
    while iteration < args.max_iter:
        
        ##### Generate removal set #####
        time = datetime.datetime.now()
        results = []
        for one_state in curr_state_list:
            _, iterInfo, open_exe = cpp_removal_replanOpen(open_exe, state=one_state, adaptive_weight=adaptive_weight)
            results.append((one_state, iterInfo))
        
        files_subsetInfo = dict() # file_name : subset_info {removalSet_info : {removal_set : *** }, tabu_list : []}
        for result in results:
            files_subsetInfo[result[0]] = result[1]
            file_tabu[result[0]] = result[1]["tabu_list"]
        print("Time for generating removal set: ", datetime.datetime.now() - time) 
        
        ##### Generate SVM input data #####
        time = datetime.datetime.now() 
        svm_subset_feature = dict()
        label_list = [] # label for each removal set
        feature_list = [] # feature for each removal set
        qid_list = [] # qid for each removal set
        cost_list = [] # improvement for each removal set
        subsetLen_list = []
        for one_state_path in curr_state_list:
            
            file_name = os.path.basename(one_state_path).strip(".json").split("_iter")[0]
            scene_num = int(re.findall(r'scene-(.*?)-agent', file_name)[0])
            one_state = read_json(one_state_path)
            print("### Map {} Scene {} Iteration {} soc {}".format(map_name, scene_num, iteration, sum([len(one_state[i]) for i in one_state]) - len(one_state)))
            shortest_path = os.path.join(args.shortest_path_folder, file_name + ".json")
            assert os.path.exists(shortest_path), "shortest path file does not exist: {}".format(shortest_path)
            shortest_path_data = read_json(shortest_path)
            
            with Pool(args.num_cores) as pool:
                args_for_func = [(one_removal_set, one_state, shortest_path_data, map_degree) for one_removal_set in list(list(one_removal_set) for one_removal_set in files_subsetInfo[one_state_path]["removalSet_info"].keys())]
                results = pool.starmap(gen_svm_feature, args_for_func)
            for result in results:
                svm_subset_feature[result[0]] = result[1] 
            one_cost_list = [files_subsetInfo[one_state_path]["removalSet_info"][one_removal_set]["avg_improvement"] for one_removal_set in files_subsetInfo[one_state_path]["removalSet_info"].keys()]
            cost_list = cost_list + one_cost_list
            label_list = label_list + gen_label_from_cost(one_cost_list) # append new label to the end of the list
        
            # normalize feature
            one_feature_list = [svm_subset_feature[one_removal_set] for one_removal_set in files_subsetInfo[one_state_path]["removalSet_info"]]
            one_feature_list = np.array(one_feature_list)
            min_val = np.min(one_feature_list, axis=0)
            max_val = np.max(one_feature_list, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # Prevent division by zero if the range is 0
            one_feature_list = (one_feature_list - min_val) / range_val
            feature_list = feature_list + one_feature_list.tolist() # append new feature to the end of the list
            # qid (instance id)
            one_removal_set_len = len(files_subsetInfo[one_state_path]["removalSet_info"])
            subsetLen_list.append(one_removal_set_len)
            qid_list = qid_list + [train_qid for _ in range(one_removal_set_len)]
            train_qid += 1
        svm_save_path = os.path.join(svm_data_output_folder, "svm_{}_agent{}_iter{}.txt".format(map_name, agent_num, iteration))
        # save svm data
        save_svm_feature(label_list, feature_list, qid_list, save_path=svm_save_path, comment_list=cost_list)
        svm_train_data += read_txt(svm_save_path)
        save_txt(svm_train_data_path, svm_train_data)
        print("Time for generating svm feature: ", datetime.datetime.now() - time)

        ##### train the model #####
        model_save_path = os.path.join(svm_data_output_folder, "svm_{}_agent{}_iter{}.model".format(map_name, agent_num, iteration))
        train_svm_command = "{} {} {}".format(args.train_svm_command, svm_train_data_path, model_save_path) 
        print("train_svm_command: ", train_svm_command)
        
        os.system(train_svm_command)
        
        ##### infer the model #####
        svm_pred_save_path = os.path.join(svm_data_output_folder, "svm_{}_agent{}_iter{}.pred".format(map_name, agent_num, iteration))
        infer_svm_command = "{} {} {} {}".format(args.infer_svm_command, svm_save_path, model_save_path, svm_pred_save_path)
        print("infer_svm_command: ", infer_svm_command)
        os.system(infer_svm_command)
        
        ##### Update state #####
        predict_scores = read_txt(svm_pred_save_path)
        args_for_func = [(i, curr_state_list, predict_scores, files_subsetInfo, iter_state_output_folder, iteration, subsetLen_list, 1) for i in range(len(curr_state_list))] # 1 for topk, train with the best subset
        print("update states")
        with Pool(args.num_cores) as pool:
            results = pool.starmap(update_one_state, args_for_func)
        temp_curr_state_list = ["" for _ in results] # temp_curr_state_list will be the same order as previous curr_state_list
        for result in results:
            file_tabu[result[1]] = file_tabu[curr_state_list[result[0]]].copy()
            del file_tabu[curr_state_list[result[0]]]
            temp_curr_state_list[result[0]] = result[1]
        curr_state_list = temp_curr_state_list.copy()
        ### validation & renamd model with ranking score
        start_time = datetime.datetime.now()
        val_pred_save_path = os.path.join(svm_data_output_folder, "val_{}_agent{}_iter{}.pred".format(map_name, agent_num, iteration))
        infer_svm_command = "{} {} {} {}".format(args.infer_svm_command, args.val_data_path, model_save_path, val_pred_save_path)
        print("validation_svm_command: ", infer_svm_command)
        os.system(infer_svm_command)
        average_ranking_score = svm_get_avg_ranking(args.val_data_path, val_pred_save_path)
        model_save_path_with_score = os.path.join(svm_data_output_folder, "svm_{}_agent{}_iter{}_score{}.model".format(map_name, agent_num, iteration, "{:.2f}".format(average_ranking_score).replace(".", "_")))
        print("average_ranking_score: ", average_ranking_score, "rename model to: ", model_save_path_with_score)
        print("Time for validation and rename model: ", datetime.datetime.now() - start_time)

        iteration += 1


def svm_inference(args):
    
    # check the input parameters
    assert len(args.initial_state) == 1, "Only support one initial state for inference"
    assert os.path.exists(args.gen_subset_exe), "gen_subset_exe does not exist: {}".format(args.gen_subset_exe)
    
    
    # make the output folder
    initial_state_path = args.initial_state[0] # initial state path only support one state
    svm_data_output_folder = os.path.join(args.svm_data_output_folder, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(svm_data_output_folder, exist_ok=True)
    iter_state_output_folder = os.path.join(args.iter_state_output_folder, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(iter_state_output_folder, exist_ok=True)
    svm_log_path = None
    
    # process one state at a time
    if args.log_path != "" and ".log" in args.log_path:
        svm_log_path = args.log_path
        log_folder = os.path.dirname(svm_log_path)
        os.makedirs(log_folder, exist_ok=True)
        if os.path.exists(svm_log_path):
            os.remove(svm_log_path)

    # initialization
    iteration = 0
    file_name = os.path.basename(initial_state_path)
    map_name = re.findall(r'map-(.*?)-scene', file_name)[0]
    agent_num = re.findall(r'agent-(.*?).json', file_name)[0]
    map_path = os.path.join(args.map_folder, map_name + ".map")
    curr_state_path = initial_state_path
    shortest_path = os.path.join(args.shortest_path_folder, file_name)
    assert os.path.exists(shortest_path), "shortest path file does not exist: {}".format(shortest_path)
    shortest_path_data = read_json(shortest_path)
    _, _, boolean_map = get_validLoc_bool_map(map_path)
    map_degree = get_map_degree(boolean_map)
    iterInfo = {"tabu_list": []}
    curr_state = read_json(curr_state_path)
    curr_state = {int(k): v for k, v in curr_state.items()}
    val_qid = 1
    sum_removal_time = 0
    sum_replan_time = 0
    
    if args.fix_adaptive_weight != []:
        adaptive_weight = args.fix_adaptive_weight
    else:
        adaptive_weight = [1,1,0]
        
    runtime_limit = args.infer_time
    sum_run_time = 0
    sum_program_time = 0
    infer_tag = ""
    if args.num_subset == 1:
        infer_tag += "oneSubset"
    
    # set up the formatted log file
    if svm_log_path is not None:
        with open(svm_log_path, "a") as f:
            f.write("### init soc {}\n".format(sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)))
    
    # Start the subprocess
    cpp_command = "{} --map {} --agentNum {} --state {} --adaptive_weight {} --pprun 1 --num_subset {} --uniform_neighbor {} --replanTime {} --destroyStrategy {} --neighborSize {}".format(args.gen_subset_exe, map_path, agent_num, curr_state_path, " ".join([str(i) for i in adaptive_weight]), args.num_subset,args.uniformNB,   args.replan_time_limit, args.destroyStrategy, args.neighborSize)
    print("cpp_command: ", cpp_command)
    open_exe = subprocess.Popen(cpp_command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    while True:
        line = open_exe.stdout.readline() 
        print(line) 
        if "CPP program" in line:
            break
        
    # Start the inference process
    while sum_removal_time + sum_replan_time < runtime_limit:
        
        program_start_time = datetime.datetime.now()
        print("### Iteration: {} init soc {}".format(iteration, sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)))
        
        ##### Generate removal set #####
        _, iterInfo, open_exe = cpp_removal_replanOpen(exe=open_exe, state=curr_state_path, adaptive_weight=adaptive_weight)
        
        ##### Generate SVM input data #####
        svm_subset_feature = dict()
        with Pool(args.num_cores) as pool:
            args_for_func = [(i, curr_state, shortest_path_data, map_degree) for i in list(list(i) for i in iterInfo["removalSet_info"].keys())]
            results = pool.starmap(gen_svm_feature, args_for_func)
        for result in results:
            svm_subset_feature[result[0]] = result[1] 
        cost_list = [iterInfo["removalSet_info"][i]["avg_improvement"] for i in iterInfo["removalSet_info"].keys()]
        removal_strategy_list = [iterInfo["removalSet_info"][i]["destroy_strategy"] for i in iterInfo["removalSet_info"].keys()]
        label_list = gen_label_from_cost(cost_list)
        
        # normalize feature
        feature_list = [svm_subset_feature[i] for i in iterInfo["removalSet_info"]]
        feature_list = np.array(feature_list)
        min_val = np.min(feature_list, axis=0)
        max_val = np.max(feature_list, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Prevent division by zero if the range is 0
        feature_list = (feature_list - min_val) / range_val
        # qid (instance id)
        qid_list = [val_qid for _ in range(len(iterInfo["removalSet_info"]))]
        val_qid += 1
        svm_save_path = os.path.join(svm_data_output_folder, "svm_{}_{}infer.txt".format(file_name.split(".")[0], infer_tag))
        # save svm data
        save_svm_feature(label_list, feature_list, qid_list, save_path=svm_save_path, comment_list=cost_list)
        
        ##### infer the model #####
        svm_pred_save_path = os.path.join(svm_data_output_folder, "svm_{}_{}infer.pred".format(file_name.split(".")[0], infer_tag))
        infer_svm_command = "{} {} {} {}".format(args.infer_svm_command, svm_save_path, args.svm_model_path, svm_pred_save_path)
        print("infer_svm_command: ", infer_svm_command) 
        infer_start_time = datetime.datetime.now()            
        os.system(infer_svm_command)
        infer_time = (datetime.datetime.now() - infer_start_time).total_seconds()

        ##### Update state #####
        subset_list = [i for i in iterInfo["removalSet_info"].keys()]
        pred_score_list = read_txt(svm_pred_save_path)
        curr_state, pp_replan_time_list, pp_removal_time, adaptive_weight, trial = update_state_weight_based_on_ranking_score(removal_strategy_list, pred_score_list, curr_state, iterInfo["removalSet_info"], subset_list, adaptive_weight=adaptive_weight, topk=len(pred_score_list), destroyStrategy=args.destroyStrategy, replan_time_limit=args.replan_time_limit)
        pp_replan_time = sum(pp_replan_time_list)
        curr_state_path = os.path.join(iter_state_output_folder, "{}_{}infer_curr_state.json".format(file_name.split(".")[0], infer_tag))
        save_json(curr_state_path, curr_state)
        program_run_time = (datetime.datetime.now() - program_start_time).total_seconds()
        
        sum_removal_time += pp_removal_time
        sum_removal_time += infer_time
        sum_replan_time += pp_replan_time
        svm_remain_time = runtime_limit - (sum_removal_time + sum_replan_time)
        sum_run_time = sum_removal_time + sum_replan_time 
        sum_program_time += program_run_time
        print("### Iteration: {} soc {} sum_run_time {:.6f} sum_program_time {:.6f} pp_replan_time {:.6f} pp_removal_time {:.6f} infer_time {:.6f} remain_time {:.6f} ".format(iteration, sum([len(curr_state[i]) for i in curr_state]) - len(curr_state), sum_run_time, sum_program_time, pp_replan_time, pp_removal_time, infer_time, svm_remain_time))
        if svm_log_path is not None and iteration % args.log_iter == 0:
            current_soc = sum([len(curr_state[i]) for i in curr_state]) - len(curr_state)
            with open(svm_log_path, "a") as f:
                f.write("### Iteration: {} soc {} sum_run_time {:.6f} sum_program_time {:.6f} pp_replan_time {:.6f} pp_removal_time {:.6f} infer_time {:.6f} remain_time {:.6f} \n".format(iteration, current_soc, sum_run_time, sum_program_time, pp_replan_time, pp_removal_time, infer_time, svm_remain_time))
            print("update state and log: ", svm_log_path)
            
        iteration += trial


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--option", type=str, default="train", help="(1) train svm (2) svm inference")
    # input config
    parser.add_argument("--initial_state", type=str, nargs='+', default=[], help="initial state for training and inference")
    parser.add_argument("--val_data_path", type=str, default="", help="validation data path")
    parser.add_argument("--svm_model_path", type=str, default="", help="path to the svm model")
    parser.add_argument("--map_folder", type=str, default="data/map", help="folder containing map downloaded from the MAPF benchmark")
    parser.add_argument("--scene_folder", type=str, default="data/scene", help="folder containing scene downloaded from the MAPF benchmark")
    parser.add_argument("--shortest_path_folder", type=str, required=True, help="folder containing shortest path data")
    # output config
    parser.add_argument("--iter_state_output_folder", type=str, default="output/svm_temp_state", help="folder to store iteration state for training and inference")
    parser.add_argument("--svm_data_output_folder", type=str, default="output/svm_val_data", help="folder to save temporary collected training data")
    parser.add_argument("--log_path", type=str, default="output/svm_log.log", help="path to save the log")
    # LNS parameter 
    parser.add_argument("--pprun", type=int, default=6, help="number of PP replan")
    parser.add_argument("--max_iter", type=int, default=100, help="maximum iteration for training")
    parser.add_argument("--infer_time", type=float, default=300, help="maximum time for inference")
    parser.add_argument("--replan_time_limit", type=float, default=0.6, help="replan time limit for each PP replan")
    parser.add_argument("--num_subset", type=int, default=20, help="number of subset for each state")
    parser.add_argument("--fix_adaptive_weight", type=int,  nargs='+', default=[], help="fix adaptive weight for Adaptive strategy")
    parser.add_argument("--destroyStrategy", type=str, default=None, help="LNS destroy strategy")
    parser.add_argument("--uniformNB", type=int, default=0, help="(0) fixed nb_size specified by --neighborSize (1) nb_size sample from {2,4,8,16,32} (2) nb_size sample from 5~16")
    parser.add_argument("--neighborSize", type=int, default=25, help="neighbor size")
    # path to exe
    parser.add_argument("--gen_subset_exe", type=str, default="", help="path to the cpp program to generate subset")
    parser.add_argument('--train_svm_command', type=str, default="exe/svm_rank/svm_rank_learn -c 0.1 -t 0 -e 0.01 -d 2 -s 1 -r 1 -l 2", help='svmrank command to run to train')
    parser.add_argument('--infer_svm_command', type=str, default="exe/svm_rank/svm_rank_classify",help='svmrank command to inference')
    # python program config
    parser.add_argument("--num_cores", type=int, default=1, help="number of cores for parallel processing")
    parser.add_argument("--overwrite", type=int, default=0, help="overwrite the existing file")
    parser.add_argument("--log_iter", type=int, default=1, help="gap for iteration to log")
    

    
    args = parser.parse_args()
    if args.option == "train":
        svm_train(args)
    elif args.option == "infer":
        svm_inference(args)
