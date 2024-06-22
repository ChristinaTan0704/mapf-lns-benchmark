import os
import datetime
import argparse
from utils import *
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



def cpp_removal_replanOpen(exe, state="", adaptive_weight=[1,1,0]):
    """
    Call the C++ executable to generate the subset, NOTE : input one state at a time
    Required input:
    --exe_path: path to the executable file
    --state: state to be updated
    --adaptive_weight: weight for adaptive removal
    """
    succ = False
    while not succ:
        cpp_input = "--state {} --adaptive_weight {} \n".format(state, " ".join([str(i) for i in adaptive_weight]))
        
        print("cpp_input: ", cpp_input)
        # To send data to the C++ process
        exe.stdin.write(cpp_input)
        exe.stdin.flush()

        # To read a line from C++ process
        cpp_output = []
        while True:
            line = exe.stdout.readline()
            cpp_output.append(line)
            if "CPP program" in line:
                break
        cpp_output = [line.strip() for line in cpp_output]
        
        # parse file_info to get removal set, removal time, average improvement, and new paths for each subset
        info_list = []
        one_info = []
        tabu_list = []
        for line in cpp_output:
            if "one removal end" in line:
                info_list.append(one_info.copy())
                one_info = []
            else:
                one_info.append(line)
            if "tabu_list:" in line and "tabu_list:" != line:
                tabu_list = [int(i) for i in re.findall(r'tabu_list: (.*?)$', line)[0].split()]
                
        iterInfo = {"tabu_list": tabu_list}
        iterInfo["removalSet_info"] = dict()
        
        for one_info in info_list:
            one_info_dict = dict()
            one_info_dict["agent_paths"] = []
            for line_idx in range(len(one_info)):
                if "removal_time:" in one_info[line_idx]:
                    one_info_dict["removal_time"] = float(re.findall(r'removal_time: (.*?)$', one_info[line_idx])[0])
                if "average_replan_time:" in one_info[line_idx]:
                    one_info_dict["average_replan_time"] = float(re.findall(r'average_replan_time: (.*?)$', one_info[line_idx])[0])
                if "destroy_strategy:" in one_info[line_idx]:
                    one_info_dict["destroy_strategy"] = int(re.findall(r'destroy_strategy: (.*?)$', one_info[line_idx])[0])
                if "removal_set:" in one_info[line_idx]:
                    try:
                        one_info_dict["removal_set"] = tuple(sorted([int(i) for i in re.findall(r'removal_set: (.*?)$', one_info[line_idx])[0].split()]))
                    except:
                        print("error in removal_set: ", one_info[line_idx])
                        import pdb; pdb.set_trace() 
                if "average_improvement:" in one_info[line_idx]:
                    one_info_dict["avg_improvement"] = float(re.findall(r'average_improvement: (.*?)$', one_info[line_idx])[0])
                if "new paths start" in one_info[line_idx]:
                    agnet_paths = defaultdict(list)
                    while "new paths end" not in one_info[line_idx]:
                        line_idx += 1
                        if "new paths end" in one_info[line_idx]:
                            break
                        if "agent" in one_info[line_idx]:
                            agent_id = int(re.findall(r'agent (.*?) ', one_info[line_idx])[0])
                            path = re.findall("\((\d+),(\d+)\)", one_info[line_idx])
                            path = [(int(i[0]), int(i[1])) for i in path]
                            agnet_paths[agent_id] = path
                    one_info_dict["agent_paths"] = agnet_paths
            if one_info_dict["removal_set"] not in iterInfo["removalSet_info"]:
                iterInfo["removalSet_info"].update({one_info_dict["removal_set"]: {"removal_time": one_info_dict["removal_time"], "avg_improvement": one_info_dict["avg_improvement"], "agent_paths": one_info_dict["agent_paths"], "average_replan_time": one_info_dict["average_replan_time"], "destroy_strategy": one_info_dict["destroy_strategy"]}})
            else:
                print("generated the same subset", one_info_dict["removal_set"])
        if len(iterInfo["removalSet_info"]) >= 1:
            succ = True
        else:
            succ = False
            print("subset number is 0, regenerate the subset")
            print(cpp_output)
            import pdb; pdb.set_trace()
    
    return state, iterInfo, exe
  