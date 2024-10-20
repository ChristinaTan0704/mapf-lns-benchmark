import os
import numpy as np
from utils import *
import argparse

def generate_random_scene_from_scen(args):
    import random
    
    def get_agnet_path_from_txt(file_path):
        lines = read_txt(file_path)
        agent_path = dict() 
        for line in lines:
            if "Agent" in line:
                agnet_id = int(re.findall("Agent (.+):", line)[0])
                path = re.findall("\((\d+),(\d+)\)", line)
                agent_path[agnet_id] = [(int(x), int(y)) for x, y in path]
        return agent_path
    

    def gen_one_scene(scene_sg, save_path):
        pick_scene = np.random.choice(list(scene_sg.keys()))
        starts, goals = scene_sg[pick_scene]
        pick_start = random.sample(starts, agent_num)
        pick_goal = random.sample(goals, agent_num)
        gen_scene_info = ["version 1"]
        for i in range(agent_num):
            gen_scene_info.append("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0".format(i+1,map_name, map_col, map_row, pick_start[i][1], pick_start[i][0], pick_goal[i][1], pick_goal[i][0]))
        save_txt(save_path, gen_scene_info)

    assert os.path.exists(args.map_folder)
    assert os.path.exists(args.scene_folder)
    assert os.path.exists(args.lns2_exe)

    map_folder = args.map_folder
    scene_folder = args.scene_folder
    log_output_folder = os.path.join(args.output_folder, "log")
    json_output_folder = os.path.join(args.output_folder, "state_json")   
    output_scene_folder =  os.path.join(args.output_folder, "scene")
    os.makedirs(output_scene_folder, exist_ok=True)
    os.makedirs(json_output_folder, exist_ok=True)
    os.makedirs(log_output_folder, exist_ok=True)
    print("make folder : ", log_output_folder)

    map_name = os.path.basename(args.map_path)

    scene_sg = dict()
    for idx in range(1,26,1):
        scene_file_name = "{}-random-{}.scen".format(map_name.strip(".map"), idx)
        scene_info = read_txt(os.path.join(scene_folder, scene_file_name))
        starts = []
        goals = []
        for line in scene_info:
            if ".map" in line:
                _, _, map_col, map_row, start_col, start_row, goal_col, goal_row, _ = line.strip("\t").split()
                starts.append((int(start_row), int(start_col)))
                goals.append((int(goal_row), int(goal_col)))
        scene_sg[idx] = [starts, goals]
        
    agent_num = args.agent_num
    for scene_id in range(args.scene_start_idx, args.scene_end_idx, 1):
        txt_path = os.path.join(log_output_folder, "map-{}-scene-{}-agent-{}.txt".format(map_name.strip(".map"), scene_id, agent_num))
        if os.path.exists(txt_path) and not args.overwrite:
            print("scene already exists, skip : ", txt_path)
            continue
        gen_scene_name = "{}-random-{}_randomGen.scen".format(map_name.strip(".map") , scene_id)
        random_gen_scene_path = os.path.join(output_scene_folder, gen_scene_name)
        
        lns2_command = "{} -m {} -a {} -k {} -t 10 --outputPaths={}".format(args.lns2_exe, os.path.join(map_folder, map_name), random_gen_scene_path, agent_num, txt_path)
        print("exe_command : ", lns2_command)
        lns2_success = False
        while not lns2_success:
            gen_one_scene(scene_sg, random_gen_scene_path)
            os.system(lns2_command)
            if os.path.exists(txt_path):
                lns2_success = True
                print("lns2 success generate random scene : ", txt_path, random_gen_scene_path)
            
                agent_path = get_agnet_path_from_txt(txt_path)
                if len(agent_path) > 0:
                    agent_path_json = os.path.join(json_output_folder, os.path.basename(txt_path).replace(".txt", ".json"))
                    print("converted {} to {}".format(txt_path, agent_path_json))
                    save_json(agent_path_json, agent_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument("--output_folder", type=str, default="output/generated_scenes")
    parser.add_argument("--map_path", type=str, default="data/example/random-32-32-20.map")
    parser.add_argument("--scene_folder", type=str, default="data/scene")
    parser.add_argument("--map_folder", type=str, default="data/map")
    parser.add_argument("--agent_num", type=int, default=100)
    parser.add_argument("--scene_start_idx", type=int, default=0, help="start seed is included")
    parser.add_argument("--scene_end_idx", type=int, default=1, help="end seed is not included")
    parser.add_argument("--overwrite", type=int, default=0, help="overwrite existing scene")
    parser.add_argument("--lns2_exe", type=str, default="exe/orig-lns2/lns")

    
    args = parser.parse_args()
    
    generate_random_scene_from_scen(args)
  

        
    

    