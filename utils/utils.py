import json
import torch
import numpy as np
from collections import defaultdict


def get_idx_from_agentPaths(agent_paths=None, spatial_pool=4, temporal_pool=4, cutoff=100, map_w=100, map_h=100):
    
    if spatial_pool != 1:
        feature_w = int(np.round(map_w / spatial_pool)) + 1
        feature_h = int(np.round(map_h / spatial_pool)) + 1
    else:
        feature_w = map_w
        feature_h = map_h
    
    if temporal_pool != 1:
        feature_t = int(np.round(cutoff / temporal_pool)) + 1
    else:
        feature_t = cutoff
    
    agent_num = len(agent_paths)
    agent_locs = np.zeros((agent_num, feature_t)) - 1 # -1 means there's no agent at the location
    time_locs = defaultdict(list)
    for agent, path in agent_paths.items():
        for t, one_loc in enumerate(path):
            t_ = int(np.round(t / temporal_pool))
            if t_ >= feature_t:
                break
            time_locs[t_].append(one_loc)
            x, y = one_loc
            x = int(np.round(x / spatial_pool))
            y = int(np.round(y / spatial_pool))
            agent_locs[int(agent)][t_] = x*feature_w + y
    
    flatMap_t = np.zeros((feature_w*feature_h , feature_t)) # 0 means the location is not occupied 
    for t_, loc_list in time_locs.items():
        for loc in loc_list:
            x, y = loc
            x = int(np.round(x / spatial_pool))
            y = int(np.round(y / spatial_pool))
            flatMap_t[x*feature_w + y, t_] = 1
    
    agent_locs_mask = agent_locs == -1 # For a binary mask, a True value indicates that the corresponding position is not allowed to attend
    return flatMap_t, agent_locs, agent_locs_mask 


def downsample_obstacle_map(obstacle_map=None, spatial_pool=4):
    """_summary_

    Args:
        obstacle_map : map_w * map_h; True for obstacle, False for free space
        spatial_pool : downsample the map by spatial_pool
    Returns:
        downsampled + flattened, obstacle map; shape : (feature_w * feature_h, 1)
    """
    
    
    from scipy.ndimage import maximum_filter
    pooled_map = maximum_filter(obstacle_map, size=spatial_pool, mode='constant', cval=0)
    pooled_map = pooled_map[::spatial_pool, ::spatial_pool]
    
    return np.expand_dims(pooled_map.flatten(), axis=-1)



def rank_list(input_list):
    arr = np.array(input_list)
    
    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(-arr)  # Multiply by -1 for descending order
    
    # Create an array to store the ranks
    ranks = np.empty_like(sorted_indices)
    
    # Assign ranks, starting from 1; handle ties by checking successive elements
    ranks[sorted_indices] = np.arange(1, len(arr) + 1)
    
    # To handle ties: where values are the same, ranks should be the same
    unique_values, index_first, count = np.unique(arr, return_index=True, return_counts=True, axis=0)
    for i, (first, cnt) in enumerate(zip(index_first, count)):
        if cnt > 1:  # More than one occurrence means a tie
            rank = ranks[first]  # Rank of the first occurrence
            indices = np.where(arr == unique_values[i])[0]
            ranks[indices] = rank  # Apply this rank to all tied elements
    
    return ranks.tolist()


def get_validLoc_bool_map(file_path): 
    lines = read_txt(file_path)
    width = int(lines[2].split()[1])
    height = int(lines[1].split()[1])
    # Extract the obstacle locations from the file
    map_data = [list(line) for line in lines if line.startswith('.') or line.startswith('@') or line.startswith('T')]
    bool_map = [[True if c == '.' else False for c in row] for row in map_data]
    bool_map = np.array(bool_map)
    return width, height, bool_map

def get_map_degree(map):
    height, width = map.shape  # Assuming 'map' is a 2D array
    map_degree = np.zeros(map.shape)
    for i in range(map_degree.shape[0]):
        for j in range(map_degree.shape[1]):
            
            if not map[i, j]:
                map_degree[i, j] = -1
                continue
            # Check boundaries before accessing the map
            indices = [(i, j + 1), (i, j - 1),
                    (i + 1, j), (i - 1, j)]
            indices = [(x, y) for x, y in indices if 0 <= x < height and 0 <= y < width]
            map_degree[i, j] = np.sum(map[x, y] for x, y in indices)
    return map_degree


class GenericConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Config:
    def __init__(self, input, network):
        self.input = GenericConfig(**input)
        self.network = GenericConfig(**network)

def load_config(filepath):
    import yaml
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
        return Config(**data)


def save_json(save_path, data, log=True):
    import json
    out_json = json.dumps(data, sort_keys=False, indent=4, separators=(',', ':'),
                          ensure_ascii=False)
    with open(save_path, "w") as fo:
        fo.write(out_json)
        fo.close()
        if log:
            print("json file saved to : ", save_path)


def get_avg_rank(predictions, targets):
    # Ensure the shapes are correct
    assert predictions.shape == targets.shape, "Shapes of predictions and targets must match"

    # Remove the last dimension for simplicity
    predictions = predictions.squeeze(-1)
    targets = targets.squeeze(-1)

    # Get the indices of the highest values in predictions
    highest_pred_values, highest_pred_indices = predictions.max(dim=1, keepdim=True)
    
    # Sort targets and get the ranks
    sorted_indices = torch.argsort(targets, dim=1, descending=True)
    ranks = torch.zeros(predictions.size(0), dtype=torch.int32, device=predictions.device)
    
    for i in range(targets.size(0)):
        # Get indices of all items in predictions that have the highest value
        highest_pred_indices = (predictions[i] == highest_pred_values[i]).nonzero(as_tuple=False).view(-1)
        
        # Find the ranks of these highest value items in sorted targets
        rank_list = []
        for idx in highest_pred_indices:
            rank = (sorted_indices[i] == idx).nonzero(as_tuple=True)[0].item()
            rank_list.append(rank)
        
        # Store the minimum rank (highest priority rank) among the tied items
        ranks[i] = min(rank_list) + 1  # Adding 1 to convert from zero-based index to rank
    
    return ranks.float().mean().item()


def pairwise_hinge_ranking_loss(predictions, targets):

    """
        predictions : shape (batch_size, num_subsets, 1)
        targets : shape  (batch_size, num_subsets, 1)
    """
    # Compute pairwise differences
    delta_predictions = predictions - predictions.transpose(1, 2)  # Shape: (batch_size, num_items, num_items)
    delta_targets = targets - targets.transpose(1, 2)  # Shape: (batch_size, num_items, num_items)

    # Calculate pairwise hinge ranking loss
    loss_positive = torch.relu(1 - delta_predictions) * (delta_targets > 0).float()
    loss_negative = torch.relu(1 + delta_predictions) * (delta_targets < 0).float()

    # Combine positive and negative loss
    loss = loss_positive + loss_negative

    # Sum and normalize the loss by the number of comparisons
    loss = loss.sum() / (predictions.size(0) * predictions.size(1) * (predictions.size(1) - 1))

    return loss


def tail(file_path, lines_to_read=10):
    """Read the last few lines of a very large file without loading it entirely."""
    command = "cat {} | tail -n {}".format(file_path, lines_to_read)
    return os.popen(command).read().split("\n")


def read_json(json_path):
    with open(json_path, 'r') as j:
        info = json.loads(j.read())
    return info


def read_txt(txt_path):
    with open(txt_path, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def save_txt(save_path, data, log=True):
    with open(save_path, "w") as file:
        for info in data:
            file.write(str(info))
            file.write("\n")
    if log:
        print("txt saved to : ", save_path, len(data))