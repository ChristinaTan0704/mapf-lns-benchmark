import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = self.kdim = self.vdim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
        self.q, self.k, self.v = None, None, None

    def forward(self, x, attn_mask=None):
        B, T, embed_dim = x.shape
        H, head_dim = self.num_heads, self.head_dim
        q, k, v = [y.view(B, T, H, head_dim) for y in F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)]
        attn = torch.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(head_dim)
        if attn_mask is not None:
            attn += attn_mask.view(B, 1, 1, T)
        attn = attn.softmax(dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.einsum('bhtT,bThd->bthd', attn, v)
        return self.out_proj(out.reshape(B, T, embed_dim))


class TransformerEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, feedforward_factor, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)

        self.dim_feedforward = feedforward_factor * d_model
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        if self.dim_feedforward > 0:
            self.linear1 = nn.Linear(d_model, self.dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(self.dim_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = activation

    def forward(self, x, attn_mask=None):
        return self.feedforward_block(self.self_attn_block(x, attn_mask=attn_mask))
    
    def self_attn_block(self, x, attn_mask=None):
        return self.norm1(x + self.dropout1(self.self_attn(x, attn_mask=attn_mask)))
    
    def feedforward_block(self, x):
        return self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))) if self.dim_feedforward > 0 else x


# Define a simple linear model
class NNS(nn.Module):
    def __init__(self, config=None):
        super(NNS, self).__init__()
        assert config is not None, "Config is required"
        self.input_dim = config.network.in_channels[0]
        self.feature_w = config.input.feature_w
        self.feature_h = config.input.feature_h
        self.feature_ = self.feature_w * self.feature_h
        self.feature_t = config.input.feature_t
        in_channels = config.network.in_channels
        out_channels = config.network.out_channels
        self.in_channels = in_channels
        self.subset_agent_max = config.network.subset_agent_max
        
        # learnable embeddings
        self.curr_occupy_loc_embedding = nn.Embedding(config.input.feature_t, self.input_dim//2) # 
        self.shortest_occupy_loc_embedding = nn.Embedding(config.input.feature_t, self.input_dim//2)
        self.obstacle_occupy_loc_embedding = nn.Embedding(1, self.input_dim)
        self.position_embedding = nn.Embedding(self.feature_, self.input_dim)

        # intra-path attention
        self.convs = nn.ModuleList([nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, padding=1),
                nn.BatchNorm3d(out_channel, momentum=0.2),
                nn.ReLU()) for in_channel, out_channel in zip(in_channels, out_channels)])

        self.attns = nn.ModuleList([TransformerEncoderLayer(dim, config.network.num_heads, config.network.feed_forward_factor) 
                    for dim in out_channels])
        
        # subset attention
        self.sub_attn_embedding = nn.Embedding(1, out_channels[-1])
        self.sub_attns = TransformerEncoder([TransformerEncoderLayer(out_channels[-1], config.network.sub_num_heads, config.network.sub_feed_forward_factor) for _ in range(config.network.sub_layers)])

        # score prediction
        self.out_fc = nn.Linear(out_channels[-1], 1)
    
    def forward(self, batch_data):
        
        """
        batch_data:
            curr_paths : shape [batch_size, flatten_feature_map_size, self.feature_t], if curr_paths_embedding[batch_idx, loc_idx, time_idx] = 1;
                            the location in current path is occupied at time_idx.
            agent_paths_idx : shape [batch_size, num_agents, self.feature_t, 1]
            agent_paths_mask : shape [batch_size, num_agents, self.feature_t], if agent_paths_mask[batch_idx, agent_idx, time_idx] = 1;
                            the location in agent_paths[batch_idx, agent_idx, time_idx] is occupied.
            shortest_path : shape [batch_size, flatten_feature_map_size, self.feature_t], if shortest_paths_embedding[batch_idx, loc_idx, time_idx] = 1;
                            the location in shortest path is occupied at time_idx.
            obstacle : shape [batch_size, flatten_feature_map_size, 1], if obstacle_embedding[batch_idx, loc_idx, 0] = 1;
                            the location in obstacle is occupied.
            subagents_idx : shape [batch_size, subset_num, self.subset_agent_max], the indices of the agents in each subset.
            subset_mask : shape [batch_size, subset_num, self.subset_agent_max], if subset_mask[batch_idx, subset_idx, agent_idx] = 1;
                            there is an agent in subagents_idx[batch_idx, subset_idx, agent_idx] is in the subset.
        """
        
        curr_paths, agent_paths_idx, agent_paths_mask, shortest_path, obstacle, subagents_idx, subset_mask = batch_data
        batch_size, subset_num, subset_agent_num, = subagents_idx.shape
        _, num_agents, _ = agent_paths_idx.shape
        device = curr_paths.device
        assert subset_agent_num == self.subset_agent_max, "padded subset size doesn't match subset_num {} vs self.subset_agent_max {}".format(subset_agent_num, self.subset_agent_max)

        curr_paths_idx = curr_paths.nonzero(as_tuple=False)
        shortest_path_idx = shortest_path.nonzero(as_tuple=False)
        obstacle_idx = obstacle.nonzero(as_tuple=False)  
        
        def make_grid(occupy_loc_indices=None, occupy_loc_embeddings=None, max_time=48, dim=128):
            """
            occupy_loc_indices : [batch_size * self.feature_ * max_time, 3];  the indices of the location that's occupied at occupy_loc_indices[i] (occupy_loc_indices[i] = [batch_idx, loc_idx, time_idx])
            occupy_loc_embeddings : [batch_size * self.feature_ * max_time, dim//2], the embedding of the location that's occupied at occupy_loc_indices[i] 
            transform location idx to a grid map [batch_size, self.feature_, max_time, dim]
            """
            nonlocal batch_size
            linear_indices = occupy_loc_indices[:, 0] * (self.feature_ * max_time) + occupy_loc_indices[:, 1] * max_time + occupy_loc_indices[:, 2]
            linear_indices = linear_indices.unsqueeze(-1).expand(-1, occupy_loc_embeddings.size(-1))
            grid_embedding = torch.zeros((batch_size * self.feature_ * max_time, dim), requires_grad=True, dtype=torch.float32, device=device)
            return grid_embedding.scatter_add(0, linear_indices, occupy_loc_embeddings).view(batch_size, self.feature_, max_time, dim)
        
        ### construct the learnable embeddings based on occpancy
        curr_paths_embedding = make_grid(occupy_loc_indices=curr_paths_idx, occupy_loc_embeddings=self.curr_occupy_loc_embedding.weight[curr_paths_idx[:, 2]].to(device), max_time=self.feature_t, dim=self.input_dim//2) # curr_paths_embedding : shape [batch_size, self.feature_, self.feature_t, dim//2], the embedding of the location that's occupied at time_idx;
        shortest_paths_embedding = make_grid(occupy_loc_indices=shortest_path_idx, occupy_loc_embeddings=self.shortest_occupy_loc_embedding.weight[shortest_path_idx[:, 2]].to(device), max_time=self.feature_t, dim=self.input_dim//2)
        obstacle_embedding = make_grid(occupy_loc_indices=obstacle_idx, occupy_loc_embeddings=self.obstacle_occupy_loc_embedding.weight[obstacle_idx[:, 2]].to(device), max_time=1, dim=self.input_dim)
        obstacle_embedding = obstacle_embedding.expand(-1, -1, self.feature_t, -1)
        position_embedding = self.position_embedding.weight.view(1, self.feature_, 1, self.input_dim).expand(batch_size, -1, self.feature_t, -1)
        
        map_embedding = torch.cat([curr_paths_embedding, shortest_paths_embedding], dim=-1) + obstacle_embedding + position_embedding
        for layer_idx, (one_conv, one_attn) in enumerate(zip(self.convs, self.attns)):

            ### shared 3D convolution
            map_embedding = map_embedding.view(batch_size, self.feature_w, self.feature_h, self.feature_t, self.in_channels[layer_idx])
            map_embedding = one_conv(map_embedding.permute(0, 4, 3, 1, 2)).permute(0, 3, 4, 2, 1).view(batch_size, self.feature_, self.feature_t, -1)
            
            batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(batch_size, num_agents, self.feature_t)
            time_indices = torch.arange(self.feature_t, device=device).view(1, 1, -1).expand(batch_size, num_agents, self.feature_t)  
            extracted_agent_paths_embedding = map_embedding[batch_indices, agent_paths_idx.squeeze(-1), time_indices]
            extracted_agent_paths_embedding = extracted_agent_paths_embedding.view(batch_size * num_agents, self.feature_t, -1)
            
            ### intra-path attention
            att_mask = agent_paths_mask.view(batch_size * num_agents, self.feature_t)
            if layer_idx < len(self.convs) - 1:
                dim = self.in_channels[layer_idx + 1]
                extracted_agent_paths_embedding = one_attn.feedforward_block(one_attn.self_attn_block(extracted_agent_paths_embedding, att_mask)[~att_mask]) # batchAgentLoc x dim
            
                mask_true = (~agent_paths_mask).nonzero(as_tuple=False)
                loc_indices = agent_paths_idx[mask_true[:, 0], mask_true[:, 1], mask_true[:, 2]].squeeze(-1)
                linear_indices = mask_true[:, 0] * (self.feature_ * self.feature_t) + loc_indices * self.feature_t + mask_true[:, 2]
                map_embedding = map_embedding.reshape(-1, dim)
                map_embedding.scatter_add_(0, linear_indices.unsqueeze(1).expand(-1, dim), extracted_agent_paths_embedding)
            else:
                dim = extracted_agent_paths_embedding.shape[-1]
                extracted_agent_paths_embedding = one_attn.self_attn_block(extracted_agent_paths_embedding, att_mask)
        
        ### subset attention
        extracted_agent_paths_embedding = extracted_agent_paths_embedding.view(batch_size, num_agents, self.feature_t, -1)
        attn_token = self.sub_attn_embedding.weight.view(1, 1, 1, dim).expand(batch_size, subset_num, 1, dim)
        
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1)
        subset_embedding = extracted_agent_paths_embedding[:, :, 0, :][batch_indices, subagents_idx] 
        subset_embedding = torch.cat([attn_token, subset_embedding], dim=2).view(batch_size * subset_num, self.subset_agent_max + 1, dim)
        
        att_mask = torch.cat([torch.zeros((batch_size, subset_num, 1), dtype=torch.bool, device=device), subset_mask], dim=2).view(batch_size * subset_num, -1)
        subset_embedding = self.sub_attns(subset_embedding, att_mask)[:,0]
        
        ### score prediction
        scores = self.out_fc(subset_embedding).view(batch_size, subset_num)
        return scores.unsqueeze(2)  # Shape: (batch_size, num_subsets, 1)