import torch
from torch.utils.data import DataLoader
from torch import optim
from nns.dataset import NssDataset
from nns.model import NNS
from utils import *
import argparse
import datetime
import numpy as np
from tensorboardX import SummaryWriter
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil
from collections import OrderedDict
    

def train_nns(args):
    assert args.training_input != "", "Please provide the path to the training data"
    config_name = os.path.basename(args.config).split(".")[0]
    output_folder = os.path.join(args.output_folder, config_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
    os.makedirs(output_folder, exist_ok=True)
    
    # save args
    with open(os.path.join(output_folder, "args.txt"), "w") as f:
        f.write(str(args))
    
    log_folder = os.path.join(output_folder, "logs")
    os.makedirs(log_folder, exist_ok=True)
    
    shutil.copy(args.config, output_folder)
    
    print(f"Logging training progress to {log_folder}")
    writer = SummaryWriter(log_folder)
    
    config = load_config(args.config)
    # Dataset preparation
    train_dataset = NssDataset(data=args.training_input)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size)
    if args.val_input != "":
        val_dataset = NssDataset(data=args.val_input)
    else:
        val_dataset = None
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model, loss, and optimizer
    model = NNS(config=config)
    
    if args.checkpoint != "":
        if torch.cuda.is_available():
            checkpoint = torch.load(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("loaded checkpoint from ", args.checkpoint)
    
    
        # Setup model for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Utilizing {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    model.train()  # Set the model to training mode
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=100000)
    start_time = datetime.datetime.now()
    for epoch in range(args.epochs):
        
        for iteration, (batch_data, targets) in enumerate(train_dataloader):
            
            global_step = epoch * len(train_dataloader) + iteration
            
            batch_data = [item.to(device) for item in batch_data]
            targets = targets.to(device)
            # debug_time = datetime.datetime.now()
            predictions = model(batch_data)
            loss = pairwise_hinge_ranking_loss(predictions, targets)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print("time for one iteration: ", (datetime.datetime.now() - debug_time).total_seconds())
            if global_step % args.log_iter == 0:
                writer.add_scalar("total_loss", loss.item(), global_step=global_step)
                
                rank = get_avg_rank(predictions, targets)
                writer.add_scalar("avg_rank", rank, global_step=global_step)
                time = (datetime.datetime.now() - start_time).total_seconds()
                print(f"Epoch {epoch}, global_step {global_step}, loss: {loss.item()} avg_rank: {rank} time: {time}")
                start_time = datetime.datetime.now()
                
            if global_step % args.save_iter == 0:
                model_save_path = os.path.join(output_folder, f"model_{epoch}_{global_step}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at Epoch {epoch}, global_step {global_step} to {model_save_path}")
                print("log saved at ", output_folder)
                
                if val_dataset is not None:
                    val_rank, val_loss = val_nns(args=args, model=model, val_dataset=val_dataset, print_tag=f"Epoch {epoch}, global_step {global_step}", log=False)
                    writer.add_scalar("val_avg_rank", val_rank, global_step=global_step)
                    writer.add_scalar("val_total_loss", val_loss, global_step=global_step)
                    model_save_path_with_score = os.path.join(output_folder, "model_e{}_s{}_r{}.pth".format(epoch, global_step, str("{:.2f}".format(val_rank)).replace(".", "_")))
                    os.rename(model_save_path, model_save_path_with_score)
                    print("model renamed to : ", model_save_path_with_score)
                    
            
        writer.add_scalar("epoch_total_loss", loss.item(), global_step=epoch)
        

def val_nns(args=None, model=None, val_dataset=None, print_tag=None, log=True):
    
    """
    to only evaluate one instance by specifying --eval_file
    args:
        eval_file : str
            path to the evaluation data
        checkpoint : str
            path to the model checkpoint
    
    
    """
    
    # Dataset preparation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model is None:
        # Model, loss, and optimizer
        config = load_config(args.config)
        model = NNS(config=config)
        checkpoint = torch.load(args.checkpoint)
        
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.`
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        print("loaded checkpoint from ", args.checkpoint)
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        model.to(device)
    model.eval()  # Set the model to training mode
        
    if val_dataset is None:
        val_dataset = NssDataset(data=args.val_input)
        
    val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size)

    rank_list = []
    loss_list = []  
    for sample_idx, (batch_data, targets) in enumerate(val_dataloader):
        
        if len(batch_data) <= 1:
            continue
        
        batch_data = [item.to(device) for item in batch_data]
        targets = targets.to(device)
        # debug_time = datetime.datetime.now()
        predictions = model(batch_data)
        loss = pairwise_hinge_ranking_loss(predictions, targets)
        rank = get_avg_rank(predictions, targets)
        
        rank_list.append(rank)
        loss_list.append(loss.item())
        if log:
            print(f"Sample {sample_idx}, loss: {loss.item()} avg_rank: {rank}")
        
    
    if print_tag is not None:
        print("##### ", print_tag, " #####")
    print(f"Average loss: {np.mean(loss_list)}")
    print(f"Average rank: {np.mean(rank_list)}")
    print("val_input ", args.val_input)
    if args.checkpoint != "":
        print("checkpoint ", args.checkpoint)
    
    return np.mean(rank_list), np.mean(loss_list)
        

# Run the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--option', type=str, default="1",help='(1) train (2) eval')
    
    # running parameters
    parser.add_argument('--num_workers', type=int, default=4, help='')
    parser.add_argument('--log_iter', type=int, default=100, help='')
    parser.add_argument('--save_iter', type=int, default=100000, help='')
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=50, help='')
    parser.add_argument('--batch_size', type=int, default=2, help='')
    parser.add_argument('--lr', type=float, default=1e-5, help='')
    
    # network input
    parser.add_argument('--output_folder', type=str, default="output/nns_output", help='')
    parser.add_argument('--checkpoint', type=str, default="", help='')
    parser.add_argument('--training_input', type=str, default="", help='')
    parser.add_argument('--val_input', type=str, default="", help='')
    parser.add_argument('--eval_file', type=str, default="", help='')
    parser.add_argument('--config', type=str, default="", help='')
    
    # other configurations
    parser.add_argument('--print_tag', type=str, default="", help='')
    
    args = parser.parse_args()
    
    train_nns(args)