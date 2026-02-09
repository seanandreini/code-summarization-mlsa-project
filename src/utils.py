import os
import torch

"""
function to save a checkpoint of a model during training.
it saves the model state dict, the optimizer state dict, 
the epoch of the save, and the loss, in the directory specified in checkpoint_dir
"""
def save_checkpoint(epoch, model, optimizer, val_loss, checkpoint_dir, exp_name, is_best):
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_path = checkpoint_dir + exp_name + ("_best_model.pt" if is_best else "_last_model.pt")
  torch.save({'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'val_loss': val_loss,
              }, checkpoint_path)
  

"""
loads a checkpoint. it gets the model and the optimizer, and returns the epoch and validation loss
load_last makes it load the last checkpoint, if false it load the best
"""
def load_checkpoint(model, optimizer, device, checkpoint_dir, exp_name, load_last):
  checkpoint_path = checkpoint_dir + exp_name + ("_last_model.pt" if load_last else "_best_model.pt")
  if(not os.path.exists(checkpoint_path)):
    print("No checkpoint found, starting from scratch")
    return None, None
  
  checkpoint = torch.load(checkpoint_path, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_epoch = checkpoint['epoch']
  val_loss = checkpoint['val_loss']
  print(f"Loaded checkpoint from epoch {start_epoch}")

  for state in optimizer.state.values():
    for k, v in state.items():
      if isinstance(v, torch.Tensor):
        state[k] = v.to(device)
  return start_epoch, val_loss

def decode_ids(pred_ids, tokenizer):
  return [tokenizer.decode(token_id, skip_special_tokens=True) for token_id in pred_ids]