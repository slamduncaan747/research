# config/train_shakespeare_char.py - OPTIMIZED VERSION
# This should give you much better performance

out_dir = "out-shakespeare"
eval_interval = 500  # less frequent evaluation
eval_iters = 50  # fewer eval iterations
log_interval = 10  # more frequent logging to see progress

always_save_checkpoint = False
wandb_log = False
wandb_project = "shakespeare"
wandb_run_name = "mini-gpt"

dataset = "shakespeare"

# KEY FIXES FOR SPEED:
gradient_accumulation_steps = 1  # Was 2, now 1 - MAJOR speedup!
batch_size = 16  # Was 16, now 64 - better GPU utilization
block_size = 512  # Was 1024, now 512 - fits in memory better

# Model size (this is fine)
n_layer = 6
n_head = 6
n_embd = 768
dropout = 0.2

learning_rate = 5e-3
max_iters = 500  # Reduced for testing
lr_decay_iters = 500
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# PERFORMANCE SETTINGS:
dtype = "float16"  # Ensure float16, not bfloat16
compile = True  # Keep compilation enabled
device = "cuda"

# Remove the torch.cuda.empty_cache() call from the training loop!
