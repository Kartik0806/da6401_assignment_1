import os
cmd1 = """python -m src.train \
-d mnist \
-e 10 \
-b 64 \
-l cross_entropy \
-o sgd \
-lr 0.01 \
-wd 0.01 \
-nhl 3 \
-sz 128 64 32 \
-a sigmoid sigmoid sigmoid \
-w_i xavier xavier xavier \
--wandb_project test \
--model_save_path models/test_model.npy \
--run_name loss_func_sigmoid
"""

cmd2 = """python -m src.train \
-d mnist \
-e 10 \
-b 64 \
-l cross_entropy \
-o sgd \
-lr 0.01 \
-wd 0.01 \
-nhl 3 \
-sz 128 64 32 \
-a relu relu relu \
-w_i xavier xavier xavier \
--wandb_project test \
--model_save_path models/test_model.npy \
--run_name loss_func_relu
"""

os.system(cmd1)
os.system(cmd2)

