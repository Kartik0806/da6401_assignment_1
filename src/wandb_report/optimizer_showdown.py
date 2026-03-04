import os
CMD1 = """python -m src.train \
-d mnist \
-e 5 \
-b 64 \
-l cross_entropy \
-o momentum \
-lr 0.1 \
-wd 0.01 \
-nhl 3 \
-sz 128 128 128 \
-a relu relu relu \
-w_i xavier xavier xavier \
--wandb_project wandb_report \
--model_save_path models/test_model.npy \
--run_name "Q-3: optimizer_momentum"
"""

CMD2 = """python -m src.train \
-d mnist \
-e 5 \
-b 64 \
-l cross_entropy \
-o sgd \
-lr 0.1 \
-wd 0.01 \
-nhl 3 \
-sz 128 128 128 \
-a relu relu relu \
-w_i xavier xavier xavier \
--wandb_project wandb_report \
--model_save_path models/test_model.npy \
--run_name "Q-3: optimizer_sgd"
"""

CMD3 = """python -m src.train \
-d mnist \
-e 5 \
-b 64 \
-l cross_entropy \
-o nag \
-lr 0.1 \
-wd 0.01 \
-nhl 3 \
-sz 128 128 128 \
-a relu relu relu \
-w_i xavier xavier xavier \
--wandb_project wandb_report \
--model_save_path models/test_model.npy \
--run_name "Q-3: optimizer_nag"
"""

CMD4 = """python -m src.train \
-d mnist \
-e 5 \
-b 64 \
-l cross_entropy \
-o rmsprop \
-lr 0.1 \
-wd 0.01 \
-nhl 3 \
-sz 128 128 128 \
-a relu relu relu \
-w_i xavier xavier xavier \
--wandb_project wandb_report \
--model_save_path models/test_model.npy \
--run_name "Q-3: optimizer_rmsprop"
"""

os.system(CMD1)
os.system(CMD2)
os.system(CMD3)
os.system(CMD4)

# cmd1 = """python -m src.train \
# -d mnist \
# -e 10 \
# -b 64 \
# -l cross_entropy \
# -o sgd \
# -lr 0.01 \
# -wd 0.01 \
# -nhl 3 \
# -sz 128 64 32 32 \
# -a sigmoid sigmoid sigmoid sigmoid \
# -w_i xavier xavier xavier xavier\
# --wandb_project test \
# --model_save_path models/test_model.npy \
# --run_name loss_func_sigmoid
# """

# cmd2 = """python -m src.train \
# -d mnist \
# -e 10 \
# -b 64 \
# -l cross_entropy \
# -o sgd \
# -lr 0.01 \
# -wd 0.01 \
# -nhl 3 \
# -sz 128 64 32 32 \
# -a relu relu relu relu \
# -w_i xavier xavier xavier xavier \
# --wandb_project test \
# --model_save_path models/test_model.npy \
# --run_name loss_func_relu
# """

# os.system(cmd1)
# os.system(cmd2)

