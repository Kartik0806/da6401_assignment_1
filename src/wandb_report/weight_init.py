import os

CMD1 = """python -m src.train \
-d mnist \
-e 5 \
-b 64 \
-l cross_entropy \
-o sgd \
-lr 0.1 \
-wd 0.01 \
-nhl 2 \
-sz 128 128 \
-a relu relu \
-w_i xavier xavier \
--wandb_project wandb_report \
--model_save_path models/test_model.npy \
--run_name "Q-9: xavier_init"
"""
CMD2 = """python -m src.train \
-d mnist \
-e 5 \
-b 64 \
-l cross_entropy \
-o sgd \
-lr 0.1 \
-wd 0.01 \
-nhl 2 \
-sz 128 128 \
-a relu relu \
-w_i zeros zeros \
--wandb_project wandb_report \
--model_save_path models/test_model.npy \
--run_name "Q-9: zeros_init"
"""

os.system(CMD1)
os.system(CMD2)


