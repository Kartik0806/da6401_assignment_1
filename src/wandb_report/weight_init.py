import os

CMD1 = """python -m train \
-d mnist \
-e 1 \
-b 64 \
-l cross_entropy \
-o sgd \
-lr 0.1 \
-wd 0.01 \
-nhl 2 \
-sz 128 128 \
-a relu relu \
-w_i xavier xavier \
--wandb_project wandb_report_ \
--model_save_path test_model.npy \
--run_name "Q-9: xavier_init" \
--analyze_weights True
"""
CMD2 = """python -m train \
-d mnist \
-e 1 \
-b 64 \
-l cross_entropy \
-o sgd \
-lr 0.1 \
-wd 0.01 \
-nhl 2 \
-sz 128 128 \
-a relu relu \
-w_i zeros zeros \
--wandb_project wandb_report_ \
--model_save_path best_model.npy \
--run_name "Q-9: zeros_init" \
--analyze_weights True
"""

os.system(CMD1)
os.system(CMD2)


