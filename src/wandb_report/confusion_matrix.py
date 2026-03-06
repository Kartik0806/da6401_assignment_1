CMD = """
python -m train \
-d mnist \
-e 5 \
-b 64 \
-l cross_entropy \
-o rmsprop \
-lr 0.001 \
-wd 0.01 \
-nhl 2 \
-sz 128 128 \
-a relu relu \
-w_i xavier xavier \
--wandb_project wandb_report \
--model_save_path test_model.npy \
--run_name "Q-7: confusion_matrix"
"""
import os
os.system(CMD)
