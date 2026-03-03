import os

CMD1 = """python -m src.train \
-d fashion_mnist \
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
--wandb_project test \
--model_save_path models/test_model.npy
"""

os.system(CMD1)
