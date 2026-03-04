import os
CMD1 = """python -m src.train \
-d mnist \
-e 5 \
-b 64 \
-l cross_entropy \
-o momentum \
-lr 0.1 \
-wd 0.01 \
-nhl 2 \
-sz 128 128 \
-a relu relu \
-w_i xavier xavier \
--wandb_project wandb_report \
--model_save_path models/test_model.npy \
--run_name "Q-5: dead_neurons_relu"
"""

CMD2 = """python -m src.train \
-d mnist \
-e 5 \
-b 64 \
-l cross_entropy \
-o momentum \
-lr 0.1 \
-wd 0.01 \
-nhl 2 \
-sz 128 128 \
-a tanh tanh \
-w_i xavier xavier \
--wandb_project wandb_report \
--model_save_path models/test_model.npy \
--run_name "Q-5: dead_neurons_tanh"
"""

os.system(CMD1)
os.system(CMD2)
