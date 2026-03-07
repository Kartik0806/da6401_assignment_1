import os

CMD1 = """python -m src.train \
-d mnist \
-e 10 \
-b 64 \
-l cross_entropy \
-o rmsprop \
-lr 0.01 \
-wd 0.01 \
-nhl 2 \
-sz 32 64 32 32 32 32 \
-a relu relu relu relu relu relu \
-w_i xavier xavier xavier xavier xavier xavier \
--wandb_project wandb_report \
--model_save_path models/test_model.npy \
--run_name "Q-4: vanishing_gradient_relu" \
--analyze_gradients True
"""
CMD2 = """python -m src.train \
-d mnist \
-e 10 \
-b 64 \
-l cross_entropy \
-o rmsprop \
-lr 0.01 \
-wd 0.01 \
-nhl 4 \
-sz 32 64 32 32 32 32 \
-a sigmoid sigmoid sigmoid sigmoid sigmoid sigmoid \
-w_i xavier xavier xavier xavier xavier xavier \
--wandb_project wandb_report \
--model_save_path models/test_model.npy \
--run_name "Q-4: vanishing_gradient_sigmoid"
--analyze_gradients True
"""

os.system(CMD1)
os.system(CMD2)


