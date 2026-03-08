import os

CMD1 = """python -m train \
-d fashion_mnist \
-e 10 \
-b 128 \
-l cross_entropy \
-o nag \
-lr 0.0337 \
-wd 0.0005 \
-nhl 4 \
-sz 256 256 256 256 \
-a tanh  \
-w_i xavier \
--wandb_project wandb_report \
--model_save_path f_mnist_model.npy \
--run_name "Q-10 - Fashion MNIST default"
"""

CMD2 = """python -m train \
-d fashion_mnist \
-e 20 \
-b 64 \
-l cross_entropy \
-o momentum \
-lr 0.05 \
-wd 0.01 \
-nhl 3 \
-sz 256 128 64 \
-a relu \
-w_i xavier \
--wandb_project wandb_report \
--model_save_path f_mnist_model.npy \
--run_name "Q-10 - Fashion MNIST"
"""

CMD3 = """python -m train \
-d fashion_mnist \
-e 20 \
-b 256 \
-l cross_entropy \
-o rmsprop \
-lr 0.001 \
-wd 0.01 \
-nhl 3 \
-sz 256 128 128 \
-a tanh  \
-w_i xavier  \
--wandb_project wandb_report \
--model_save_path f_mnist_model.npy \
--run_name "Q-10 - Fashion MNIST"
"""

os.system(CMD1)
# os.system(CMD2)
# os.system(CMD3)
