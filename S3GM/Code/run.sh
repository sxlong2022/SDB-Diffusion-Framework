# CUDA_VISIBLE_DEVICES=2 python ./trainer/train.py --data kol --version test --data_location /media/group3/lzy/generative_model/data/ns4/kolmogorov_vel_train.npy

python sample_cy.py > /results/log_cy.txt
python sample_kol.py > /results/log_kol.txt