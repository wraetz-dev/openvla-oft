torchrun --standalone --nnodes 1 --nproc-per-node auto vla-scripts/finetune.py --data_root_dir ~/bluecat-trainingdata --dataset_name evo_center_dodge_center
bash sync_models.sh
