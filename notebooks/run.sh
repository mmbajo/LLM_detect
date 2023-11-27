CUDA_VISIBLE_DEVICES=15 nohup python3 mistral_train.py >> exp003.txt&

CUDA_VISIBLE_DEVICES=13 nohup python3 mistral_infer.py --model_path "mistralai/Mistral-7B-v0.1" --pretrain_path "/workspaces/LLM_detect/_OUTPUT/mistral_exp004_clean_data" --data_path "/workspaces/LLM_detect/data/daigt-proper-train-dataset/train_drcat_04.csv" >> infer_04.txt&

nohup python3 tfidf_train.py --data_path "/workspaces/LLM_detect/data/daigt-proper-train-dataset/train_v2_drcat_01.csv" --out_dir "/workspaces/LLM_detect/_OUTPUT/tfidf_exp000" >> tfidf_exp001.txt&