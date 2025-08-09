python analyze.py \
--model_file ./checkpoint/girt2pl-assist0910-random-split/checkpoint-epoch-20.pt \
--evidence_data ./data/ASSIST0910-Random-Split/train.txt \
--device cuda:1 \
--save_path ./visualization/girt_assist_train
