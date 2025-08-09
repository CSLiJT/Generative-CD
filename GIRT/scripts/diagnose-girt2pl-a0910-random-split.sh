python diagnose.py \
--model_file ./checkpoint/girt2pl-assist0910-random-split/checkpoint-epoch-20.pt \
--score_matrix_file ./data/ASSIST0910-Random-Split/train.txt \
--output_path ./diagnosis/girt2pl-a0910-random-split-theta \
--device cuda:1
