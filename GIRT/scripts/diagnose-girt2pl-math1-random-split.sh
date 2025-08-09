python diagnose.py \
--model_file ./checkpoint/girt2pl-math1-random-split/checkpoint-epoch-10.pt \
--score_matrix_file ./data/Math1-Random-Split/train.txt \
--output_path ./diagnosis/girt2pl-math1-random-split-theta \
--device cuda:1
