import os
if __name__ =="__main__":
    projects = []
    #func
    for p in projects:
        os.system("python unixcoder_main.py \
      --output_dir=../results/saved_models \
      --model_type=roberta \
      --nofinetune \
      --do_test \
      --train_data_file=../../data/{}_train.csv \
      --eval_data_file=../../data/{}_val.csv \
      --test_data_file=../../data/{}_test.csv \
      --epochs 10 \
      --block_size 512 \
      --train_batch_size 16 \
      --eval_batch_size 16 \
      --learning_rate 2e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --model_name unixcoder_{}.bin \
      --n_gpu 1\
      --seed 123456  2>&1 | tee train.log".format(p,p,p,p))
    for i in [0.2,0.4,0.6,0.8]:
        for p in projects:
            os.system("python unixcoder_main.py \
          --output_dir=../results/saved_models \
          --model_type=roberta \
          --nofinetune \
          --do_test \
          --train_data_file=../../data/{}_train.csv \
          --eval_data_file=../../data/{}_val.csv \
          --test_data_file=../../data/{}_test.csv \
          --epochs 10 \
          --block_size 512 \
          --train_batch_size 16 \
          --eval_batch_size 16 \
          --learning_rate 2e-5 \
          --max_grad_norm 1.0 \
          --evaluate_during_training \
          --model_name unixcoder_{}_{}.bin \
          --fine_tune_factor {}\
          --n_gpu 1\
          --seed 123456  2>&1 | tee train.log".format(p, p, p, p,i,i))