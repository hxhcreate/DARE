python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name gsm8k --split train --repeat 1 --task math
python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name math --split train --repeat 1 --task math
python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name sudoku --split train --repeat 1 --sample_num 20000 --task math
python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name countdown --split train --repeat 1 --sample_num 20000 --task math

python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name gsm8k --split test --repeat 1 --task math
python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name math500 --split test --repeat 1 --task math
python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name humanevalplus --split test --repeat 1 --task code
python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name mbpp --split test --repeat 1 --task code
python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name humaneval --split test --repeat 1 --task code
python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name sudoku --split test --repeat 1 --task math
python -m verl.utils.preprocess.preprocess --input_dir data/source --output_dir data/preprocessed/rl --dataset_name countdown --split test --repeat 1 --task math
