python3 -m compiler.batch_binary_level_compiler_dataset_generation \
  --dataset_path /dev/shm/split_dataset/train \
  --output_path /mnt/ssd1/anonymous/binary_level_compiler_dataset/normalized_instruction/binaries_train.pkl;
python3 -m compiler.batch_binary_level_compiler_dataset_generation \
  --dataset_path /dev/shm/split_dataset/test \
  --output_path /mnt/ssd1/anonymous/binary_level_compiler_dataset/normalized_instruction/binaries_test.pkl;
