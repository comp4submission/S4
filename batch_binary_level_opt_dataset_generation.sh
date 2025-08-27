python3 -m opt.batch_binary_level_opt_dataset_generation \
  --dataset_path /dev/shm/split_dataset/train \
  --output_path /mnt/ssd1/anonymous/binary_level_opt_dataset/normalized_instruction/binaries_train.pkl;
python3 -m opt.batch_binary_level_opt_dataset_generation \
  --dataset_path /dev/shm/split_dataset/test \
  --output_path /mnt/ssd1/anonymous/binary_level_opt_dataset/normalized_instruction/binaries_test.pkl;
