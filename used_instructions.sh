python src/train_cyclegan_turbo.py         \
    --pretrained_model_name_or_path="stabilityai/sd-turbo"         \
    --output_dir="output/cyclegan_turbo/100k_clear2rain"         \
    --dataset_folder "/home/custom_users/wangkang/git/dataset/100k_clear2rain_sim"         \
    --train_img_prep "resize_286_randomcrop_256x256_hflip" --val_img_prep "no_resize"         \
    --learning_rate="1e-5" --max_train_steps=25000         \
    --train_batch_size=1 --gradient_accumulation_steps=1         \
    --report_to "wandb" --tracker_project_name "unpaired_clear2fog_cycle_sim"         \
    --enable_xformers_memory_efficient_attention --validation_steps 250         \
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1