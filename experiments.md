clear2rain
```bash
python inference_folder2folder_cycle.py --input_folder ~/git/dataset/100k_1/testB --output_folder ~/git/dataset/100k_1/testB_out_turbo --model_path /home/custom_users/wangkang/git/img2img-turbo/checkpoints/clear2rainy_23001.pkl --prompt "driving in rain" --direction b2a
```
平均生成时间: 0.3528 秒/张，共处理 127 张图片。
```bash
python inference_folder2folder_cycle.py \
    --input_folder ~/git/dataset/100k_clear2rain_sim/test_B \
    --output_folder ~/git/dataset/100k_clear2rain_sim/test_B_out_turbo_old \
    --model_path /home/custom_users/wangkang/git/img2img-turbo/checkpoints/clear2rainy_23001.pkl \
    --prompt "driving in rain" \
    --direction b2a

python inference_folder2folder_cycle.py \
    --input_folder ~/git/dataset/100k_clear2rain_sim/test_B \
    --output_folder ~/git/dataset/100k_clear2rain_sim/test_B_out_turbo_new \
    --model_path /home/custom_users/wangkang/git/img2img-turbo/checkpoints/clear2rainy_sim_20001.pkl \
    --prompt "driving in rain" \
    --direction b2a

python fid_dino.py --source_folder ~/git/dataset/100k_clear2rain_sim/test_B --generated_folder  ~/git/dataset/100k_clear2rain_sim/test_B_out_turbo_old --target_folder ~/git/dataset/100k_clear2rain_sim/test_A

python fid_dino.py --source_folder ~/git/dataset/100k_clear2rain_sim/test_B --generated_folder  ~/git/dataset/100k_clear2rain_sim/test_B_out_turbo_new --target_folder ~/git/dataset/100k_clear2rain_sim/test_A

```
python 'fid&dino.py' --source_folder ~/git/dataset/100k_1/testB --generated_folder ~/git/dataset/100k_1/testB_out_turbo --target_folder ~/git/dataset/100k_1/testA

DINO结构相似度均值: 0.0050
FID分数: 179.30

CYC
python 'fid&dino.py' --source_folder ~/git/dataset/100k_1/test_cyc/Cycle_Rain_BDD/test_latest/images/real_B --generated_folder ~/git/dataset/100k_1/test_cyc/Cycle_Rain_BDD/test_latest/images/fake_A --target_folder ~/git/dataset/100k_1/test_cyc/Cycle_Rain_BDD/test_latest/images/real_A
DINO结构相似度均值: 0.0227，共比对18对图片
FID分数: 213.02
python 'fid&dino.py' --source_folder ~/git/dataset/100k_clear2fog/test_cyc/Cycle_Fog_BDD/test_latest/images/real_B --generated_folder ~/git/dataset/100k_clear2fog/test_cyc/Cycle_Fog_BDD/test_latest/images/fake_A --target_folder ~/git/dataset/100k_clear2fog/test_cyc/Cycle_Fog_BDD/test_latest/images/real_A

python 'fid&dino.py' --source_folder ~/git/dataset/100k_day2night/test_cyc/Cycle_Night_BDD/test_latest/images/real_B --generated_folder ~/git/dataset/100k_day2night/test_cyc/Cycle_Night_BDD/test_latest/images/fake_A --target_folder ~/git/dataset/100k_day2night/test_cyc/Cycle_Night_BDD/test_latest/images/real_A
python 'fid&dino.py' \
    --source_folder ~/git/dataset/100k_day2night/test_cyc/CUT_Night_BDD/test_latest/images/real_A \
    --generated_folder ~/git/dataset/100k_day2night/test_cyc/CUT_Night_BDD/test_latest/images/fake_B \
    --target_folder ~/git/dataset/100k_day2night/test_cyc/CUT_Night_BDD/test_latest/images/real_B
python 'fid&dino.py' \
    --source_folder ~/git/dataset/100k_day2night/test_cyc/TPS_Night_BDD/test_latest/images/real_A \
    --generated_folder ~/git/dataset/100k_day2night/test_cyc/TPS_Night_BDD/test_latest/images/fake_B \
    --target_folder ~/git/dataset/100k_day2night/test_cyc/TPS_Night_BDD/test_latest/images/real_B


CUT
python 'fid&dino.py' --source_folder ~/git/dataset/100k_1/test_cyc/CUT_Rain_BDD/test_latest/images/real_A --generated_folder ~/git/dataset/100k_1/test_cyc/CUT_Rain_BDD/test_latest/images/fake_B --target_folder ~/git/dataset/100k_1/test_cyc/CUT_Rain_BDD/test_latest/images/real_B
DINO结构相似度均值: 0.0114，共比对68对图片
FID分数: 102.32

TPS
python 'fid&dino.py' --source_folder ~/git/dataset/100k_1/test_cyc/TPS_Rain_BDD/test_latest/images/real_A --generated_folder ~/git/dataset/100k_1/test_cyc/TPS_Rain_BDD/test_latest/images/fake_B --target_folder ~/git/dataset/100k_1/test_cyc/TPS_Rain_BDD/test_latest/images/real_B
DINO结构相似度均值: 0.0206，共比对68对图片
FID分数: 133.71
----
clear2fog
python inference_folder2folder_cycle.py --input_folder ~/git/dataset/100k_clear2fog/testB --output_folder ~/git/dataset/100k_clear2fog/testB_out_turbo --model_path /home/custom_users/wangkang/git/img2img-turbo/checkpoints/clear2foggy_1001.pkl --prompt "driving in heavy fog" --direction b2a
平均生成时间: 0.3845 秒/张，共处理 127 张图片。

python 'fid&dino.py' --source_folder ~/git/dataset/100k_clear2fog/testB --generated_folder ~/git/dataset/100k_clear2fog/testB_out_turbo --target_folder ~/git/dataset/100k_clear2fog/testA

DINO结构相似度均值: 0.0081
FID分数: 165.09

----
day2night
*预训练模型*
python inference_folder2folder_cycle.py --input_folder ~/git/dataset/100k_day2night/testA --output_folder ~/git/dataset/100k_day2night/testA_out_turbo --model_name day_to_night --max_images 100
平均生成时间: 0.3860 秒/张，共处理 100 张图片。

python 'fid&dino.py' --source_folder ~/git/dataset/100k_day2night/testA --generated_folder ~/git/dataset/100k_day2night/testA_out_turbo --target_folder ~/git/dataset/100k_day2night/testB
DINO结构相似度均值: 0.0326，共比对100对图片

FID分数: 70.81