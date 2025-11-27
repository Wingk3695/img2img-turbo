import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
from src.my_utils.dino_struct import DinoStructureLoss

def compute_dino_struct(source_folder, generated_folder, device="cuda"):
    """
    计算生成图片与原图的DINO结构相似度（根据文件名一一对应）
    """
    net_dino = DinoStructureLoss()
    # net_dino.to(device)  # 如果需要可在DinoStructureLoss内部处理device

    gen_files = sorted(glob(os.path.join(generated_folder, "*")))
    # 用文件名在原图文件夹中查找对应原图
    scores = []
    matched = 0
    for gen_path in tqdm(gen_files, desc="DINO结构相似度"):
        fname = os.path.basename(gen_path)
        src_path = os.path.join(source_folder, fname)
        if not os.path.exists(src_path):
            print(f"警告: 找不到原图 {src_path}，跳过")
            continue
        src_img = Image.open(src_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")
        a = net_dino.preprocess(src_img).unsqueeze(0).to(device)
        b = net_dino.preprocess(gen_img).unsqueeze(0).to(device)
        score = net_dino.calculate_global_ssim_loss(a, b).item()
        scores.append(score)
        matched += 1
    if matched == 0:
        print("未找到任何可比对的图片对。")
        return None
    mean_score = np.mean(scores)
    print(f"DINO结构相似度均值: {mean_score:.4f}，共比对{matched}对图片")
    return mean_score

def compute_fid(generated_folder, target_folder, device="cuda"):
    """
    计算生成图片与目标域图片的FID
    """
    feat_model = build_feature_extractor("clean", device, use_dataparallel=False)
    # 目标域特征
    tgt_features = get_folder_features(target_folder, model=feat_model, num_workers=0, num=None,
        shuffle=False, seed=0, batch_size=8, device=torch.device(device),
        mode="clean", custom_fn_resize=None, description="", verbose=True,
        custom_image_tranform=None)
    tgt_mu, tgt_sigma = np.mean(tgt_features, axis=0), np.cov(tgt_features, rowvar=False)
    # 生成域特征
    gen_features = get_folder_features(generated_folder, model=feat_model, num_workers=0, num=None,
        shuffle=False, seed=0, batch_size=8, device=torch.device(device),
        mode="clean", custom_fn_resize=None, description="", verbose=True,
        custom_image_tranform=None)
    gen_mu, gen_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    fid_score = frechet_distance(tgt_mu, tgt_sigma, gen_mu, gen_sigma)
    print(f"FID分数: {fid_score:.2f}")
    return fid_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="计算生成图片与原图的DINO结构相似度，以及生成图片与目标域图片的FID")
    parser.add_argument("--source_folder", type=str, required=True, help="原图文件夹")
    parser.add_argument("--generated_folder", type=str, required=True, help="生成图片文件夹")
    parser.add_argument("--target_folder", type=str, required=True, help="目标域图片文件夹")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    compute_dino_struct(args.source_folder, args.generated_folder, device=args.device)
    compute_fid(args.generated_folder, args.target_folder, device=args.device)