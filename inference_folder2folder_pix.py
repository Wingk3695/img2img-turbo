import os
import glob
import argparse
import time
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from src.pix2pix_turbo import Pix2Pix_Turbo
from src.image_prep import canny_from_pil
from tqdm import tqdm

def main(
    input_folder,
    output_folder,
    model_name='',
    model_path='',
    prompt='',
    low_threshold=100,
    high_threshold=200,
    gamma=0.4,
    seed=42,
    use_fp16=False,
    device='cuda'
):
    os.makedirs(output_folder, exist_ok=True)

    model = Pix2Pix_Turbo(pretrained_name=model_name, pretrained_path=model_path)
    model.set_eval()
    if use_fp16:
        model.half()
    model.to(device)

    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
    files = sorted(files)

    times = []

    for file in tqdm(files, desc="Processing images"):
        input_image = Image.open(file).convert('RGB')

        # 调整尺寸为8的倍数
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        with torch.no_grad():
            if model_name == 'edge_to_image':
                canny = canny_from_pil(input_image, low_threshold, high_threshold)
                c_t = F.to_tensor(canny).unsqueeze(0).to(device)
                if use_fp16:
                    c_t = c_t.half()
                start_time = time.time()
                output_image = model(c_t, prompt)
                elapsed = time.time() - start_time

            elif model_name == 'sketch_to_image_stochastic':
                image_t = F.to_tensor(input_image) < 0.5
                c_t = image_t.unsqueeze(0).to(device).float()
                torch.manual_seed(seed)
                B, C, H, W = c_t.shape
                noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
                if use_fp16:
                    c_t = c_t.half()
                    noise = noise.half()
                start_time = time.time()
                output_image = model(c_t, prompt, deterministic=False, r=gamma, noise_map=noise)
                elapsed = time.time() - start_time

            else:
                c_t = F.to_tensor(input_image).unsqueeze(0).to(device)
                if use_fp16:
                    c_t = c_t.half()
                start_time = time.time()
                output_image = model(c_t, prompt)
                elapsed = time.time() - start_time

        times.append(elapsed)
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        out_path = os.path.join(output_folder, os.path.basename(file))
        output_pil.save(out_path)

    avg_time = sum(times) / len(times) if times else 0
    print(f"平均生成时间: {avg_time:.4f} 秒/张，共处理 {len(times)} 张图片。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pix2Pix Turbo 文件夹批量推理并保存图片")
    parser.add_argument("--input_folder", type=str, required=True, help="输入图片文件夹路径")
    parser.add_argument("--output_folder", type=str, default="output", help="输出文件夹")
    parser.add_argument("--model_name", type=str, default='', help="预训练模型名称")
    parser.add_argument("--model_path", type=str, default='', help="本地模型权重路径")
    parser.add_argument("--prompt", type=str, required=True, help="文本提示")
    parser.add_argument("--low_threshold", type=int, default=100, help="Canny低阈值")
    parser.add_argument("--high_threshold", type=int, default=200, help="Canny高阈值")
    parser.add_argument("--gamma", type=float, default=0.4, help="草图插值指导强度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use_fp16", action="store_true", help="是否使用fp16推理")
    args = parser.parse_args()

    if (args.model_name == '') == (args.model_path == ''):
        raise ValueError("必须且只能指定一个：model_name 或 model_path")

    main(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_name=args.model_name,
        model_path=args.model_path,
        prompt=args.prompt,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        gamma=args.gamma,
        seed=args.seed,
        use_fp16=args.use_fp16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )