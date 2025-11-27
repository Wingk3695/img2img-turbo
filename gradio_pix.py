import gradio as gr
from PIL import Image
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
from src.pix2pix_turbo import Pix2Pix_Turbo
from src.image_prep import canny_from_pil

# 这里复用原代码的推理逻辑，封装成函数：
def infer_image(
    input_image: Image.Image,
    prompt: str,
    model_name: str,
    model_path: str,
    low_threshold: int,
    high_threshold: int,
    gamma: float,
    use_fp16: bool
):
    # 参数校验
    if (model_name == '') == (model_path == ''):
        return None, "请提供 model_name 或 model_path 中的一个"

    # 调整输入图像大小为8的倍数
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

    # 初始化模型（为了速度，实际项目建议缓存模型避免每次都加载）
    model = Pix2Pix_Turbo(pretrained_name=model_name if model_name != '' else None,
                          pretrained_path=model_path if model_path != '' else None)
    model.set_eval()
    if use_fp16:
        model.half()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        if model_name == 'edge_to_image':
            canny = canny_from_pil(input_image, low_threshold, high_threshold)
            canny_viz_inv = Image.fromarray(255 - np.array(canny))
            # canny_viz_inv.save(...) # 可选保存中间结果
            c_t = F.to_tensor(canny).unsqueeze(0).to(device)
            if use_fp16:
                c_t = c_t.half()
            output_image = model(c_t, prompt)

        elif model_name == 'sketch_to_image_stochastic':
            image_t = F.to_tensor(input_image) < 0.5
            c_t = image_t.unsqueeze(0).to(device).float()
            torch.manual_seed(42)
            B, C, H, W = c_t.shape
            noise = torch.randn((1, 4, H // 8, W // 8), device=device)
            if use_fp16:
                c_t = c_t.half()
                noise = noise.half()
            output_image = model(c_t, prompt, deterministic=False, r=gamma, noise_map=noise)

        else:
            c_t = F.to_tensor(input_image).unsqueeze(0).to(device)
            if use_fp16:
                c_t = c_t.half()
            output_image = model(c_t, prompt)

    output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
    return output_pil, None


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Pix2Pix_Turbo 图像转换演示")

        with gr.Row():
            input_img = gr.Image(type="pil", label="输入图像")
            output_img = gr.Image(type="pil", label="输出图像")

        prompt = gr.Textbox(label="文本提示 (prompt)", value="a cat", lines=1)

        model_name = gr.Dropdown(label="预训练模型名", choices=['edge_to_image', 'sketch_to_image_stochastic', ''], value='edge_to_image')
        model_path = gr.Textbox(label="本地模型权重路径（与预训练模型二选一）", value="", placeholder="比如 ./my_model.ckpt")

        low_threshold = gr.Slider(0, 255, value=100, step=1, label="Canny低阈值")
        high_threshold = gr.Slider(0, 255, value=200, step=1, label="Canny高阈值")
        gamma = gr.Slider(0.0, 1.0, value=0.4, step=0.01, label="Sketch引导强度 (gamma)")

        use_fp16 = gr.Checkbox(label="使用 FP16 加速推理", value=False)

        run_btn = gr.Button("开始推理")

        error_output = gr.Textbox(label="错误信息", interactive=False, visible=False)

        def run(
            input_img,
            prompt,
            model_name,
            model_path,
            low_threshold,
            high_threshold,
            gamma,
            use_fp16
        ):
            out, err = infer_image(
                input_img,
                prompt,
                model_name,
                model_path,
                low_threshold,
                high_threshold,
                gamma,
                use_fp16
            )
            if err:
                error_output.visible = True
                error_output.value = err
            else:
                error_output.visible = False
            return out

        run_btn.click(
            run,
            inputs=[input_img, prompt, model_name, model_path, low_threshold, high_threshold, gamma, use_fp16],
            outputs=output_img
        )

    demo.launch()

if __name__ == "__main__":
    main()
