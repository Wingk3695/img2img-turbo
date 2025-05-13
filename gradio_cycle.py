import gradio as gr
from PIL import Image
import torch
from torchvision import transforms
from src.cyclegan_turbo import CycleGAN_Turbo
from src.my_utils.training_utils import build_transform
import os

# 初始化模型函数，避免每次推理都reload
_model_cache = {}

def load_model(pretrained_name, pretrained_path, use_fp16):
    key = f"{pretrained_name}_{pretrained_path}_{use_fp16}"
    if key in _model_cache:
        return _model_cache[key]

    model = CycleGAN_Turbo(pretrained_name=pretrained_name, pretrained_path=pretrained_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if use_fp16:
        model.half()
    model.to("cuda")
    _model_cache[key] = model
    return model

def infer(
    input_image: Image.Image,
    pretrained_name: str,
    pretrained_path: str,
    prompt: str,
    direction: str,
    use_fp16: bool,
    image_prep: str = "resize_512x512",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(pretrained_name if pretrained_name.strip() else None,
                       pretrained_path if pretrained_path.strip() else None,
                       use_fp16)
    model.to(device)

    T_val = build_transform(image_prep)
    input_img = input_image.convert("RGB")
    input_img = T_val(input_img)
    x_t = transforms.ToTensor()(input_img)
    # 注意您的normalize是均值0.5方差0.5，和推理代码一致
    x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).to(device)
    if use_fp16:
        x_t = x_t.half()

    # 参数校验，和脚本一致
    if pretrained_path and (prompt is None or prompt.strip() == ""):
        return None, "Error: prompt is required when loading a custom model_path."
    if pretrained_name and (prompt is not None and prompt.strip() != ""):
        return None, "Error: prompt should NOT be provided when using pretrained model_name."
    if pretrained_name and direction not in [None, ""]:
        return None, "Error: direction should NOT be set when using pretrained model_name."

    with torch.no_grad():
        output = model(x_t, direction=direction if direction else None, caption=prompt if prompt else None)

    output_image = output[0].cpu() * 0.5 + 0.5  # 解归一化到0-1
    output_pil = transforms.ToPILImage()(output_image.clamp(0, 1))
    output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

    return output_pil, ""

title = "CycleGAN Turbo Image Translation"
description = "基于 CycleGAN_Turbo 模型的图像风格转换。上传图片，选择模型和参数，点击运行生成。"

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}\n\n{description}")

    with gr.Row():
        input_image = gr.Image(type="pil", label="输入图片")
        output_image = gr.Image(type="pil", label="输出图片")

    pretrained_name = gr.Textbox(label="预训练模型名称（与脚本参数 --model_name 对应，不填则使用本地权重）", value="", interactive=True)
    pretrained_path = gr.Textbox(label="本地模型权重路径（与脚本参数 --model_path 对应，优先于预训练模型）", value="", interactive=True)
    prompt = gr.Textbox(label="文本提示 Prompt（仅本地权重时必填）", value="", interactive=True)
    direction = gr.Dropdown(label="翻译方向", choices=["a2b", "b2a", ""], value="", interactive=True,
                           info="预训练模型时留空，自定义模型时请选择")
    use_fp16 = gr.Checkbox(label="使用 FP16 精度推理", value=False)

    run_btn = gr.Button("运行")
    status = gr.Textbox(label="状态", interactive=False)

    def run_fn(img, pname, ppath, p, dir_, fp16):
        status_msg = ""
        try:
            out_img, err = infer(img, pname, ppath, p, dir_, fp16)
            if err:
                status_msg = err
                return None, status_msg
            else:
                status_msg = "推理完成"
                return out_img, status_msg
        except Exception as e:
            return None, f"运行出错: {str(e)}"

    run_btn.click(run_fn, inputs=[input_image, pretrained_name, pretrained_path, prompt, direction, use_fp16],
                  outputs=[output_image, status])

if __name__ == "__main__":
    demo.launch()
