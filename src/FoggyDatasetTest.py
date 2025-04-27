from transformers import CLIPTokenizer
from FoggyDataset import PairedMultiDataset

def test_dataset():
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset_folder = "/home/custom_users/wangkang/git/dataset/prep"
    ds = PairedMultiDataset(dataset_folder=dataset_folder, split="train", image_prep="resize_256", tokenizer=tokenizer)

    print(f"Dataset size: {len(ds)}")

    sample = ds[0]
    print("Keys:", sample.keys())

    print("Caption:", sample["caption"])
    print("Output pixel values shape:", sample["output_pixel_values"].shape)
    print("Conditioning pixel values shape:", sample["conditioning_pixel_values"].shape)
    print("Depth pixel values shape:", sample["depth_pixel_values"].shape)
    print(f"Sampled depth map:{sample["depth_pixel_values"]}")
    print("Input IDs shape:", sample["input_ids"].shape)

if __name__ == "__main__":
    test_dataset()
