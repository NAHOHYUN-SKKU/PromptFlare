<div align="center">
<h2> PromptFlare: Prompt-Generalized Defense via Cross-Attention Decoy in Diffusion-Based Inpainting [MM 2025]</h2>
</div>

## Quick Installation
```
git clone https://github.com/NAHOHYUN-SKKU/PromptFlare.git
cd PromptFlare
conda create -n promptflare python==3.10
conda activate promptflare
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requiremetns.txt
```

## Run the Protection

```
python protect.py \
    --image [write_your_image_folder] \
    --mask [write_your_mask_folder] \
    -output [write_your_output_folder]
```
