import torch
from tqdm import tqdm

from utils import *
from attention_control import AttnController, MyAttnProcessor2_0

def compute_loss(pipe, attn_controller, mask, masked_image, prompt, args):
    num_inference_steps = 4
    k = 1
    loss_mask = True
    loss_depth = [1024, 256, 64]
    
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]
    text_embeddings = text_embeddings.repeat(2, 1, 1) # [2, 77, 768]
    text_embeddings = text_embeddings.detach()
    
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps_all = pipe.scheduler.timesteps.to(pipe.device)

    num_channels_latents = pipe.vae.config.latent_channels
    noisy_model_input_shape = (1, num_channels_latents, 64, 64)
    latents = torch.randn(noisy_model_input_shape, device=pipe.device, dtype=text_embeddings.dtype)
    latents = latents * pipe.scheduler.init_noise_sigma

    mask64 = torch.nn.functional.interpolate(mask, size=(64, 64)).to(dtype=text_embeddings.dtype)
    mask64 = mask64.repeat(2, 1, 1, 1) # [2, 1, 64, 64]

    masked_image_latents = pipe.vae.encode(masked_image).latent_dist.sample()
    masked_image_latents = 0.18215 * masked_image_latents
    masked_image_latents = masked_image_latents.repeat(2, 1, 1, 1) # [2, 4, 64, 64]

    encoder_attention_mask = torch.ones(2, 77).to(device=pipe.device)
    encoder_attention_mask[1][1:] = 0
    
    text_losses = []
    for i in range(min(k, num_inference_steps)):
        timesteps = timesteps_all[i].long()
        latents = latents.repeat(2, 1, 1, 1)

        latent_model_input = torch.cat([latents, mask64, masked_image_latents], dim=1) # [2, 9, 64, 64]

        noise_pred = pipe.unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings, encoder_attention_mask=encoder_attention_mask)[0]
        pred_noise, target_noise = noise_pred.chunk(2)
        
        text_loss = attn_controller.cal_loss(loss_mask=loss_mask, loss_depth=loss_depth)
        text_losses.append(text_loss)
        
        latents = pred_noise
    
    loss = torch.stack(text_losses).mean()
    return loss

def method(pipe, init_image, mask_image, args):
    image = prepare_image(init_image).cuda()
    mask = prepare_mask(mask_image).cuda()
    
    attn_controller = AttnController(mask)
    for n, m in pipe.unet.named_modules():
        if n.endswith('attn2'): # or n.endswith('attn1'):
            m.set_processor(MyAttnProcessor2_0(attn_controller, n))
    
    src_image_orig = image.clone()
    adv = src_image_orig.clone()
    
    quality_tag_prompt = "professional photography, best quality, ultra high res, photo, art, high quality, realistic, anime, masterpiece, best quality, artistic, detail, 4k, 8k"
    
    grad_reps = 1
    
    iterator = tqdm(range(args.epochs))
    for it in iterator:
        cur_mask = mask.clone()
        attn_controller.set_mask(cur_mask.clone())
        
        masked_adv = adv * (1 - cur_mask)
        
        grads = []
        losses = []
        for i in range(grad_reps):
            cur_mask = cur_mask.clone()
            cur_masked_adv = masked_adv.clone()
            cur_mask.requires_grad = True
            cur_masked_adv.requires_grad = True
            
            loss = compute_loss(pipe, attn_controller, cur_mask, cur_masked_adv, quality_tag_prompt, args)
            
            grad = torch.autograd.grad(loss, [cur_masked_adv])[0] * (1 - cur_mask) # [chunk_size, 3, 512, 512]
            grad = grad.detach().sum(dim=0, keepdim=True) # [1, 3, 512, 512]
            
            grads.append(grad)
            losses.append(loss)
        
        iterator.set_description_str(f'AVG Loss: {torch.stack(losses).mean().item()}')
        
        avg_grad = torch.stack(grads).mean(0)
        
        adv = adv - avg_grad.detach().sign() * args.step_size
        adv = torch.minimum(torch.maximum(adv, src_image_orig - args.eps), src_image_orig + args.eps)
        adv.data = torch.clamp(adv, min=-1.0, max=1.0)
    
    torch.cuda.empty_cache()
    
    return tensor_to_pil(adv)