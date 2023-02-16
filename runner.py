from stable_diffusion import StableDiffusion

if __name__ == "__main__":
    SD = StableDiffusion()

    text_prompts = "a horse on a beach"
    ckpt = "sd_model.ckpt"

    #TODO: implement into generator
    lora = ""
    lora_weight = 1.5
    SD.generate(
        lora = lora,
        lora_weight = lora_weight,
        text_prompts=text_prompts, 
        negative_prompts="", 
        init_image_str="", 
        mask_b64="",
        invert=False, 
        txt_cfg_scale=1.5, 
        steps=50, 
        H=512, 
        W=512, 
        strength=0.75, 
        cfg_scale=7.5, 
        seed=5,
        sampler="ddim",
        n_iter=4, 
        ckpt=ckpt, 
        vae="", 
        image_save_path="",
        skip_grid=False, 
        batch_id=0)