from diffusers import StableDiffusionPipeline
import torch
from PIL import ImageDraw, Image
import gc
import time


models = [
    "/workspace/models/subject1/400",
    "/workspace/models/subject1/800",
    "/workspace/models/subject1/1200",
    "/workspace/models/subject1/1600",
    "/workspace/models/subject1/2000",
    "/workspace/models/subject1/2400",
    "/workspace/models/subject1/2800",
    "/workspace/models/subject1/3200",
    "/workspace/models/subject1/3600",
    "/workspace/models/subject1/4000",
]

images = []
generator = torch.Generator(device="cuda").manual_seed(854)
prompt = "Portrait of a sbjI, cybernetic, cyberpunk, detailed gorgeous face, vaporwave aesthetic, synthwave , digital painting, artstation, concept art, smooth, sharp focus, illustration, octane render, 8k, art by artgerm and greg rutkowski and alphonse mucha"

for model in models:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    images.append(pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0])

gc.collect()
torch.cuda.empty_cache()
    
def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), "  " + models[i].split("/")[-1],(255,255,255))
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

timestamp = int(time.time())

image = image_grid(images, 2, 4)
image.save(f"/workspace/comparaison/subject1/{timestamp}.png", "PNG")
image # for displaying the image when run in notebook
