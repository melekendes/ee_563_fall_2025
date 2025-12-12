import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

#load model
print("Loading Stable Diffusion model.")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

#GPU for speed
pipe = pipe.to("cuda")
non_random_seed= 777


cases = [ 
    {
        "prompt": "A futuristic city with flying cars, cyberpunk style, neon lights",
        "steps": 15,  # Low steps
        "name": "City_Low_Quality",
        "guidance":7.5
    },
    {
        "prompt": "A futuristic city with flying cars, cyberpunk style, neon lights",
        "steps": 50,  # High steps
        "name": "City_High_Quality",
        "guidance":7.5
    },
    {
        "prompt": "A cute robot playing chess in a park, realistic photo, 8k",
        "steps": 30,
        "name": "Robot_Chess_Low_Gui",
        "guidance":7.5
    },
    {
        "prompt": "A cute robot playing chess in a park, realistic photo, 8k",
        "steps": 30,
        "name": "Robot_Chess_High_Gui",
        "guidance":15
    }
]

print(f"Starting..")

# Generation Loop
for test in cases:
    print(f"Generating: {test['name']}...")
    print(f"Prompt: {test['prompt']}")
    print(f"Steps: {test['steps']}")
    print(f"Guidance Scale:{test['guidance']}")
    
    generator = torch.Generator("cuda").manual_seed(non_random_seed)
    start_time = time.time()
    
    # Generate image
    image = pipe(
        test['prompt'],
        num_inference_steps=test['steps'],
        guidance_scale=test['guidance'],
        generator=generator
    ).images[0]

    end_time = time.time()
    duration = end_time - start_time
    
    # Save image
    filename = f"{test['name']}.png"
    image.save(filename)
    print(f"Saved to: {filename}\n")
    print(f"Time Taken:{duration:.2f} sec")

print("All images completed successfully!")