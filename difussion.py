import os
import gc
import torch
import argparse

from PIL import Image
from segmentation import ImageSegmenter
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from plot import plot_images, write_strings_to_file

gc.collect()
torch.cuda.empty_cache()

SEED = 1024
INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
STABLE_DIFUSSION = "runwayml/stable-diffusion-v1-5"
SEGMENTATION_BACKGROUND = 'deeplab_v3.tflite'
SEGMENTATION_HAIR = 'hair_segmenter.tflite'


class Diffusion:
    """
    This class performs the following operations:
    1. Generates an image given an input prompt.
    2. Segment the image automatically. Two models are available:
        hair or background segmentation.
    3. Take a new prompt from the user and inpaint the first generated image.
    """
    def __init__(self,
                 prompt_inpaint: str,
                 prompt: str,
                 output_path: str,
                 segmentation_model: str,
                 n_images: int = 1,
                 inpaint_model: str = INPAINT_MODEL,
                 generator_model: str = STABLE_DIFUSSION) -> None:
        """Initialize Diffusion models.

        Parameters:
        -----------
        prompt: str
            text prompt or description to be transform into images.
        prompt_inpaint: str
            text prompt or description to be used to inpaint.
        output_path: str
            Path use to save results.
        segmentation_model: str
            segmentation model from MediaPipe.
        n_images: int
            number of images generated for the difussion model.
        inpaint_model: str
            Path to the inpaint diffusion model - hugginface
        generator_model: str
            Path to the diffusion model - huggingface
        """
        self.prompt = prompt
        self.prompt_inpaint = prompt_inpaint
        self.output_path = output_path
        self.n_images = n_images
        self.model_path = segmentation_model
        self.inpaint_model = inpaint_model
        self.generator_model = generator_model

    def load_resize(self, image_path: str) -> Image.Image:
        """Load generated images and resize them."""
        # 512*512 is the default height of Stable Diffusion
        image = Image.open(image_path)
        return image.resize((512, 512))

    def run_diffusion(self) -> None:
        """Run diffusion demo."""
        generator = torch.Generator("cuda").manual_seed(SEED)
        pipe = StableDiffusionPipeline.from_pretrained(
            self.generator_model,
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        image = pipe(self.prompt, generator=generator).images[0]
        name_file = "/".join([self.output_path, 'image']) + ".jpg"
        image.save(name_file)

        # Mask image using MediaPipe
        segmenter = ImageSegmenter(
            model_path=self.model_path,
            image_path=name_file,
            output_path=self.output_path
        )
        segmenter.segment_image()

        # Performs inpainting
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.inpaint_model,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")

        image_1 = self.load_resize(name_file)
        image_mask = self.load_resize("/".join([self.output_path, 'mask']) + ".jpg")
        image = pipe(prompt=self.prompt_inpaint,
                     image=image_1,
                     mask_image=image_mask).images[0]
        image.save("/".join([self.output_path, 'image_inpaint']) + ".jpg")


def main():
    """
    Command line entry point for Inpainting demo.

    Parameters
    ----------
    ouput_directory: command line arg
        Output path

    Example:
    --------
        python3 diffusion.py /ouput_directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", help="Specify the output directory")
    args = parser.parse_args()
    output_directory = args.output_directory

    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Directory '{output_directory}' created successfully")
    except Exception as e:
        raise Exception(f"An error occurred while creating the directory: {str(e)}")

    print('-------StoryCraft-------')
    prompt = input("Enter a prompt describing your character and context: ")
    segm_type = input("Enter 1 for hair change or 2 for character change: ")
    prompt_inpaint = input("Describe either your new character or hair style: ")

    # Check segmentation model.
    # If enter incorrect number, choose background as default.
    if segm_type == '1':
        segmentation_model = SEGMENTATION_HAIR
    else:
        segmentation_model = SEGMENTATION_BACKGROUND

    diffusion = Diffusion(
        prompt_inpaint=prompt_inpaint,
        prompt=prompt,
        segmentation_model=segmentation_model,
        output_path=output_directory)
    diffusion.run_diffusion()

    write_strings_to_file(
        prompt,
        prompt_inpaint,
        "/".join([output_directory, 'prompt']) + ".txt")
    plot_images(output_directory)


if __name__ == "__main__":
    main()
