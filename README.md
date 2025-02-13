# Description pipeline: 

1. The pipeline generates a photorealistic image based on any text input.
2. The image produced by the diffusion model is automatically segmented. Two types of segmentation are available: hair segmentation or segmentation into the following categories—background, person, cat, dog, and potted plant. The user must choose the type of segmentation.
3. Finally, the inpainting diffusion model uses the initially generated image, its mask, and a new prompt to replace the segmented portion of the original image with another image based on the given textual prompt.

Examples: 

![combined_image](https://github.com/stbedoya/storycraft/assets/17913665/ff99609b-a07c-42e0-a1fd-fa72c4d8c27c)

![combined_image](https://github.com/stbedoya/storycraft/assets/17913665/a5f5b22c-316e-44e4-affb-5141bbd71e97)

![combined_image](https://github.com/stbedoya/storycraft/assets/17913665/63db0144-a8e6-4860-8291-babbecfbb752)

## Installation

1. Clone the repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.

## Usage

- Run `python diffusion.py /ouput_directory` to start the application.
- Follow the on-screen instructions.


## Challenges and Constraints with the Diffusion model used in this pipeline
- Diffusion models may not generate accurate faces or limbs due to insufficient training data containing these features.
- The inpainting diffusion model was not specifically trained to inpaint hair. However, it was trained on samples that include cats and dogs, along with descriptions of their fur. As a result, it is possible to modify a person's hair using textual descriptions of animal fur.
- Diffusion models struggle with complex compositional tasks, such as rendering an image based on the prompt: "A red cube on top of a blue sphere."
- Stable Diffusion inpainting performs best on lower-resolution images (e.g., 256×256 or 512×512 pixels). When applied to high-resolution images (768×768 or higher), it may struggle to maintain quality and detail.
- The pipeline offers two segmentation models: DeepLab-v3 and hair_segmenter. DeepLab-v3 enables character modification while preserving the background, whereas hair_segmenter retains both the character and background, allowing only hair color changes.
- The DeepLab-v3 model segments images into categories such as background, person, cat, dog, and potted plant.
- The hair_segmenter model is designed to segment a single person's hair in images or videos captured by a smartphone camera. It may not reliably segment hairstyles with thin or elongated strands, such as mohawks or long braids.

