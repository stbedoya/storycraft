# Description pipeline: 

1. The pipeline generates a photo-realistic image given any text input. 
2. The image generated by the diffusion model is segmented automatically. Two types of segmentation are available: hair segmentation or segmenting the image into the following categories: background, person, cat, dog, and potted plant. The user needs to choose the type of segmentation.
3. Finally, an inpainting diffusion model takes the first generated image, its mask, and a new prompt. The inpainting model replaces the portion of the original image segmented with another image based on a textual prompt.

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
