"""Plot functions."""
import os
import matplotlib.pyplot as plt


def plot_images(output_directory: str):
    """Concatenate images generated for the Diffusion class.

    Paramaters:
    ----------
    output_directory: str
        directory containing the results of the difussion class.
    """
    if not os.path.exists(output_directory):
        print(f"Output directory '{output_directory}' does not exist.")
        return

    image_path = os.path.join(output_directory, 'image.jpg')
    mask_path = os.path.join(output_directory, 'mask.jpg')
    inpainting_path = os.path.join(output_directory, 'image_inpaint.jpg')

    if not os.path.isfile(image_path) or not os.path.isfile(mask_path) or not os.path.isfile(inpainting_path):
        print("One or more image files not found.")
        return

    image = plt.imread(image_path)
    mask = plt.imread(mask_path)
    inpainting = plt.imread(inpainting_path)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(mask)
    ax[1].set_title("Mask")
    ax[2].imshow(inpainting)
    ax[2].set_title("Inpainted Image")

    for a in ax:
        a.axis("off")

    combined_path = os.path.join(output_directory, 'combined_image.jpg')
    plt.savefig(combined_path)
    plt.close()


def write_strings_to_file(string1: str, string2: str, file_path: str):
    """Take two strings: prompt and prompt_inpainting and save it to file."""
    with open(file_path, 'w') as file:
        file.write(string1 + '\n')
        file.write(string2 + '\n')
