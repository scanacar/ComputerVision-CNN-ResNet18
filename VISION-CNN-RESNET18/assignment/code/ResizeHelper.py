# NECESSARY PYTHON LIBRARIES
from PIL import Image

# RESIZE FUNCTION FOR RESIZING THE DATASET IMAGES
def resize(img, size):

    new_image = Image.new("RGB", size)
    new_image.paste(img, (int((size[0] - img.size[0]) / 2),
                          int((size[1] - img.size[1]) / 2))
                    )

    return new_image
