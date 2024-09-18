import numpy
# import PIL
# from PIL import Image
# import torch
# from torchvision.transforms.v2 import (
#     Compose,
#     Resize,
#     InterpolationMode,
#     ToImage,
#     ToDtype,
#     Normalize,
# )
# import torch.nn.functional as F

# def preprocess(image: PIL.Image.Image):        
#     im_size = (378, 378)
#     return Compose(
#         [
#             Resize(size=im_size, interpolation=InterpolationMode.BICUBIC),
#             ToImage(),
#             ToDtype(torch.float32, scale=True),
#             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         ]
#     )(image)


# def image_preprocessing(img_path):
#     print("yes")
#     image = Image.open(img_path)
#     im = image
#     im = preprocess(im.convert("RGB"))
#     resized_image = F.interpolate(im.unsqueeze(0), size=(378, 378), mode="bilinear")
#     combined_image = resized_image
#     return combined_image


def image_preprocessing(img_path):
    return "xyz"

# x = image_preprocessing("download.jpeg")
# print(x)
# print(x.shape)