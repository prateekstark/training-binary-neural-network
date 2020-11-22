def permute_image(image, permutation, batch_size, channels):
    image = image.view(batch_size, channels, -1)
    image = image[:, :, permutation]
    image = image.view(batch_size, channels, image.shape[-2], image.shape[-1])
    return image
