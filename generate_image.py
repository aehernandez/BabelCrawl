import math
import random

try:
    # using pathos.multiprocess
    from multiprocess import Pool
except:
    # if not found, use regular map
    Pool = dict()
    Pool['map'] = map

from PIL import Image


def random_image(width, height, color_depth, gray=True):
    """ Generate a random image.
    """
    size = get_universe_size(width, height, color_depth)
    value = random.getrandbits(size)
    return generate_image(value, width, height, color_depth, gray)


def get_universe_size(width, height, color_depth):
    """ Get the log base 2 number of all possible images.
        The result of this function is intended to be used with
        random.getrandbits.
    """
    universe_size = (width * height) * color_depth
    return int(math.ceil(universe_size))


def generate_image(value, width=320, height=210, color_depth=8, gray=True):
    num_pixels = width * height

    # TODO: Currently PIL only supports 24-bit pixels for color images, and
    # 8-bit pixels for grayscale images. This limitation is encoded in the
    # library. However, conversion between lower/higher original color depths
    # to PIL modes is still possible.
    # reference:
    # http://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes

    mode = 'L'
    if not gray:
        mode = 'RGB'

    if color_depth != 8 or not gray:
        converter = get_depth_converter(color_depth,
                                        8 if gray else 24, not gray)
        mask = int(math.pow(2, color_depth) - 1)

        def extract_pixel(pos):
            return converter((value >> (pos * color_depth)) & mask)
    else:
        # no need to use converter
        def extract_pixel(pos):
            return (value >> (pos * color_depth)) & 0xff

    pixels = Pool().map(extract_pixel, xrange(num_pixels))
    if not gray:
        # when in rgb pixels will be a list of tupels of rgb values. We flatten
        # this list so that PIL can parse it.
        pixels = [item for sublist in pixels for item in sublist]

    image = Image.frombytes(mode, (width, height), str(bytearray(pixels)))
    return image


def get_depth_converter(original, target, to_rgb=False):
    """ Return a function to convert pixel values of one bit depth to another.
    """
    ratio = (math.pow(2, target) - 1) / (math.pow(2, original) - 1)

    if not to_rgb:
        def converter(value):
            return int(ratio * value)
    else:
        def depth_slice(depth):
            """ Given a bit depth, provide r, g, b mask to seperate a n-bit
            number into 3 (n/3)-bit rgb values. Remainder bits are given to
            more important color channels, green being the most sensitive to
            the human eye followed by red.
            """
            remainder = depth % 3
            red = blue = green = depth // 3
            if remainder == 1:
                green = green + 1
            if remainder == 2:
                red = red + 1
                green = green + 1

            return (red, green, blue)

        def get_rgb_slicer(depth):
            """ Return a function which seperates a value into its rgb components
            """
            (red, green, blue) = depth_slice(depth)
            red_mask = int(math.pow(2, red) - 1)
            green_mask = int(math.pow(2, green) - 1)
            blue_mask = int(math.pow(2, blue) - 1)

            def slicer(value):
                return ((value >> (green + blue)) & red_mask,
                        (value >> (blue)) & green_mask,
                        value & blue_mask)
            return slicer

        slicer = get_rgb_slicer(original)
        original_slice = depth_slice(original)
        target_slice = depth_slice(target)
        rgb_ratios = map(lambda original, target: (math.pow(2, target) - 1) /
                         (math.pow(2, original) - 1),
                         original_slice, target_slice)

        def converter(value):
            (r, g, b) = slicer(value)
            pixel_value = [int(rgb_ratios[0] * r),
                           int(rgb_ratios[1] * g),
                           int(rgb_ratios[2] * b)]
            return pixel_value

    return converter
