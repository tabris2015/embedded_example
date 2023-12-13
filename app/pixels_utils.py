from functools import cache


@cache
def get_pixels_hw():
    print("connecting to SPI...")
    import board
    import neopixel_spi as neopixel

    spi = board.SPI()
    pixels = neopixel.NeoPixel_SPI(
        spi,
        8,
        brightness=0.3,
        auto_write=False,
        pixel_order=neopixel.GRB,
        bit0=0b10000000,
    )

    return pixels


def display_pixels(result_list):
    try:
        pixels = get_pixels_hw()
        if result_list:
            people_count = sum(
                [1 for label in result_list[0].labels if label == "person"]
            )
            for i in range(min(people_count, 8)):
                pixels[i] = (i * 16, 255 - i * 16, 0)

            pixels.show()
    except ImportError:
        # print("display pixels!!!")
        pass
