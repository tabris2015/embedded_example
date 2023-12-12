

def display_pixels(result_list):
    try:
        import board
        import neopixel_spi as neopixel
        spi = board.SPI()
        pixels = neopixel.NeoPixel_SPI(
            spi,
            8,
            brightness=0.3,
            auto_write=False,
            pixel_order=neopixel.GRB,
            bit0=0b10000000
        )
    except ImportError:
        print("display pixels!!!")

