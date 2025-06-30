import rasterio
from rasterio.enums import Resampling
import numpy as np

def crop_and_resample_rgb(input_path, output_path, target_resolution=0.50):
    """
    Opens a 3-band RGB TIFF, crops the bottom-right 60% square,
    resamples it to the target resolution, and saves the output.

    Parameters:
    - input_path: Path to input RGB TIFF (3 bands)
    - output_path: Path to save output TIFF
    - target_resolution: Desired resolution in meters/pixel (default 0.25 for 25 cm)
    """
    with rasterio.open(input_path) as src:
        height, width = src.height, src.width

        # Crop: 60% bottom-right
        crop_height = int(0.6 * height)
        crop_width = int(0.6 * width)
        start_row = height - crop_height
        start_col = width - crop_width

        window = rasterio.windows.Window(start_col, start_row, crop_width, crop_height)

        # Adjust the transform for the cropped window
        cropped_transform = src.window_transform(window)

        # Compute resampling scale
        scale_factor = src.res[0] / target_resolution
        out_height = int(crop_height * scale_factor)
        out_width = int(crop_width * scale_factor)

        # Read and resample all 3 bands
        resampled = np.zeros((3, out_height, out_width), dtype=np.uint8)

        for i in range(3):
            resampled[i] = src.read(
                i + 1,
                window=window,
                out_shape=(out_height, out_width),
                resampling=Resampling.bilinear
            )

        # Prepare output metadata
        profile = src.profile.copy()
        profile.update({
            "height": out_height,
            "width": out_width,
            "transform": rasterio.Affine(
                target_resolution, 0, cropped_transform.c,
                0, -target_resolution, cropped_transform.f
            ),
            "driver": "GTiff",
            "compress": "lzw",
            "photometric": "RGB",
            "count": 3,
            "dtype": np.uint8
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(resampled)

if __name__ == "__main__":
    input_file = r"C:\Users\Usuario\Downloads\furb-pci-trabalhofinal-jupyter\previsoes\em-blumenau\resolucao-ampliada\q42-2023.tif"
    output_file = "preparado.tif"
    crop_and_resample_rgb(input_file, output_file, target_resolution=0.50)