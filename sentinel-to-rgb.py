import numpy as np
import rasterio
from rasterio.enums import Resampling
from skimage import exposure

def sentinel_to_rgb(input_path, output_path, target_resolution=0.25, brightness_factor=0.9, crop_center=False):
    """
    Convert Sentinel-2 image to RGB with controlled brightness and optional center crop.
    
    Parameters:
    - input_path: Input Sentinel-2 .tif file
    - output_path: Output RGB .tif file
    - target_resolution: Target resolution in meters/pixel (0.25 for 25 cm/px)
    - brightness_factor: Adjust brightness (0.5-1.5 range, default 0.9 for slightly darker)
    - crop_center: Boolean, if True crops central 1/4 of the image before processing
    """
    # Open the Sentinel-2 image
    with rasterio.open(input_path) as src:
        # Calculate crop window if requested
        if crop_center:
            # Calculate central 1/4 window
            height, width = src.height, src.width
            start_row = height // 4
            end_row = start_row + height // 2
            start_col = width // 4
            end_col = start_col + width // 2
            
            # Update transform for the cropped area
            transform = src.transform * src.transform.translation(start_col, start_row)
            window = rasterio.windows.Window(start_col, start_row, 
                                           end_col - start_col, end_row - start_row)
        else:
            window = None
            transform = src.transform
        
        # Sentinel-2 bands for RGB (B4: Red, B3: Green, B2: Blue)
        band_indices = [3, 2, 1]  # 0-based indexing
        
        # Calculate scale factor for resolution change (10m to 0.25m)
        scale_factor = src.res[0] / target_resolution
        
        # Read and resample each band
        rgb_data_resampled = np.zeros(
            (3, int((end_row - start_row if crop_center else src.height) * scale_factor),
                 int((end_col - start_col if crop_center else src.width) * scale_factor)),
            dtype=src.dtypes[0]
        )
        
        for i, band_idx in enumerate(band_indices):
            rgb_data_resampled[i] = src.read(
                band_idx + 1,  # Rasterio uses 1-based band indexing
                window=window,
                out_shape=(
                    int((end_row - start_row if crop_center else src.height) * scale_factor),
                    int((end_col - start_col if crop_center else src.width) * scale_factor)
                ),
                resampling=Resampling.bilinear
            )
        
        # Convert reflectance values (typically scaled by 10000)
        rgb_normalized = rgb_data_resampled / 10000.0
        
        # Clip values to 0-1 range
        rgb_clipped = np.clip(rgb_normalized, 0, 1)
        
        # Apply adaptive contrast stretching
        def adaptive_stretch(band):
            # Calculate percentiles (using narrower range for better contrast)
            p5, p95 = np.percentile(band, (5, 95))
            return exposure.rescale_intensity(band, in_range=(p5, p95))
        
        rgb_stretched = np.array([adaptive_stretch(band) for band in rgb_clipped])
        
        # Apply brightness adjustment
        rgb_adjusted = rgb_stretched * brightness_factor
        
        # Apply gamma correction with slightly higher gamma for less brightness
        rgb_gamma = exposure.adjust_gamma(rgb_adjusted, gamma=1/2.4)  # Slightly higher than standard 2.2
        
        # Final scaling to 0-255
        rgb_8bit = (np.clip(rgb_gamma, 0, 1) * 255).astype(np.uint8)
        
        # Update metadata
        profile = src.profile
        profile.update(
            count=3,
            dtype=np.uint8,
            driver='GTiff',
            compress='lzw',
            photometric='RGB',
            width=rgb_8bit.shape[2],
            height=rgb_8bit.shape[1],
            transform=rasterio.Affine(
                target_resolution, 0, transform[2],
                0, -target_resolution, transform[5]
            )
        )
        
        # Write output
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(rgb_8bit)

if __name__ == "__main__":
    # Example usage with brightness control and optional center crop
    input_file = "C:\\Users\\Augusto Fimose\\Documents\\aurelio-trabalho-final\\output\\quadrantes\\quadrante42\\BLU_2017-07-29_quadrante42.tif"
    output_file = "quadrante-42_blu-2017_29.tif"

    # Process with center crop and reduced brightness
    sentinel_to_rgb(input_file, output_file, brightness_factor=0.75, crop_center=False, target_resolution=10)