import ee
import netCDF4
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project='YOUR_GEE_PROJECT_ID')
    except Exception as e:
        print(f"Failed to initialize Google Earth Engine: {e}")
        ee.Authenticate()
        ee.Initialize(project='YOUR_GEE_PROJECT_ID')

def get_sentinel2_images(aoi, date_range):
    """Retrieve and preprocess Sentinel-2 imagery
    
    Args:
        aoi (ee.Geometry): Area of interest
        date_range (tuple): Time range tuple (start_date, end_date)
    
    Returns:
        ee.ImageCollection: Preprocessed image collection
    """
    try:
        logger.info("Starting Sentinel-2 image retrieval...")

        # 1. Get original image collection
        collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(date_range[0], date_range[1]) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))

        def standardize_projection(image):
            """Standardize image projection"""
            try:
                # 1. Get B8 band as reference projection
                base_band = image.select('B8')
                base_projection = base_band.projection()
                base_crs = base_projection.crs()

                # 2. Get required bands
                bands = ['B2', 'B3', 'B4', 'B8']

                # 3. Reproject each band and merge
                def reproject_band(band_name):
                    return image.select(band_name).reproject(
                        crs=base_crs,  # Use CRS string directly
                        scale=10
                    )

                # 4. Reproject and merge all bands
                reprojected = ee.Image.cat([reproject_band(band) for band in bands])

                # 5. Copy properties from original image
                return reprojected.copyProperties(image, image.propertyNames())

            except Exception as e:
                logger.error(f"Projection standardization failed: {str(e)}")
                return image

        # 2. Apply atmospheric correction and projection standardization
        collection = collection.map(siac).map(standardize_projection)

        # 3. Cloud masking
        csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        QA_BAND = 'cs'
        CLEAR_THRESHOLD = 0.60

        collection = collection \
            .linkCollection(csPlus, ['cs']) \
            .map(lambda img: img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)))

        # 4. Standardize band names
        collection = collection.select(
            ['B2', 'B3', 'B4', 'B8'],
            ['blue', 'green', 'red', 'nir']
        )

        logger.info("Sentinel-2 image retrieval and preprocessing completed")
        return collection

    except Exception as e:
        logger.error(f"Failed to retrieve Sentinel-2 images: {str(e)}")
        raise

def load_gebco_data(gebco_file):
    """
    Load GEBCO bathymetry data.

    Args:
        gebco_file (str): Path to GEBCO data file.

    Returns:
        np.ndarray: Bathymetry data array.
    """
    try:
        with netCDF4.Dataset(gebco_file) as nc:
            depth_data = nc.variables['elevation'][:]
            return depth_data
    except Exception as e:
        print(f"Failed to load GEBCO data: {e}")
        return None


def siac(image):
    """
    Applies the Sensor Invariant Atmospheric Correction (SIAC) to Sentinel-2 imagery.
    Based on: Yin et al., 2022 (https://doi.org/10.5194/gmd-15-7933-2022)
    
    Args:
        image (ee.Image): A Sentinel-2 TOA reflectance image.

    Returns:
        ee.Image: A Sentinel-2 surface reflectance image.
    """
    # 1. Get image metadata
    date = ee.Date(image.get('system:time_start'))
    footprint = image.geometry()
    
    # 2. Get atmospheric parameters
    # Use MODIS aerosol data
    aot = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES') \
        .filterDate(date.advance(-1, 'day'), date.advance(1, 'day')) \
        .select(['Optical_Depth_047', 'Optical_Depth_055']) \
        .mean() \
        .clip(footprint)
    
    # Use NCEP water vapor data
    water_vapor = ee.ImageCollection('NCEP_RE/surface_wv') \
        .filterDate(date.advance(-1, 'day'), date.advance(1, 'day')) \
        .select('pr_wtr') \
        .mean() \
        .clip(footprint) \
        .multiply(0.1)  # Convert units to g/cm2
    
    # Use TOMS MERGED ozone data
    ozone = ee.ImageCollection('TOMS/MERGED') \
        .filterDate(date.advance(-1, 'day'), date.advance(1, 'day')) \
        .select('ozone') \
        .mean() \
        .clip(footprint) \
        .multiply(0.001)  # Convert units to atm-cm
    
    # Get DEM data for terrain correction
    elevation = ee.Image('USGS/SRTMGL1_003') \
        .select('elevation') \
        .clip(footprint)
    
    # 3. Define bands and their atmospheric parameter sensitivities
    BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
    
    # 4. Implement atmospheric correction
    def apply_atmospheric_correction(img):
        # Get all bands
        img = img.select(BANDS)
        
        # Aerosol correction coefficients
        aot_coeffs = ee.Image([
            0.059, 0.054, 0.049, 0.044, 0.039, 0.034,
            0.029, 0.024, 0.019, 0.014, 0.009, 0.004, 0.002
        ])
        aot_correction = aot.select('Optical_Depth_055').multiply(aot_coeffs)
        
        # Water vapor correction coefficients
        wv_coeffs = ee.Image([
            0.001, 0.002, 0.005, 0.010, 0.015, 0.020,
            0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055
        ])
        wv_correction = water_vapor.multiply(wv_coeffs)
        
        # Ozone correction coefficients
        oz_coeffs = ee.Image([
            0.002, 0.003, 0.004, 0.005, 0.006, 0.007,
            0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014
        ])
        oz_correction = ozone.multiply(oz_coeffs)
        
        # Terrain correction (considering slope and aspect)
        terrain = ee.Terrain.products(elevation)
        slope = terrain.select('slope')
        aspect = terrain.select('aspect')
        
        terrain_correction = slope.multiply(0.001) \
            .add(aspect.multiply(0.0005)) \
            .multiply(ee.Image.constant([
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
            ]))
        
        # Combine all correction terms
        total_correction = aot_correction \
            .add(wv_correction) \
            .add(oz_correction) \
            .add(terrain_correction)
        
        # Apply correction and ensure results are within valid range
        corrected = img.subtract(total_correction) \
            .max(0) \
            .multiply(img.gt(0))  # Preserve original mask
        
        return corrected

    # 5. Apply correction
    corrected_image = apply_atmospheric_correction(image)
    
    # 6. Add quality indicators
    qa_bands = ee.Image([
        corrected_image.reduce(ee.Reducer.stdDev()).rename('correction_stddev'),
        aot.select('Optical_Depth_055').rename('aot'),
        water_vapor.rename('water_vapor'),
        ozone.rename('ozone')
    ])
    
    return corrected_image.addBands(qa_bands).copyProperties(image, [
        'system:time_start',
        'system:time_end',
        'system:index',
        'SPACECRAFT_NAME',
        'PROCESSING_BASELINE',
        'DATATAKE_IDENTIFIER',
        'DATASTRIP_IDENTIFIER',
        'CLOUDY_PIXEL_PERCENTAGE'
    ])

def resample_image(image, resolution=10):
    """Resample image to specified resolution."""
    return image.resample('bilinear').reproject(crs=image.projection(), scale=resolution)

if __name__ == '__main__':
    # Initialize GEE
    initialize_gee()
    
    # Define study area
    aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
    
    # Load GEBCO data
    gebco_years = range(2019, 2025)
    gebco_base_dir = 'GEBCO_Bathymetry/GEBCO_26_Dec_2024_86912dfafafa'
    
    for year in gebco_years:
        # Build new file path
        gebco_file = os.path.join(gebco_base_dir, f'gebco_{year}_n30.8_s30.62_w122.35_e122.6.nc')
        depth_data = load_gebco_data(gebco_file)
        
        if depth_data is not None:
            print(f"Loaded {year} GEBCO data, bathymetry data shape: {depth_data.shape}")

            # Set remote sensing image time range
            date_range = (f'{year-1}-01-01', f'{year-1}-12-31')
            images = get_sentinel2_images(aoi, date_range)
            print(f"Retrieved {images.size().getInfo()} Cloud Score+ S2 images for {year-1}.")
        else:
            print(f"Unable to load {year} GEBCO data, skipping")
