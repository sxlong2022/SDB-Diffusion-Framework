import ee
from datetime import datetime
import logging
try:
    ee.Initialize()
except Exception as e:
    print(f"Initialization failed: {e}")
    ee.Authenticate()
    ee.Initialize()

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def negative_pixels_mask(image):
    """Remove negative pixels"""
    try:
        # Use list format to select bands
        bands = ['blue', 'green', 'red', 'nir']
        masks = [image.select([band]).gt(0) for band in bands]
        combined_mask = ee.Image.cat(masks).reduce(ee.Reducer.min())
        return image.updateMask(combined_mask)
    except Exception as e:
        logger.error(f"Negative pixel removal failed: {str(e)}")
        return image

def calculate_common_threshold(image_collection, region_of_interest, band='nir', scale=10):
    """Calculate common water segmentation threshold"""
    try:
        # Ensure band is a string, not a list
        band_name = band[0] if isinstance(band, list) else band
        return image_collection.map(lambda img: img.select(band_name)) \
            .reduce(ee.Reducer.percentile([5])) \
            .rename('threshold')
    except Exception as e:
        logger.error(f"Threshold calculation failed: {str(e)}")
        return ee.Image(0)  # Return default threshold

def calculate_nir_limits(image_collection, common_threshold, band='nir'):
    """Calculate NIR reflectance lower limit"""
    try:
        # Ensure band is a string, not a list
        band_name = band[0] if isinstance(band, list) else band
        return {
            'nir_limit': image_collection.select(band_name).min(),
            'image_ids': image_collection.aggregate_array('system:index')
        }
    except Exception as e:
        logger.error(f"NIR limit calculation failed: {str(e)}")
        return {'nir_limit': ee.Image(0), 'image_ids': []}

def calculate_weight(image, nir_lower, common_threshold, p_value=4):
    """Calculate weight for a single image"""
    try:
        # Use list format to select band
        nir_band = image.select(['nir'])
        water_mask = nir_band.lt(common_threshold)
        
        # Calculate weight
        nir_diff = nir_band.subtract(nir_lower)
        threshold_diff = common_threshold.subtract(nir_lower)
        weight = ee.Image(1.0).subtract(nir_diff.divide(threshold_diff)).pow(p_value)
        
        return weight.updateMask(water_mask)
        
    except Exception as e:
        logger.error(f"Weight calculation failed: {str(e)}")
        return ee.Image(1.0)  # Return default weight

def multi_temporal_weighted_composition(image_collection, region_of_interest, p=4, band='nir', scale=10):
    """Perform multi-temporal weighted composition"""
    try:
        # Ensure band is in list format
        band_name = [band] if isinstance(band, str) else band
        
        # Calculate common threshold and NIR limits
        common_threshold = calculate_common_threshold(
            image_collection, 
            region_of_interest,
            band=band_name,
            scale=scale
        )
        
        nir_limits = calculate_nir_limits(
            image_collection,
            common_threshold,
            band=band_name
        )
        
        # Process each band separately
        bands = ['blue', 'green', 'red', 'nir']
        composite_bands = []
        
        for band in bands:
            # Calculate weighted sum
            weighted_sum = image_collection.map(
                lambda img: img.select([band]).multiply(
                    calculate_weight(
                        img, 
                        nir_limits['nir_limit'], 
                        common_threshold,
                        p_value=p
                    )
                )
            ).reduce(ee.Reducer.sum()).rename(band)
            
            composite_bands.append(weighted_sum)
        
        # Merge all bands
        return ee.Image.cat(composite_bands)
        
    except Exception as e:
        logger.error(f"Multi-temporal weighted composition failed: {str(e)}")
        return image_collection.first()

def enhanced_water_segmentation(image, common_threshold, region_of_interest, band='nir'):
    """Enhanced water segmentation module"""
    # Use renamed band names
    return image.select([band]).lt(common_threshold)

def holes_interpolation(image):
    """Interpolate and fill holes in the image"""
    try:
        # Use renamed bands
        bands = ['blue', 'green', 'red', 'nir']
        interpolated_bands = []
        
        for band in bands:
            # Select single band (using list format)
            single_band = image.select([band])
            
            # Create mask
            mask = single_band.mask()
            
            # Interpolate single band
            filled = single_band.unmask()
            filled = filled.focal_mean(
                radius=2,
                kernelType='circle',
                units='pixels'
            ).updateMask(mask.Not())
            
            # Merge original data and filled data
            interpolated = ee.Image(
                single_band.unmask().where(mask.Not(), filled)
            ).rename(band)
            
            interpolated_bands.append(interpolated)
        
        # Merge all bands
        return ee.Image.cat(interpolated_bands)
        
    except Exception as e:
        logger.error(f"Hole interpolation failed: {str(e)}")
        return image

def mean_filtering(image):
    """Apply mean filtering"""
    try:
        # Use renamed bands
        bands = ['blue', 'green', 'red', 'nir']
        filtered_bands = []
        
        for band in bands:
            # Select single band (using list format)
            single_band = image.select([band])
            
            # Apply mean filtering
            filtered = single_band.focal_mean(
                radius=1,
                kernelType='circle',
                units='pixels'
            ).rename(band)
            
            filtered_bands.append(filtered)
        
        # Merge all bands
        return ee.Image.cat(filtered_bands)
        
    except Exception as e:
        logger.error(f"Mean filtering failed: {str(e)}")
        return image

def validate_results(result_image, reference_points, band='B8'):
    """Validate result accuracy"""
    accuracy = result_image.select(band).reduceRegion(
        reducer=ee.Reducer.accuracy(),
        geometry=reference_points,
        scale=10,
        maxPixels=1e13
    )
    return accuracy

def seasonal_water_quality_adjustment(image):
    """Seasonal water quality adjustment"""
    try:
        # 1. Calculate NDWI (using list format to select bands)
        nir = image.select(['nir'])
        green = image.select(['green'])
        ndwi = nir.subtract(green).divide(nir.add(green))
        
        # 2. Create adjustment factor
        adjustment_factor = ee.Image(1.0).add(ndwi)
        
        # 3. Adjust each band separately
        bands = ['blue', 'green', 'red', 'nir']
        adjusted_bands = []
        
        for band in bands:
            adjusted = image.select([band]).multiply(adjustment_factor).rename(band)
            adjusted_bands.append(adjusted)
        
        # 4. Merge adjusted bands
        return ee.Image.cat(adjusted_bands)
        
    except Exception as e:
        logger.error(f"Seasonal water quality adjustment failed: {str(e)}")
        return image

def verify_projections(image):
    """Server-side function to verify projections"""
    try:
        # Get reference projection (using 'nir' band)
        base_projection = image.select('nir').projection()
        base_crs = base_projection.crs()
        
        # Create function to verify each band's projection
        def check_band_projection(band_name):
            band_projection = image.select(band_name).projection()
            band_crs = band_projection.crs()
            return ee.String(band_crs).compareTo(ee.String(base_crs)).eq(0)
        
        # Get all band names
        bands = image.bandNames()
        
        # Verify all band projections on server side
        projection_checks = bands.map(lambda band: check_band_projection(band))
        
        # If all checks pass, return original image
        all_valid = projection_checks.reduce(ee.Reducer.min())
        
        return image.set('valid_projections', all_valid)
        
    except Exception as e:
        logger.error(f"Projection verification failed: {str(e)}")
        return image.set('valid_projections', 0)

def apply_miwc(
    image_collection: ee.ImageCollection,
    aoi: ee.Geometry,
    p_value: int = 4,
    band_to_process: str = 'nir',
    scale: int = 10
) -> ee.Image:
    try:
        logger.info("Starting MIWC algorithm...")
        
        # Validate input collection
        if image_collection.size().getInfo() == 0:
            raise ValueError("Input image collection is empty")
        
        # Validate band names
        first_image = image_collection.first()
        available_bands = first_image.bandNames().getInfo()
        required_bands = ['blue', 'green', 'red', 'nir']
        
        if not all(band in available_bands for band in required_bands):
            raise ValueError(f"Missing required bands. Available bands: {available_bands}")
        
        # Ensure band_to_process is a string
        band_name = band_to_process[0] if isinstance(band_to_process, list) else band_to_process   
        
        # Step 1: Negative pixel mask
        logger.info("1. Removing negative pixels...")
        masked_collection = image_collection.map(negative_pixels_mask)
        
        # Step 2: Seasonal water quality adjustment (additional step for enhanced processing)
        logger.info("2. Applying seasonal water quality adjustment...")
        adjusted_collection = masked_collection.map(seasonal_water_quality_adjustment)
        
        # Step 3: Calculate enhanced threshold
        logger.info("3. Calculating common threshold...")
        common_threshold = calculate_common_threshold(
            adjusted_collection,
            aoi,
            band=band_name,
            scale=scale
        )
        
        # Step 4: Calculate NIR limits
        logger.info("4. Calculating NIR limits...")
        nir_limits = calculate_nir_limits(
            adjusted_collection,
            common_threshold,
            band=band_name
        )
        
        # Step 5: Perform weighted composition
        logger.info("5. Performing multi-temporal weighted composition...")
        composite = multi_temporal_weighted_composition(
            adjusted_collection,
            aoi,
            p=p_value,
            band=band_name,
            scale=scale
        )
        
        # Ensure composite result is ee.Image type
        composite = ee.Image(composite)
        
        # Step 6: Hole interpolation
        logger.info("6. Interpolating holes...")
        interpolated = holes_interpolation(composite)
        
        # Step 7: Mean filtering
        logger.info("7. Applying mean filtering...")
        filtered = mean_filtering(interpolated)
        
        # Verify projection consistency
        logger.info("Verifying projection consistency...")
        verified = verify_projections(filtered)
        
        # Before returning results, ensure all necessary bands are included
        first_image = image_collection.first()
        required_bands = ['blue', 'green']
        band_images = []
        
        for band in required_bands:
            band_data = first_image.select([band])
            band_images.append(band_data)
        
        # Merge all bands
        result = ee.Image.cat(band_images)
        
        # Add metadata
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        result = result.set({
            'processing_timestamp': timestamp,
            'common_threshold': common_threshold.getInfo(),
            'image_count': adjusted_collection.size().getInfo(),
            'aoi_bounds': aoi.bounds().getInfo()
        })
        
        logger.info("MIWC processing completed")
        return result
        
    except Exception as e:
        logger.error(f"MIWC processing failed: {str(e)}")
        return image_collection.first()

def main():
    """Main execution function"""
    # Configuration parameters
    aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])  # Zhoushan Islands study area
    date_range = ('2019-01-01', '2019-12-31')
    p_value = 4
    band_to_process = 'B8'
    scale = 10
    cloud_cover_threshold = 50

    try:
        # Create image collection
        logger.info("Filtering image collection...")
        image_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(date_range[0], date_range[1]) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_threshold))

        # Ensure collection is not empty
        image_count = image_collection.size().getInfo()
        logger.info(f"Found {image_count} images")

        if image_count == 0:
            logger.error("No images found in collection")
            return

        # Preprocessing: remove negative values
        logger.info("Masking negative pixels...")
        masked_collection = image_collection.map(negative_pixels_mask)

        # Seasonal water quality adjustment
        logger.info("Applying seasonal water quality adjustment...")
        adjusted_collection = masked_collection.map(seasonal_water_quality_adjustment)

        # Calculate enhanced threshold
        logger.info("Calculating enhanced threshold...")
        common_threshold = calculate_common_threshold(
            adjusted_collection,
            aoi,
            band=[band_to_process],  # Use list format
            scale=scale
        )

        # Calculate NIR limits
        logger.info("Calculating NIR limits...")
        nir_limits = calculate_nir_limits(
            adjusted_collection, 
            common_threshold,
            band=[band_to_process]
        )

        # Perform weighted composition
        logger.info("Performing weighted composition...")
        composite = multi_temporal_weighted_composition(
            adjusted_collection,
            aoi,
            p=p_value,
            band=[band_to_process],
            scale=scale
        )

        # Ensure composite is ee.Image
        composite = ee.Image(composite)

        # Interpolate holes
        logger.info("Interpolating holes...")
        interpolated_composite = holes_interpolation(composite.select(band_to_process))

        # Apply mean filtering
        logger.info("Applying mean filter...")
        filtered_composite = mean_filtering(interpolated_composite)

        # Export results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_task = ee.batch.Export.image.toDrive(
            image=filtered_composite,
            description=f'MIWC_Composite_{timestamp}',
            scale=scale,
            region=aoi,
            maxPixels=1e13,
            fileFormat='GeoTIFF',
            formatOptions={
                'cloudOptimized': True
            }
        )

        # Start export task
        export_task.start()
        logger.info("Export task started. Check Earth Engine Tasks panel for progress.")

        # Optional: add display parameters
        vis_params = {
            'min': 0,
            'max': 3000,
            'bands': [band_to_process]
        }

        # Return processing results and statistics
        results = {
            'composite': filtered_composite,
            'visualization_params': vis_params,
            'processing_stats': {
                'image_counts': {
                    'total': image_count,
                    'processed': adjusted_collection.size().getInfo(),
                    'by_season': {
                        'spring': adjusted_collection.filter(
                            ee.Filter.calendarRange(3, 5, 'month')
                        ).size().getInfo(),
                        'summer': adjusted_collection.filter(
                            ee.Filter.calendarRange(6, 8, 'month')
                        ).size().getInfo(),
                        'other': adjusted_collection.filter(
                            ee.Filter.calendarRange(9, 2, 'month')
                        ).size().getInfo()
                    }
                },
                'thresholds': {
                    'common': common_threshold.getInfo(),
                    'nir_limits': nir_limits
                },
                'region': aoi.bounds().getInfo()
            }
        }

        logger.info("MIWC processing completed successfully")
        
        # Print detailed processing statistics
        logger.info("\nProcessing Statistics:")
        logger.info(f"Total images processed: {results['processing_stats']['image_counts']['total']}")
        logger.info(f"Common threshold value: {results['processing_stats']['thresholds']['common']}")
        logger.info(f"Spring images: {results['processing_stats']['image_counts']['by_season']['spring']}")
        logger.info(f"Summer images: {results['processing_stats']['image_counts']['by_season']['summer']}")
        logger.info(f"Other season images: {results['processing_stats']['image_counts']['by_season']['other']}")
        
        return results

    except Exception as e:
        logger.error(f"Error in MIWC processing: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        results = main()
        
        # Print processing statistics
        logger.info("Processing Statistics:")
        for key, value in results['processing_stats'].items():
            logger.info(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")