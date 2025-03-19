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
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def negative_pixels_mask(image):
    """移除负值像素"""
    try:
        # 使用列表形式选择波段
        bands = ['blue', 'green', 'red', 'nir']
        masks = [image.select([band]).gt(0) for band in bands]
        combined_mask = ee.Image.cat(masks).reduce(ee.Reducer.min())
        return image.updateMask(combined_mask)
    except Exception as e:
        logger.error(f"负值像素移除失败: {str(e)}")
        return image

def calculate_common_threshold(image_collection, region_of_interest, band='nir', scale=10):
    """计算水体共同分割阈值"""
    try:
        # 确保band是字符串而不是列表
        band_name = band[0] if isinstance(band, list) else band
        return image_collection.map(lambda img: img.select(band_name)) \
            .reduce(ee.Reducer.percentile([5])) \
            .rename('threshold')
    except Exception as e:
        logger.error(f"计算阈值失败: {str(e)}")
        return ee.Image(0)  # 返回默认阈值

def calculate_nir_limits(image_collection, common_threshold, band='nir'):
    """计算NIR反射率下限"""
    try:
        # 确保band是字符串而不是列表
        band_name = band[0] if isinstance(band, list) else band
        return {
            'nir_limit': image_collection.select(band_name).min(),
            'image_ids': image_collection.aggregate_array('system:index')
        }
    except Exception as e:
        logger.error(f"计算NIR限值失败: {str(e)}")
        return {'nir_limit': ee.Image(0), 'image_ids': []}

def calculate_weight(image, nir_lower, common_threshold, p_value=4):
    """计算单个影像的权重"""
    try:
        # 使用列表格式选择波段
        nir_band = image.select(['nir'])
        water_mask = nir_band.lt(common_threshold)
        
        # 计算权重
        nir_diff = nir_band.subtract(nir_lower)
        threshold_diff = common_threshold.subtract(nir_lower)
        weight = ee.Image(1.0).subtract(nir_diff.divide(threshold_diff)).pow(p_value)
        
        return weight.updateMask(water_mask)
        
    except Exception as e:
        logger.error(f"权重计算失败: {str(e)}")
        return ee.Image(1.0)  # 返回默认权重

def multi_temporal_weighted_composition(image_collection, region_of_interest, p=4, band='nir', scale=10):
    """执行多时相加权组合"""
    try:
        # 确保band是列表格式
        band_name = [band] if isinstance(band, str) else band
        
        # 计算共同阈值和NIR限值
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
        
        # 对每个波段单独处理
        bands = ['blue', 'green', 'red', 'nir']
        composite_bands = []
        
        for band in bands:
            # 计算加权和
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
        
        # 合并所有波段
        return ee.Image.cat(composite_bands)
        
    except Exception as e:
        logger.error(f"多时相加权组合失败: {str(e)}")
        return image_collection.first()

def enhanced_water_segmentation(image, common_threshold, region_of_interest, band='nir'):
    """增强的水体分割模块"""
    # 使用重命名后的波段名称
    return image.select([band]).lt(common_threshold)

def holes_interpolation(image):
    """对影像中的空洞进行插值填充"""
    try:
        # 使用重命名后的波段
        bands = ['blue', 'green', 'red', 'nir']
        interpolated_bands = []
        
        for band in bands:
            # 选择单个波段（使用列表格式）
            single_band = image.select([band])
            
            # 创建掩膜
            mask = single_band.mask()
            
            # 对单个波段进行插值
            filled = single_band.unmask()
            filled = filled.focal_mean(
                radius=2,
                kernelType='circle',
                units='pixels'
            ).updateMask(mask.Not())
            
            # 合并原始数据和填充数据
            interpolated = ee.Image(
                single_band.unmask().where(mask.Not(), filled)
            ).rename(band)
            
            interpolated_bands.append(interpolated)
        
        # 合并所有波段
        return ee.Image.cat(interpolated_bands)
        
    except Exception as e:
        logger.error(f"空洞插值失败: {str(e)}")
        return image

def mean_filtering(image):
    """应用均值滤波"""
    try:
        # 使用重命名后的波段
        bands = ['blue', 'green', 'red', 'nir']
        filtered_bands = []
        
        for band in bands:
            # 选择单个波段（使用列表格式）
            single_band = image.select([band])
            
            # 应用均值滤波
            filtered = single_band.focal_mean(
                radius=1,
                kernelType='circle',
                units='pixels'
            ).rename(band)
            
            filtered_bands.append(filtered)
        
        # 合并所有波段
        return ee.Image.cat(filtered_bands)
        
    except Exception as e:
        logger.error(f"均值滤波失败: {str(e)}")
        return image

def validate_results(result_image, reference_points, band='B8'):
    """验证结果的准确性"""
    accuracy = result_image.select(band).reduceRegion(
        reducer=ee.Reducer.accuracy(),
        geometry=reference_points,
        scale=10,
        maxPixels=1e13
    )
    return accuracy

def seasonal_water_quality_adjustment(image):
    """季节性水质调整"""
    try:
        # 1. 计算NDWI (使用列表格式选择波段)
        nir = image.select(['nir'])
        green = image.select(['green'])
        ndwi = nir.subtract(green).divide(nir.add(green))
        
        # 2. 创建调整因子
        adjustment_factor = ee.Image(1.0).add(ndwi)
        
        # 3. 分别调整每个波段
        bands = ['blue', 'green', 'red', 'nir']
        adjusted_bands = []
        
        for band in bands:
            adjusted = image.select([band]).multiply(adjustment_factor).rename(band)
            adjusted_bands.append(adjusted)
        
        # 4. 合并调整后的波段
        return ee.Image.cat(adjusted_bands)
        
    except Exception as e:
        logger.error(f"季节性水质调整失败: {str(e)}")
        return image

def verify_projections(image):
    """验证投影的服务器端函数"""
    try:
        # 获取基准投影（使用'nir'波段）
        base_projection = image.select('nir').projection()
        base_crs = base_projection.crs()
        
        # 创建一个函数来验证每个波段的投影
        def check_band_projection(band_name):
            band_projection = image.select(band_name).projection()
            band_crs = band_projection.crs()
            return ee.String(band_crs).compareTo(ee.String(base_crs)).eq(0)
        
        # 获取所有波段名称
        bands = image.bandNames()
        
        # 在服务器端验证所有波段的投影
        projection_checks = bands.map(lambda band: check_band_projection(band))
        
        # 如果所有检查都通过，返回原始图像
        all_valid = projection_checks.reduce(ee.Reducer.min())
        
        return image.set('valid_projections', all_valid)
        
    except Exception as e:
        logger.error(f"投影验证失败: {str(e)}")
        return image.set('valid_projections', 0)

def apply_miwc(
    image_collection: ee.ImageCollection,
    aoi: ee.Geometry,
    p_value: int = 4,
    band_to_process: str = 'nir',
    scale: int = 10
) -> ee.Image:
    try:
        logger.info("开始应用MIWC算法...")
        
        # 验证输入集合
        if image_collection.size().getInfo() == 0:
            raise ValueError("输入影像集合为空")
        
        # 验证波段名称
        first_image = image_collection.first()
        available_bands = first_image.bandNames().getInfo()
        required_bands = ['blue', 'green', 'red', 'nir']
        
        if not all(band in available_bands for band in required_bands):
            raise ValueError(f"缺少必要的波段。可用波段: {available_bands}")
        
        # 确保band_to_process是字符串
        band_name = band_to_process[0] if isinstance(band_to_process, list) else band_to_process   
        
        # Step 1: 负像素掩膜
        logger.info("1. 移除负值像素...")
        masked_collection = image_collection.map(negative_pixels_mask)
        
        # Step 2: 季节性水质调整（额外步骤，增强处理效果）
        logger.info("2. 应用季节性水质调整...")
        adjusted_collection = masked_collection.map(seasonal_water_quality_adjustment)
        
        # Step 3: 计算增强版阈值
        logger.info("3. 计算公共阈值...")
        common_threshold = calculate_common_threshold(
            adjusted_collection,
            aoi,
            band=band_name,
            scale=scale
        )
        
        # Step 4: 计算近红外限值
        logger.info("4. 计算NIR限值...")
        nir_limits = calculate_nir_limits(
            adjusted_collection,
            common_threshold,
            band=band_name
        )
        
        # Step 5: 执行加权组合
        logger.info("5. 执行多时相加权组合...")
        composite = multi_temporal_weighted_composition(
            adjusted_collection,
            aoi,
            p=p_value,
            band=band_name,
            scale=scale
        )
        
        # 确保组合结果是ee.Image类型
        composite = ee.Image(composite)
        
        # Step 6: 空洞插值
        logger.info("6. 插值填充空洞...")
        interpolated = holes_interpolation(composite)
        
        # Step 7: 均值滤波
        logger.info("7. 应用均值滤波...")
        filtered = mean_filtering(interpolated)
        
        # 验证投影一致性
        logger.info("验证投影一致性...")
        verified = verify_projections(filtered)
        
        # 在返回结果之前，确保包含所有必要的波段
        first_image = image_collection.first()
        required_bands = ['blue', 'green']
        band_images = []
        
        for band in required_bands:
            band_data = first_image.select([band])
            band_images.append(band_data)
        
        # 合并所有波段
        result = ee.Image.cat(band_images)
        
        # 添加元数据
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        result = result.set({
            'processing_timestamp': timestamp,
            'common_threshold': common_threshold.getInfo(),
            'image_count': adjusted_collection.size().getInfo(),
            'aoi_bounds': aoi.bounds().getInfo()
        })
        
        logger.info("MIWC处理完成")
        return result
        
    except Exception as e:
        logger.error(f"MIWC处理失败: {str(e)}")
        return image_collection.first()

def main():
    """主执行函数"""
    # 配置参数
    aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])  # 舟山群岛研究区域
    date_range = ('2019-01-01', '2019-12-31')
    p_value = 4
    band_to_process = 'B8'
    scale = 10
    cloud_cover_threshold = 50

    try:
        # 创建图像集合
        logger.info("Filtering image collection...")
        image_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(date_range[0], date_range[1]) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_threshold))

        # 确保集合不为空
        image_count = image_collection.size().getInfo()
        logger.info(f"Found {image_count} images")

        if image_count == 0:
            logger.error("No images found in collection")
            return

        # 预处理：移除负值
        logger.info("Masking negative pixels...")
        masked_collection = image_collection.map(negative_pixels_mask)

        # 季节性水质调整
        logger.info("Applying seasonal water quality adjustment...")
        adjusted_collection = masked_collection.map(seasonal_water_quality_adjustment)

        # 计算增强版阈值
        logger.info("Calculating enhanced threshold...")
        common_threshold = calculate_common_threshold(
            adjusted_collection,
            aoi,
            band=[band_to_process],  # 使用列表形式
            scale=scale
        )

        # 计算 NIR limits
        logger.info("Calculating NIR limits...")
        nir_limits = calculate_nir_limits(
            adjusted_collection, 
            common_threshold,
            band=[band_to_process]
        )

        # 执行加权组合
        logger.info("Performing weighted composition...")
        composite = multi_temporal_weighted_composition(
            adjusted_collection,
            aoi,
            p=p_value,
            band=[band_to_process],
            scale=scale
        )

        # 确保组合是 ee.Image
        composite = ee.Image(composite)

        # 插值填充空洞
        logger.info("Interpolating holes...")
        interpolated_composite = holes_interpolation(composite.select(band_to_process))

        # 应用均值滤波
        logger.info("Applying mean filter...")
        filtered_composite = mean_filtering(interpolated_composite)

        # 导出结果
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

        # 启动导出任务
        export_task.start()
        logger.info("Export task started. Check Earth Engine Tasks panel for progress.")

        # 可选：添加显示参数
        vis_params = {
            'min': 0,
            'max': 3000,
            'bands': [band_to_process]
        }

        # 返回处理结果和统计信息
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
        
        # 打印详细的处理统计信息
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
        
        # 打印处理统计信息
        logger.info("Processing Statistics:")
        for key, value in results['processing_stats'].items():
            logger.info(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")