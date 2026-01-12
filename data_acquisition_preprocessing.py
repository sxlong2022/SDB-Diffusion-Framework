import ee
import netCDF4
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def initialize_gee():
    """初始化 Google Earth Engine."""
    try:
        ee.Initialize(project='fast-banner-452901-c8')
    except Exception as e:
        print(f"初始化 Google Earth Engine 失败: {e}")
        ee.Authenticate()
        ee.Initialize(project='fast-banner-452901-c8')

def get_sentinel2_images(aoi, date_range):
    """获取并预处理Sentinel-2影像
    
    Args:
        aoi (ee.Geometry): 感兴趣区域
        date_range (tuple): 时间范围元组 (start_date, end_date)
    
    Returns:
        ee.ImageCollection: 预处理后的影像集合
    """
    try:
        logger.info("开始获取Sentinel-2影像...")

        # 1. 获取原始影像集合
        collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(date_range[0], date_range[1]) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))

        def standardize_projection(image):
            """标准化影像投影"""
            try:
                # 1. 获取B8波段作为基准投影
                base_band = image.select('B8')
                base_projection = base_band.projection()
                base_crs = base_projection.crs()

                # 2. 获取所需波段
                bands = ['B2', 'B3', 'B4', 'B8']

                # 3. 重投影每个波段并合并
                def reproject_band(band_name):
                    return image.select(band_name).reproject(
                        crs=base_crs,  # 直接使用CRS字符串
                        scale=10
                    )

                # 4. 重投影并合并所有波段
                reprojected = ee.Image.cat([reproject_band(band) for band in bands])

                # 5. 复制原始影像的属性
                return reprojected.copyProperties(image, image.propertyNames())

            except Exception as e:
                logger.error(f"投影标准化失败: {str(e)}")
                return image

        # 2. 应用大气校正和投影标准化
        collection = collection.map(siac).map(standardize_projection)

        # 3. 云掩膜处理
        csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        QA_BAND = 'cs'
        CLEAR_THRESHOLD = 0.60

        collection = collection \
            .linkCollection(csPlus, ['cs']) \
            .map(lambda img: img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)))

        # 4. 统一波段命名
        collection = collection.select(
            ['B2', 'B3', 'B4', 'B8'],
            ['blue', 'green', 'red', 'nir']
        )

        logger.info("Sentinel-2影像获取和预处理完成")
        return collection

    except Exception as e:
        logger.error(f"获取Sentinel-2影像失败: {str(e)}")
        raise

def load_gebco_data(gebco_file):
    """
    加载GEBCO水深数据。

    Args:
        gebco_file (str): GEBCO数据文件路径。

    Returns:
        np.ndarray: 水深数据数组。
    """
    try:
        with netCDF4.Dataset(gebco_file) as nc:
            depth_data = nc.variables['elevation'][:]
            return depth_data
    except Exception as e:
        print(f"加载GEBCO数据失败: {e}")
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
    # 1. 获取图像元数据
    date = ee.Date(image.get('system:time_start'))
    footprint = image.geometry()
    
    # 2. 获取大气参数
    # 使用 MODIS 气溶胶数据
    aot = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES') \
        .filterDate(date.advance(-1, 'day'), date.advance(1, 'day')) \
        .select(['Optical_Depth_047', 'Optical_Depth_055']) \
        .mean() \
        .clip(footprint)
    
    # 使用 NCEP 水汽数据
    water_vapor = ee.ImageCollection('NCEP_RE/surface_wv') \
        .filterDate(date.advance(-1, 'day'), date.advance(1, 'day')) \
        .select('pr_wtr') \
        .mean() \
        .clip(footprint) \
        .multiply(0.1)  # 转换单位到 g/cm2
    
    # 使用 TOMS MERGED 臭氧数据
    ozone = ee.ImageCollection('TOMS/MERGED') \
        .filterDate(date.advance(-1, 'day'), date.advance(1, 'day')) \
        .select('ozone') \
        .mean() \
        .clip(footprint) \
        .multiply(0.001)  # 转换单位到 atm-cm
    
    # 获取 DEM 数据用于地形校正
    elevation = ee.Image('USGS/SRTMGL1_003') \
        .select('elevation') \
        .clip(footprint)
    
    # 3. 定义波段和其对应的大气参数敏感度
    BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
    
    # 4. 实现大气校正
    def apply_atmospheric_correction(img):
        # 获取所有波段
        img = img.select(BANDS)
        
        # 气溶胶校正系数
        aot_coeffs = ee.Image([
            0.059, 0.054, 0.049, 0.044, 0.039, 0.034,
            0.029, 0.024, 0.019, 0.014, 0.009, 0.004, 0.002
        ])
        aot_correction = aot.select('Optical_Depth_055').multiply(aot_coeffs)
        
        # 水汽校正系数
        wv_coeffs = ee.Image([
            0.001, 0.002, 0.005, 0.010, 0.015, 0.020,
            0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055
        ])
        wv_correction = water_vapor.multiply(wv_coeffs)
        
        # 臭氧校正系数
        oz_coeffs = ee.Image([
            0.002, 0.003, 0.004, 0.005, 0.006, 0.007,
            0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014
        ])
        oz_correction = ozone.multiply(oz_coeffs)
        
        # 地形校正（考虑坡度和方位角）
        terrain = ee.Terrain.products(elevation)
        slope = terrain.select('slope')
        aspect = terrain.select('aspect')
        
        terrain_correction = slope.multiply(0.001) \
            .add(aspect.multiply(0.0005)) \
            .multiply(ee.Image.constant([
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
            ]))
        
        # 组合所有校正项
        total_correction = aot_correction \
            .add(wv_correction) \
            .add(oz_correction) \
            .add(terrain_correction)
        
        # 应用校正并确保结果在有效范围内
        corrected = img.subtract(total_correction) \
            .max(0) \
            .multiply(img.gt(0))  # 保持原始掩膜
        
        return corrected

    # 5. 应用校正
    corrected_image = apply_atmospheric_correction(image)
    
    # 6. 添加质量指标
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
    """重采样影像到指定分辨率。"""
    return image.resample('bilinear').reproject(crs=image.projection(), scale=resolution)

if __name__ == '__main__':
    # 初始化GEE
    initialize_gee()
    
    # 定义研究区域
    aoi = ee.Geometry.Rectangle([122.35, 30.62, 122.6, 30.8])
    
    # 加载GEBCO数据
    gebco_years = range(2019, 2025)
    gebco_base_dir = 'GEBCO_Bathymetry/GEBCO_26_Dec_2024_86912dfafafa'
    
    for year in gebco_years:
        # 构建新的文件路径
        gebco_file = os.path.join(gebco_base_dir, f'gebco_{year}_n30.8_s30.62_w122.35_e122.6.nc')
        depth_data = load_gebco_data(gebco_file)
        
        if depth_data is not None:
            print(f"加载了 {year} 年 GEBCO 数据，水深数据形状为: {depth_data.shape}")

            # 设定遥感影像时间范围
            date_range = (f'{year-1}-01-01', f'{year-1}-12-31')
            images = get_sentinel2_images(aoi, date_range)
            print(f"获取到 {images.size().getInfo()} 景 {year-1} 年 Cloud Score+ S2 影像。")
        else:
            print(f"无法加载 {year} 年 GEBCO 数据，跳过")
