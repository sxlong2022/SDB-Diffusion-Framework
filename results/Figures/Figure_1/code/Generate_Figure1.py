import ee
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import os
import time

# --- Configuration ---
# Assuming the script is in results/Figures/Figure_1/code/
OUTPUT_DIR = '../' # Save figs in the parent directory relative to /code/
CODE_DIR = '.'      # Directory where this script is located
os.makedirs(OUTPUT_DIR, exist_ok=True)
DPI = 600
REGENERATE_COMPONENTS = True # Set to True to re-download/re-create components

# --- Component Filenames ---
main_map_filename = os.path.join(OUTPUT_DIR, f"figure1_main_map_{DPI}dpi.png")
inset_map_filename = os.path.join(OUTPUT_DIR, f"figure1_inset_map_{DPI}dpi.png")
final_figure_filename = os.path.join(OUTPUT_DIR, f"Figure_1_Study_Area_{DPI}dpi.png")

# Font settings - increase size for better readability in one-column format
FONTSIZE_MAIN_LABELS = 12  # Main map labels
FONTSIZE_MAIN_TICKS = 10   # Main map tick labels
FONTSIZE_INSET_LABELS = 9  # Inset map labels

# Define the main Region of Interest (ROI) bounding box [lon_min, lat_min, lon_max, lat_max]
roi_coords = [122.35, 30.62, 122.6, 30.8]
roi_lon_min, roi_lat_min, roi_lon_max, roi_lat_max = roi_coords
roi_width = roi_lon_max - roi_lon_min
roi_height = roi_lat_max - roi_lat_min

# Define asymmetric buffers to shift ROI towards top-left within the main map frame
main_buffer_left = 0.02
main_buffer_right = 0.08
main_buffer_bottom = 0.02
main_buffer_top = 0.08

# Calculate main map extent using asymmetric buffers
main_extent = [roi_lon_min - main_buffer_left, roi_lon_max + main_buffer_right,
               roi_lat_min - main_buffer_bottom, roi_lat_max + main_buffer_top]

# Define the bounding box for the inset map (showing context with Shanghai/Yangtze)
# Ensure order is [lon_min, lon_max, lat_min, lat_max] for Cartopy
inset_extent = [118.0, 124.0, 28.0, 33.0] # Corrected order

# Point for Majishan Island (approximate coordinates)
majishan_lon, majishan_lat = 122.416, 30.667 # Approx. 122°25' E, 30°40' N
majishan_label = "Majishan Is."

# Points for inset map labels (approximate) - Shanghai will be removed
yangtze_lon, yangtze_lat = 122.0, 31.7 # Adjusted for better label placement
yangtze_label = "Yangtze R.\nEstuary" # Added newline for clarity

# Visualization parameters for Sentinel-2 RGB
rgb_vis = {
    'bands': ['B4', 'B3', 'B2'], # RGB
    'min': 0.0,
    'max': 3000, # Use raw reflectance values (0-10000), adjust max for contrast
    'gamma': 1.4,
}
# Dimensions for downloaded images (pixels). Larger means higher detail but slower download.
main_dims = 1024

# --- GEE Initialization ---
def initialize_gee():
    try:
        ee.Authenticate()
        ee.Initialize(project='fast-banner-452901-c8')
        print("GEE Initialized Successfully.")
        return True
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        print("Please ensure GEE is authenticated.")
        return False

# --- Helper Functions ---
# Function to mask clouds using the SCL band
def mask_s2_clouds(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    mask = mask.And(scl.neq(0)).And(scl.neq(1))
    return image.updateMask(mask).select("B4", "B3", "B2", "SCL")

# Function to get GEE image tile as PIL Image
def get_ee_image_pil(image, vis_params, region_coords, dimensions):
    try:
        url = image.visualize(**vis_params).getThumbURL({
            'region': ee.Geometry.Rectangle(region_coords).toGeoJSONString(),
            'dimensions': dimensions,
            'format': 'png'
        })
        response = requests.get(url, timeout=120) # Increased timeout
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.exceptions.Timeout:
        print(f"Error fetching image tile: Request timed out.")
        return None
    except Exception as e:
        print(f"Error fetching/processing GEE image: {e}")
        return None

# === Generate Component Maps (if needed) ===
if REGENERATE_COMPONENTS or not os.path.exists(main_map_filename) or not os.path.exists(inset_map_filename):
    print("Regenerating map components...")
    if not initialize_gee():
        exit("Exiting due to GEE initialization failure.")

    # --- Fetch Data from GEE for Main Map ---
    print("Fetching GEE data for main map...")
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterDate('2022-01-01', '2022-12-31')
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
    s2_median = s2_collection.map(mask_s2_clouds).select('B4', 'B3', 'B2').median()
    main_map_region = [main_extent[0], main_extent[2], main_extent[1], main_extent[3]]
    main_img_pil = get_ee_image_pil(s2_median, rgb_vis, main_map_region, main_dims)
    if main_img_pil is None:
        exit("Failed to download main map image. Exiting.")
    print("Main map image downloaded.")

    # --- Create and Save Main Map Component ---
    print("Creating Main Map component...")
    plt.rcParams.update({'font.size': FONTSIZE_MAIN_TICKS})  # Update default font size
    
    fig_main_comp = plt.figure(figsize=(8, 8))
    ax_main_comp = fig_main_comp.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax_main_comp.set_extent(main_extent, crs=ccrs.PlateCarree())
    ax_main_comp.imshow(main_img_pil, origin='upper', extent=main_extent, transform=ccrs.PlateCarree())
    roi_rect = Rectangle((roi_lon_min, roi_lat_min), roi_width, roi_height,
                         edgecolor='red', facecolor='none', linewidth=0.8,
                         transform=ccrs.PlateCarree())
    ax_main_comp.add_patch(roi_rect)
    ax_main_comp.plot(majishan_lon, majishan_lat, 'bo', markersize=6, transform=ccrs.Geodetic())
    ax_main_comp.text(majishan_lon + 0.01, majishan_lat + 0.01, majishan_label,
                 transform=ccrs.Geodetic(), fontsize=FONTSIZE_MAIN_LABELS, color='blue',
                 ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none'))
    
    # Add manual scale bar - calculate scale in kilometers
    scale_km = 5  # 5 km scale bar
    
    # Using WGS84 ellipsoid parameters for more accurate scale calculation
    # WGS84: Equatorial radius (a) = 6378.137 km, polar flattening (f) = 1/298.257223563
    equatorial_radius = 6378.137  # km
    flattening = 1/298.257223563
    
    # Calculate meridional radius of curvature at the given latitude
    # Formula: a(1-e²)/(1-e²sin²φ)^(3/2) where e² = 2f - f²
    lat_rad = np.radians(roi_lat_min)
    e_sq = 2*flattening - flattening**2
    denominator = (1 - e_sq * np.sin(lat_rad)**2)**(3/2)
    meridional_radius = equatorial_radius * (1 - e_sq) / denominator
    
    # Calculate parallel radius of curvature
    # Formula: a/sqrt(1-e²sin²φ)
    parallel_radius = equatorial_radius / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)
    
    # Calculate meters per degree longitude at this latitude
    meters_per_deg_lon = (np.pi/180) * parallel_radius * np.cos(lat_rad) * 1000
    
    # Calculate the scale in degrees
    scale_degrees = scale_km * 1000 / meters_per_deg_lon
    
    print(f"Scale calculation: {scale_km} km at latitude {roi_lat_min}° = {scale_degrees:.6f} degrees longitude")
    print(f"Using simplified formula would give: {scale_km / (111 * np.cos(lat_rad)):.6f} degrees")
    
    # Position for scale bar (in data coordinates)
    scale_x_start = main_extent[0] + 0.03  # Offset from left edge
    scale_y_pos = main_extent[3] - 0.03  # Offset from top edge
    
    # Add scale bar rectangle - make it thinner
    scale_rect = Rectangle((scale_x_start, scale_y_pos - 0.002), 
                          scale_degrees, 0.004,  # Reduced height
                          edgecolor='black', 
                          facecolor='black',
                          transform=ccrs.PlateCarree())
    ax_main_comp.add_patch(scale_rect)
    
    # Add scale bar text - closer to the bar
    ax_main_comp.text(scale_x_start + scale_degrees/2, scale_y_pos - 0.003,  # Reduced gap
                    f'{scale_km} km', 
                    transform=ccrs.PlateCarree(),
                    ha='center', va='top', 
                    fontsize=FONTSIZE_MAIN_TICKS,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none'))
    
    # Add white backing rectangle for better contrast with the map
    backing_rect = Rectangle((scale_x_start - 0.005, scale_y_pos - 0.012), 
                           scale_degrees + 0.01, 0.016,
                           edgecolor='white',
                           facecolor='white',
                           alpha=0.5,
                           zorder=1,  # Put this behind the scale bar
                           transform=ccrs.PlateCarree())
    ax_main_comp.add_patch(backing_rect)
    
    # Re-add the scale bar on top of the backing
    ax_main_comp.add_patch(scale_rect)
    
    gl = ax_main_comp.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    glx_ticks = np.arange(np.ceil(main_extent[0]*10)/10, np.floor(main_extent[1]*10)/10 + 0.1, 0.1)
    gly_ticks = np.arange(np.ceil(main_extent[2]*10)/10, np.floor(main_extent[3]*10)/10 + 0.1, 0.1)
    gl.xlocator = mticker.FixedLocator(glx_ticks)
    gl.ylocator = mticker.FixedLocator(gly_ticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': FONTSIZE_MAIN_TICKS}
    gl.ylabel_style = {'size': FONTSIZE_MAIN_TICKS}
    
    plt.savefig(main_map_filename, dpi=DPI, bbox_inches='tight')
    print(f"Main map component saved to: {main_map_filename}")
    plt.close(fig_main_comp)

    # --- Create and Save Inset Map Component (Using Cartopy Features) ---
    print("Creating Inset Map component using Cartopy features...")
    fig_inset_comp = plt.figure(figsize=(4, 4))
    ax_inset_comp = fig_inset_comp.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax_inset_comp.set_extent(inset_extent, crs=ccrs.PlateCarree())
    land_color = 'lightgrey'
    ax_inset_comp.add_feature(cfeature.LAND.with_scale('50m'), facecolor=land_color, edgecolor='darkgrey')
    ax_inset_comp.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.5)
    ax_inset_comp.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='gray', linewidth=0.5)
    ax_inset_comp.set_facecolor('aliceblue')
    main_extent_rect = Rectangle((main_extent[0], main_extent[2]),
                                 main_extent[1] - main_extent[0],
                                 main_extent[3] - main_extent[2],
                                 edgecolor='yellow', facecolor='yellow', linewidth=1.0, alpha=0.5,
                                 transform=ccrs.PlateCarree())
    ax_inset_comp.add_patch(main_extent_rect)
    
    # Shanghai label removed, only using Yangtze and other labels
    ax_inset_comp.text(yangtze_lon, yangtze_lat, yangtze_label,
                  transform=ccrs.Geodetic(), fontsize=FONTSIZE_INSET_LABELS, color='black', weight='bold',
                  ha='center', va='top', bbox=dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none'))

    # Add labels for Zhejiang and East China Sea - enlarged
    zhejiang_lon, zhejiang_lat = 120.5, 29.5
    east_china_sea_lon, east_china_sea_lat = 123.0, 29.5
    ax_inset_comp.text(zhejiang_lon, zhejiang_lat, "Zhejiang\nProvince",
                  transform=ccrs.Geodetic(), fontsize=FONTSIZE_INSET_LABELS, color='black', weight='bold',
                  ha='center', va='center', style='italic', bbox=dict(facecolor='lightgrey', alpha=0.5, pad=0.1, edgecolor='none'))
    ax_inset_comp.text(east_china_sea_lon, east_china_sea_lat, "East China\nSea",
                  transform=ccrs.Geodetic(), fontsize=FONTSIZE_INSET_LABELS, color='darkblue', weight='bold',
                  ha='center', va='center', style='italic', bbox=dict(facecolor='aliceblue', alpha=0.5, pad=0.1, edgecolor='none'))

    gl_inset = ax_inset_comp.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                  linewidth=0.3, color='gray', alpha=0.4, linestyle=':')
    gl_inset.xlocator = mticker.MultipleLocator(2)
    gl_inset.ylocator = mticker.MultipleLocator(2)
    plt.savefig(inset_map_filename, dpi=DPI, bbox_inches='tight')
    print(f"Inset map component saved to: {inset_map_filename}")
    plt.close(fig_inset_comp)
else:
    print("Using existing map component files.")

# === Combine Maps ===
print("Combining maps into final figure...")

# Load the saved component images
try:
    img_main = Image.open(main_map_filename)
    img_inset = Image.open(inset_map_filename)
except FileNotFoundError:
    print("Error: Component image files not found. Set REGENERATE_COMPONENTS=True and re-run.")
    exit()

# Create the final figure
# Determine figure size based on main map aspect ratio
main_width_px, main_height_px = img_main.size
fig_width_inches = 8 # Set desired width for the final figure
fig_height_inches = fig_width_inches * (main_height_px / main_width_px)
fig_final = plt.figure(figsize=(fig_width_inches, fig_height_inches))

# Add the main map to the figure
ax_main_final = fig_final.add_axes([0, 0, 1, 1]) # Main axes covering the whole figure
ax_main_final.imshow(img_main)
ax_main_final.axis('off') # Hide axes borders and ticks for the image display

# Add the inset map
# Define inset position and size relative to the figure (e.g., top right)
# [left, bottom, width, height] in figure coordinates (0 to 1)
inset_pos_x = 0.70 # Horizontal position (fraction from left) - Moved further right
inset_pos_y = 0.67 # Vertical position (fraction from bottom) - Keep in top right area
inset_width = 0.28 # Width (fraction of figure width)
inset_height = inset_width * (img_inset.height / img_inset.width) * (fig_width_inches / fig_height_inches)

ax_inset_final = fig_final.add_axes([inset_pos_x, inset_pos_y, inset_width, inset_height])
ax_inset_final.imshow(img_inset)
ax_inset_final.axis('off')
# Add a border around the inset axes
for spine in ax_inset_final.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)

# Save the combined figure
plt.savefig(final_figure_filename, dpi=DPI, bbox_inches='tight')
print(f"Final combined figure saved to: {final_figure_filename}")
plt.close(fig_final)

print("Script finished.")
