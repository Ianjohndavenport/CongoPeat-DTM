import gdal
from osgeo import gdal, osr, osr
from gdalconst import *
import os
import sys
import shutil
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from osgeo import gdal,ogr,osr
from datetime import datetime
from sklearn.linear_model import LinearRegression
import gc

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1)]
    # Return colormap object.
    return colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def lookup(lon,lat,data_array,xres,yres,ulx,uly) :
    which_xcell=int((lon-ulx)/xres)  
    which_ycell=int((lat-uly)/yres)
    return(data_array[which_ycell][which_xcell])

def area_mean_lookup(lon_low,lat_low,lon_high,lat_high,data_array,xres,yres,ulx,uly, nodata) :
    which_xcell_low= int((lon_low-ulx)/xres)  
    which_xcell_high=int((lon_high-ulx)/xres)  
    which_ycell_low= int((lat_low-uly)/yres)
    which_ycell_high=int((lat_high-uly)/yres)
    #print("{} to {}, {} to {}".format(which_xcell_low, which_xcell_high, which_ycell_low, which_ycell_high))
    allsum=0
    count=0
    for which_ycell in range(which_ycell_low,which_ycell_high+1,int(math.copysign(1,which_ycell_high+1-which_ycell_low))) :
        for which_xcell in range(which_xcell_low,which_xcell_high+1) :
            if data_array[which_ycell][which_xcell]!=nodata :
                allsum+=data_array[which_ycell][which_xcell]
                #print(f"{which_ycell},{which_xcell} is {data_array[which_ycell][which_xcell]}")
                count+=1
    #print(f"In area lookup fn, allsum {allsum}, count {count}")
    #sys.exit()
    if count>0 :
        return(allsum/count)
    else :
        return nodata

def most_common(lst):
    return max(set(lst), key=lst.count)

def area_most_common_class_lookup(lon_low,lat_low,lon_high,lat_high,class_data_array,xres,yres,ulx,uly) :
    which_xcell_low= int((lon_low-ulx)/xres)  
    which_xcell_high=int((lon_high-ulx)/xres)  
    which_ycell_low= int((lat_low-uly)/yres)
    which_ycell_high=int((lat_high-uly)/yres)
    #print(f"{which_xcell_low}-{which_xcell_high},{which_ycell_low}-{which_ycell_high}")
    #print(f"{which_ycell_low} to {which_ycell_high+1} step {int(math.copysign(1,yres))}")
    all_values=[]
    #sys.exit()
    for which_ycell in range(which_ycell_low,which_ycell_high+1,int(math.copysign(1,which_ycell_high-which_ycell_low+1))) :
        for which_xcell in range(which_xcell_low,which_xcell_high+1) :
            all_values.append(class_data_array[which_ycell][which_xcell])
    #print(f"In most common fn {all_values}, {most_common(all_values)}")
    #sys.exit()
    #print(f"Values {all_values}")
    if len(all_values)>0 :
        classification=most_common(all_values)
        count=0
        how_many=0
        #print("{which_ycell_low} to {which_ycell_high+1} step {int(math.copysign(1,yres)}")
        #sys.exit()
        for which_ycell in range(which_ycell_low,which_ycell_high+1,int(math.copysign(1,which_ycell_high-which_ycell_low+1))) :
            for which_xcell in range(which_xcell_low,which_xcell_high+1) :
                count+=1
                if class_data_array[which_ycell][which_xcell]==classification :
                    how_many+=1
        percentage=how_many*100/count
        #print (f"{how_many} of {count} = {percentage}")
        return(most_common(all_values)), percentage
    else :
        return(0,100)

def classification_proportions_percentage(lon_low,lat_low,lon_high,lat_high,class_data_array,xres,yres,ulx,uly) :
    which_xcell_low= int((lon_low-ulx)/xres)  
    which_xcell_high=int((lon_high-ulx)/xres)  
    which_ycell_low= int((lat_low-uly)/yres)
    which_ycell_high=int((lat_high-uly)/yres)
    all_sum=np.zeros(6)
    count=0
    for which_ycell in range(which_ycell_low,which_ycell_high+1,int(math.copysign(1,which_ycell_high-which_ycell_low+1))) :
        for which_xcell in range(which_xcell_low,which_xcell_high+1) :
            if which_ycell>=0 and which_ycell<len(class_data_array) and which_xcell>=0 and which_xcell<len(class_data_array[0]) :
                classification=class_data_array[which_ycell][which_xcell]
                all_sum[classification]+=1
                count+=1
                #print(f"All sum {all_sum}, count{count}")
    
    classification_list=np.zeros(6)
    if count>0 :
        for classification_count in range(6) :
            classification_list[classification_count]=100*all_sum[classification_count]/count
    else :
        classification_list=[100,0,0,0,0,0]
    return classification_list

def coords_from_pixels(which_xcell,which_ycell,ulx,uly,xres,yres) :
    lon=ulx+which_xcell*xres
    lat=uly+which_ycell*yres
    return lon,lat

def classification(number) :
    return {
        0 : "No_data",
        1 : "Water",
        2 : "Savanna",
        3 : "Terra_firme_forest",
        4 : "Palm-dominated_swamp",
        5 : "Hardwood swamp",
        }[number]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

pi=3.14159265
canopy_nodata=-999

#os.getcwd()

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
verbose='save' # plot for interactive, save for save to files, skip to produce no plots (avoid calls to test if matplotlib leaking)
location='Alienware'
if location=='Alienware' :
    input_directory_root="D:\\Dropbox\\"
elif location=='xps8930' :
    input_directory_root="D:\\Hard Disk User Folder\\Dropbox\\"
elif location=='work' :
    input_directory_root="D:\\Dropbox\\"
else :
    print("Don't know location")
    sys.exit()
version="AltAltAlt_v18"
filled_tandemx="yes"
buffer=20
dilate_canopy='no'
canopy_interpolation_distance=12000
number_in_colourbar=100

max_ground_discontinuity_threshold=2
ic2_subset_filter1='' # 'percentile95_Canopy_measure=basic'
ic2_subset_filter2='' # 'Widening=10_pixels'
water_pixel_limit=1 # number of isolated pixels to remove in a resampled TanDEM-X cell, not used currently as explicit percentage preferable
canopy_max_plot=40
max_ground_discontinuity_threshold=1.0
classification_purity_threshold=50
atl08_version='4' # v4 or v4 analysis
tandemx_cellsize=90
canopy_technique="weighted" # How to determine canopy for a cell from classifications -  "majority" for using the majority class, weighted for average weighted by class proportions
mix_diag="no" # Generate a text file showing how canopy_technique mixing has worked. 
tandemx_datum_adjustment=15.0 # 15m higher than GLO-030

ICESat2ATL08_input_directory=input_directory_root+"Work - Edinburgh\\IceSat2\\ATL08 redo\\extractedv14_buffer20\\basin_Alienware\\percentile99_trim_1m\\ATL08v"+atl08_version+"\\"
tandemx_input_directory=input_directory_root+"Work - Edinburgh\\TanDEM-X\\Wider download 90m\\"
classification_input_directory=input_directory_root+"Work - Edinburgh\\Bart Classification Map\\"
#icesat2_input_directory=input_directory_root+"Work - Edinburgh\\TanDEM-X\\Wider download 90m\\results\\v17\\buffer20\\percentile99_trim_1m\\"
lidar_input_directory=input_directory_root+"Work - Edinburgh\\DRC LiDAR\\Analysis\\Data\\ResultsV18_aggregated_500m\\"
output_directory=input_directory_root+"Work - Edinburgh\\TanDEM-X\\Wider download 90m\\CanopyRemoval\\"+version+"\\"
lidar_translation_filename=input_directory_root+"Work - Edinburgh\\DRC LiDAR\\Analysis\\Cross-comparison\\Alt_v05\\het_limit_50\\ground_disc_limit_1\\Overall_results_.txt"
lidar_translation_class_directory=input_directory_root+"Work - Edinburgh\\DRC LiDAR\\Analysis\\Cross-comparison\\Parallel_v02_500m\\het_limit_50pc\\ground_disc_limit_1000000.0m\\cross-comparison-class_v03\\"
if not os.path.isdir(output_directory) :
    os.makedirs(output_directory)

print("Starting classification import at "+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
# Import classification raster
classification_input_filename=classification_input_directory+"ENVI_ML_Dec20_1000runs_Most_likely_class_Masked.tif"
classification_raster=gdal.Open(classification_input_filename)
classification_xcells=classification_raster.RasterXSize
classification_ycells=classification_raster.RasterYSize
classification_bands=classification_raster.RasterCount
classification_metadata=classification_raster.GetMetadata()
classification_band = classification_raster.GetRasterBand(1)
classification_stats=classification_band.ComputeStatistics(0)
classification_bandmin=classification_band.GetMinimum()
classification_bandmax=classification_band.GetMaximum()
classification_ulx, classification_xres, classification_xskew, classification_uly, classification_yskew, classification_yres  = classification_raster.GetGeoTransform()
classification_lrx = classification_ulx + (classification_raster.RasterXSize * classification_xres)
classification_lry = classification_uly + (classification_raster.RasterYSize * classification_yres)
classification_rasterArray = classification_raster.ReadAsArray()
classification_mostlikely_rasterArray=classification_rasterArray.copy()
print(f"Classification raster covers longitude {classification_ulx:4.2f} to {classification_lrx:4.2f}, longitude {classification_uly:4.2f} to {classification_lry:7.2f}, min {classification_bandmin:7.2f}, max {classification_bandmax:4.2f}")
print(f"{classification_xcells} by {classification_ycells} pixels")
# plot classification

print("Finished classification import at "+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
plotmin=0
plotmax=6
# 0 - no data
# 1xxx - Water
# 2xxx - Savanna
# 3xxx - Terra firme forest
# 4xxx - Palm-dominated swamp
# 5xxx - Hardwood swamp
k=number_in_colourbar
c = cmap_discretize('jet', k)

if not 'skip' in verbose :
    
    plt.imshow(classification_mostlikely_rasterArray,interpolation='nearest',cmap=c,vmin=plotmin,vmax=5,extent=[classification_ulx,classification_lrx,classification_lry,classification_uly])
    plt.colorbar()
    plt.title(f"Classes")
    if verbose == 'plot' :
        plt.show()
    else :
        plt.savefig(output_directory+"Classification.png", format="png",dpi=1200)               
    plt.close()



print("Importing TanDEM-X DSM at "+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

if filled_tandemx == "yes" :
    tandemx_input_filename=tandemx_input_directory+"TanDEM-X_SRTM_gapfilled.tif"
else :
    tandemx_input_filename=tandemx_input_directory+"TanDEM-X_mosaic.tif"
tandemx_raster=gdal.Open(tandemx_input_filename)
tandemx_xcells=tandemx_raster.RasterXSize
tandemx_ycells=tandemx_raster.RasterYSize
tandemx_bands=tandemx_raster.RasterCount
tandemx_metadata=tandemx_raster.GetMetadata()
tandemx_band = tandemx_raster.GetRasterBand(1)
tandemx_stats=tandemx_band.ComputeStatistics(0)
#tandemx_nodata=tandemx_band.GetNoDataValue()
tandemx_nodata=-32767
tandemx_bandmin=tandemx_band.GetMinimum()
tandemx_bandmax=tandemx_band.GetMaximum()
tandemx_ulx, tandemx_xres, tandemx_xskew, tandemx_uly, tandemx_yskew, tandemx_yres  = tandemx_raster.GetGeoTransform()
tandemx_lrx = tandemx_ulx + (tandemx_raster.RasterXSize * tandemx_xres)
tandemx_lry = tandemx_uly + (tandemx_raster.RasterYSize * tandemx_yres)
tandemx_rasterArray = tandemx_raster.ReadAsArray()
tandemx_bandmin=9999
for x in range(tandemx_xcells) :
    for y in range(tandemx_ycells) :
        if (tandemx_rasterArray[y][x] < tandemx_bandmin and tandemx_rasterArray[y][x] >-32700) :
            tandemx_bandmin=tandemx_rasterArray[y][x]
print(f"TanDEM-X raster covers longitude {tandemx_ulx:4.2f} to {tandemx_lrx:4.2f}, latitude {tandemx_uly:4.2f} to {tandemx_lry:7.2f}, min {tandemx_bandmin:7.2f}, max {tandemx_bandmax:4.2f}, {tandemx_xcells} by {tandemx_ycells} pixels")

k=number_in_colourbar
c = cmap_discretize('jet', k)
plotmin=300
plotmax=400
if not 'skip' in verbose :
    plt.imshow(tandemx_rasterArray,interpolation='nearest',cmap=c,vmin=plotmin,vmax=plotmax,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
    plt.colorbar()
    plt.title(f"TanDEM-X DSM")
    if verbose == 'plot' :
        plt.show()
    else :
        plt.savefig(output_directory+"TanDEM-X_raw.png",dpi=1200)               
    plt.close()

#output original TanDEM-X data to TIFF
tandemx_raw_cropped_FileName=output_directory+"TanDEM-X_raw.tif"
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(tandemx_raw_cropped_FileName, tandemx_xcells, tandemx_ycells, 1, gdal.GDT_Float32)
geotransform = (tandemx_ulx, tandemx_xres, 0, tandemx_uly, 0, tandemx_yres)
outdata.SetGeoTransform(geotransform)
outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(tandemx_rasterArray)
outdata.GetRasterBand(1).SetNoDataValue(tandemx_nodata)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!

# TanDEM-X datum adjustment
print("Adjusting TanDEM-X datum at "+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
for y_datum_adjustment in range(tandemx_ycells) :
    for x_datum_adjustment in range(tandemx_xcells) :
        tandemx_rasterArray[y_datum_adjustment,x_datum_adjustment]=tandemx_rasterArray[y_datum_adjustment,x_datum_adjustment]+tandemx_datum_adjustment

#output datum adjusted TanDEM-X data to TIFF
tandemx_raw_cropped_FileName=output_directory+"TanDEM-X_raw_datum_adjusted.tif"
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(tandemx_raw_cropped_FileName, tandemx_xcells, tandemx_ycells, 1, gdal.GDT_Float32)
geotransform = (tandemx_ulx, tandemx_xres, 0, tandemx_uly, 0, tandemx_yres)
outdata.SetGeoTransform(geotransform)
outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(tandemx_rasterArray)
outdata.GetRasterBand(1).SetNoDataValue(tandemx_nodata)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!

        
# Remove elevations in TanDEM-X close to water from classification

water_classification_buffer_pixels=4
water_classification_buffer_degrees=water_classification_buffer_pixels*classification_xres
print(f"Removing water-classified points, buffer {water_classification_buffer_pixels} pixels at "+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
y_water_buffer_ten_percent=int(tandemx_ycells/10)
for y_water_buffer in range(tandemx_ycells) :
    if (y_water_buffer/y_water_buffer_ten_percent).is_integer() :
        print (f"{100*y_water_buffer/tandemx_ycells:4.0f}% complete.")
    for x_water_buffer in range(tandemx_xcells) :
        lon_water_buffer, lat_water_buffer = coords_from_pixels(x_water_buffer,y_water_buffer,tandemx_ulx,tandemx_uly,tandemx_xres,tandemx_yres)
        classification_list=classification_proportions_percentage(lon_water_buffer-water_classification_buffer_degrees,lat_water_buffer-water_classification_buffer_degrees,lon_water_buffer+water_classification_buffer_degrees,lat_water_buffer+water_classification_buffer_degrees,classification_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)        
        if classification_list[1]>0.1 :
            tandemx_rasterArray[y_water_buffer,x_water_buffer]=tandemx_nodata

if not 'skip' in verbose :
    plt.imshow(tandemx_rasterArray,interpolation='nearest',cmap=c,vmin=plotmin,vmax=plotmax,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
    plt.title("Water-removed TanDEM-X")
    plt.colorbar()
    if verbose == 'plot' :
        plt.show()
    else :
        plt.savefig(output_directory+"TanDEM-X_raw_waterrem.png", format="png",dpi=1200)               
    plt.close()

#output cropped TanDEM-X data to TIFF
tandemx_raw_cropped_FileName=output_directory+"TanDEM-X_raw_waterrem.tif"
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(tandemx_raw_cropped_FileName, tandemx_xcells, tandemx_ycells, 1, gdal.GDT_Float32)
geotransform = (tandemx_ulx, tandemx_xres, 0, tandemx_uly, 0, tandemx_yres)
outdata.SetGeoTransform(geotransform)
outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(tandemx_rasterArray)
outdata.GetRasterBand(1).SetNoDataValue(tandemx_nodata)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!


# Remove outlier elevations in TanDEM-X from height information

print("Removing errors in TanDEM-x DSM...")

tandemx_error_removal="neither"
if tandemx_error_removal=="ceiling" :
    # Remove over ceiling
    tandemx_ceiling=500 # Max believable level (m) amsl in basin. May be higher in edges outside the basin.
    print("Removing TanDEM-X points over {tandemx_ceiling}m AMSL")
    y_crop_ten_percent=int(tandemx_ycells/10)
    for y_crop in range(tandemx_ycells) :
        if (y_crop/y_crop_ten_percent).is_integer() :
            print (f"{100*y_crop/tandemx_xcells:4.0f}% complete.")
        for x_crop in range(tandemx_xcells) :
            lon_crop, lat_crop = coords_from_pixels(x_crop,y_crop,tandemx_ulx,tandemx_uly,tandemx_xres,tandemx_yres)
            if (tandemx_rasterArray[y_crop,x_crop]>300 and lon_crop>16.612 and lon_crop<16.717 and lat_crop<-1.46 and lat_crop>-1.67) or (tandemx_rasterArray[y_crop,x_crop]>300 and lon_crop>18.250 and lon_crop<18.293 and lat_crop<-1.94 and lat_crop>-1.993) or tandemx_rasterArray[y_crop,x_crop] > tandemx_celing:
                #print(f"Found bad point {tandemx_rasterArray[y_crop,x_crop]}m")
                tandemx_rasterArray[y_crop,x_crop]=tandemx_nodata

elif tandemx_error_removal=="gradient" :
    tandemx_gradient_limit=100 # Maximum number of metres above local minimum allowed.
    minimum_distance=5
    print(f"Using gradient method, excluding TanDEM-X points {tandemx_gradient_limit}m higher than local minimum within {minimum_distance} pixels")
    y_crop_ten_percent=int(tandemx_ycells/10)
    for y_crop in range(tandemx_ycells) :
        if (y_crop/y_crop_ten_percent).is_integer() :
            print (f"{100*y_crop/tandemx_xcells:4.0f}% complete.")
        for x_crop in range(tandemx_xcells) :
            local_minimum=999
            for y_minima in range(y_crop-minimum_distance,y_crop+minimum_distance+1) :
                for x_minima in range(x_crop-minimum_distance,x_crop+minimum_distance+1) :
                    if y_minima>=0 and y_minima<tandemx_ycells and x_minima>=0 and x_minima<tandemx_xcells :
                        if tandemx_rasterArray[y_minima,x_minima]<local_minimum and tandemx_rasterArray[y_minima,x_minima] != tandemx_nodata :
                            local_minimum=tandemx_rasterArray[y_minima,x_minima]
            if tandemx_rasterArray[y_crop,x_crop]-local_minimum>tandemx_gradient_limit :
                tandemx_rasterArray[y_crop,x_crop]=tandemx_nodata
else :
    print("Didn't use gradient or ceiling methods")
    
if not 'skip' in verbose :
    plt.imshow(tandemx_rasterArray,interpolation='nearest',cmap=c,vmin=plotmin,vmax=plotmax,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
    plt.title("Cropped TanDEM-X")
    plt.colorbar()
    if verbose == 'plot' :
        plt.show()
    else :
        plt.savefig(output_directory+"TanDEM-X_raw_waterrem_cropped.png", format="png",dpi=1200)               
    plt.close()

#output cropped TanDEM-X data to TIFF
tandemx_raw_cropped_FileName=output_directory+"TanDEM-X_raw_waterrem_cropped.tif"
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(tandemx_raw_cropped_FileName, tandemx_xcells, tandemx_ycells, 1, gdal.GDT_Float32)
geotransform = (tandemx_ulx, tandemx_xres, 0, tandemx_uly, 0, tandemx_yres)
outdata.SetGeoTransform(geotransform)
outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(tandemx_rasterArray)
outdata.GetRasterBand(1).SetNoDataValue(tandemx_nodata)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!

# Fill in outlier gaps
tandemx_interpolation_distance=100*tandemx_cellsize
tandemx_rasterArray_original=tandemx_rasterArray.copy() # Copy and preserve the original for processing
interpolated_outFileName_tandemx=output_directory+"TanDEM-X_raw_waterrem_cropped_filled.tif"
shutil.copyfile(tandemx_raw_cropped_FileName,interpolated_outFileName_tandemx)
forinterp = gdal.Open(interpolated_outFileName_tandemx, GA_Update)
interpband=forinterp.GetRasterBand(1)
interpolation_distance=int(tandemx_interpolation_distance/tandemx_cellsize)
result = gdal.FillNodata(targetBand = interpband, maskBand = None, maxSearchDist = interpolation_distance, smoothingIterations = 0)
tandemx_rasterArray = forinterp.ReadAsArray()
forinterp = None

if not 'skip' in verbose :
    plt.imshow(tandemx_rasterArray,interpolation='nearest',cmap=c,vmin=plotmin,vmax=plotmax,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
    plt.title("Cropped and filled TanDEM-X")
    plt.colorbar()
    if verbose == 'plot' :
        plt.show()
    else :
        plt.savefig(output_directory+"TanDEM-X_raw_waterrem_cropped_filled.png", format="png",dpi=1200)               
    plt.close()

# Remove unclassified pixels

print(f"Removing unclassified points.")
y_unclassified_ten_percent=int(tandemx_ycells/10)
for y_unclassified in range(tandemx_ycells) :
    if (y_unclassified/y_unclassified_ten_percent).is_integer() :
        print (f"{100*y_unclassified/tandemx_xcells:4.0f}% complete.")
    for x_unclassified in range(tandemx_xcells) :
        lon_unclassified, lat_unclassified = coords_from_pixels(x_unclassified,y_unclassified,tandemx_ulx,tandemx_uly,tandemx_xres,tandemx_yres)
        classification_list=classification_proportions_percentage(lon_unclassified-tandemx_xres/2,lat_unclassified-tandemx_yres/2,lon_unclassified+tandemx_xres/2,lat_unclassified+tandemx_yres/2,classification_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)        
        if classification_list[0]>99 :
            tandemx_rasterArray[y_unclassified,x_unclassified]=tandemx_nodata

# Output filled-in TanDEM-X

if not 'skip' in verbose :
    plt.imshow(tandemx_rasterArray,interpolation='nearest',cmap=c,vmin=plotmin,vmax=plotmax,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
    plt.title("Cropped, filled and trimmed TanDEM-X")
    plt.colorbar()
    if verbose == 'plot' :
        plt.show()
    else :
        plt.savefig(output_directory+"TanDEM-X_raw_waterrem_cropped_filled_unclass-removed.png", format="png",dpi=1200)               
    plt.close()

tandemx_trimmed_FileName=output_directory+"TanDEM-X_raw_waterrem_cropped_filled_unclass-removed.tif"
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(tandemx_trimmed_FileName, tandemx_xcells, tandemx_ycells, 1, gdal.GDT_Float32)
geotransform = (tandemx_ulx, tandemx_xres, 0, tandemx_uly, 0, tandemx_yres)
outdata.SetGeoTransform(geotransform)
outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(tandemx_rasterArray)
outdata.GetRasterBand(1).SetNoDataValue(tandemx_nodata)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!

# Smooth TanDEM-X if necessary to remove high frequency noise

for tandemx_smoothing in [3] :
    for tandemx_resample_scale in [5] : # 5, 40,20,10
        # Resample TanDEM-X    
        tandemx_resampled_cellsize=tandemx_cellsize*tandemx_resample_scale # meters
        tandemx_resampled_xcells=int(tandemx_xcells/tandemx_resample_scale)
        tandemx_resampled_ycells=int(tandemx_ycells/tandemx_resample_scale)
        tandemx_resampled_rasterArray=np.zeros((tandemx_resampled_ycells,tandemx_resampled_xcells))

        # Reset output directory to output to specific smoothing and resolution
        output_directory=input_directory_root+"Work - Edinburgh\\TanDEM-X\\Wider download 90m\\CanopyRemoval\\"+version+"\\"+str(tandemx_resampled_cellsize)+"m_cellsize\\"+str(tandemx_smoothing)+"pixels_smoothing\\"
        if not os.path.isdir(output_directory) :
            os.makedirs(output_directory)

        if tandemx_resample_scale>1 :
            print(f"Resampling TanDEM-X to {tandemx_resampled_cellsize}m")
            for y_resample in range(tandemx_resampled_ycells) :
                for x_resample in range(tandemx_resampled_xcells) :
                    sum_scan=0
                    count_scan=0
                    for y_resample_scan in range(y_resample*tandemx_resample_scale,y_resample*tandemx_resample_scale+tandemx_resample_scale) :
                        for x_resample_scan in range(x_resample*tandemx_resample_scale,x_resample*tandemx_resample_scale+tandemx_resample_scale) :
                            if x_resample_scan<tandemx_xcells and y_resample_scan<tandemx_ycells :
                                if tandemx_rasterArray[y_resample_scan,x_resample_scan] != tandemx_nodata :
                                    sum_scan+=tandemx_rasterArray[y_resample_scan,x_resample_scan]
                                    count_scan+=1
                                    #print(f"[{x_resample_scan},{y_resample_scan}]={tandemx_rasterArray[y_resample_scan,x_resample_scan]}")
                                    
                    #sys.exit()
                    #print(f"[{x_resample},{y_resample}]={sum_scan/count_scan}")
                    if count_scan>0 :
                        tandemx_resampled_rasterArray[y_resample,x_resample] = sum_scan/count_scan
                    else :
                        tandemx_resampled_rasterArray[y_resample,x_resample] = tandemx_nodata
        else :
            tandemx_resampled_rasterArray=tandemx_rasterArray.copy()
        tandemx_resampled_xres=tandemx_xres*tandemx_resample_scale
        tandemx_resampled_yres=tandemx_yres*tandemx_resample_scale    

        # Smooth TanDEM-X
        if tandemx_smoothing>0 :
            print(f"Smoothing TanDEM-X by {tandemx_smoothing} pixels")
            tandemx_smoothed_rasterArray=tandemx_resampled_rasterArray.copy()
            for y_smooth in range(tandemx_resampled_ycells) :
                #if (y_smooth/(tandemx_resampled_ycells/100)).is_integer() :
                #print(f"Row {y_smooth}/{tandemx_resampled_ycells}")
                for x_smooth in range(tandemx_resampled_xcells) :
                    sum_for_smoothing=0
                    points_using=0
                    for y_within in range(y_smooth-tandemx_smoothing, y_smooth+tandemx_smoothing+1) :
                        for x_within in range(x_smooth-tandemx_smoothing, x_smooth+tandemx_smoothing+1) :
                            if y_within>=0 and y_within<tandemx_resampled_ycells and x_within>=0 and x_within<tandemx_resampled_xcells :
                                sum_for_smoothing+=tandemx_resampled_rasterArray[y_within,x_within]
                                points_using+=1
                    if points_using>2 :
                        tandemx_smoothed_rasterArray[y_smooth][x_smooth]=sum_for_smoothing/points_using
                        #print(f"Was {tandemx_resampled_rasterArray[y_smooth][x_smooth]} smoothed {tandemx_smoothed_rasterArray[y_smooth][x_smooth]:4.2f}")
                    else :
                        print("Problem smoothing, no cells")
                        sys.exit()
            tandemx_resampled_rasterArray=tandemx_smoothed_rasterArray.copy()
            
        if not 'skip' in verbose :
            plt.imshow(tandemx_resampled_rasterArray,interpolation='nearest',cmap=c,vmin=plotmin,vmax=plotmax,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
            plt.title("Resampled, smoothed TanDEM-X")
            plt.colorbar()
            if verbose == 'plot' :
                plt.show()
            else :
                plt.savefig(output_directory+"01-TanDEM-X_DSM_resampled_"+str(tandemx_resampled_cellsize)+"m_smoothed_"+str(tandemx_smoothing)+"_pixels.png", format="png",dpi=1200)               
            plt.close()
        #output resampled TanDEM-X data to TIFF
        outFileName=output_directory+"01-TanDEM-X_DSM_resampled_"+str(tandemx_resampled_cellsize)+"m_smoothed_"+str(tandemx_smoothing)+"_pixels.tif"
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(outFileName, tandemx_resampled_xcells, tandemx_resampled_ycells, 1, gdal.GDT_Float32)
        geotransform = (tandemx_ulx, tandemx_xres*tandemx_resample_scale, 0, tandemx_uly, 0, tandemx_yres*tandemx_resample_scale)
        outdata.SetGeoTransform(geotransform)
        outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(tandemx_resampled_rasterArray)
        outdata.GetRasterBand(1).SetNoDataValue(tandemx_nodata)##if you want these values transparent
        outdata.FlushCache() ##saves to disk!!

        #sys.exit()
        
        # Set up the array after canopy subtraction.
        tandemx_resampled_dtm_rasterArray=tandemx_resampled_rasterArray.copy()
        tandemx_resampled_dtm_rasterArray[:][:]=tandemx_nodata
        #sys.exit()

        for water_percentage_limit in [25] :
            #

            for which_beams in ['both'] : # ['strong','weak','both']
                if which_beams=="strong" :
                    beams_to_use=["GT1L","GT2L","GT3L"] # L= strong
                elif which_beams=="weak" :
                    beams_to_use=["GT1R","GT2R","GT3R"] # R=weak
                else :
                    beams_to_use=["GT1L","GT2L","GT3L","GT1R","GT2R","GT3R"] # L=strong, R=weak

                for ic2_canopy_type in ['mean'] : # ['mean','median','basic']
                    for ic2_analysis_type in ['mean'] : # ['mean','median','percentile75','percentile95']

                        print("Reading ICESat-2 data into 3D array")

                        icesat_lat=np.array([])
                        icesat_lon=np.array([])
                        icesat_canopy=np.array([])
                        icesat_count=0
                        icesat_outside_bounds=0
                        os.chdir(ICESat2ATL08_input_directory)
                        entries = os.scandir(ICESat2ATL08_input_directory)
                        # Create 3D array for ICESat-2 data
                        icesat_points_per_cell=int(2*100*(tandemx_resample_scale/5)**2)
                        icesat_3d=np.full([tandemx_resampled_ycells,tandemx_resampled_xcells,icesat_points_per_cell],canopy_nodata)
                        icesat_3d_next=np.full([tandemx_resampled_ycells,tandemx_resampled_xcells],0)
                        print(f"Created a {tandemx_resampled_ycells}x{tandemx_resampled_xcells}x{icesat_points_per_cell} array for IC2 data, analysing {which_beams} {ic2_canopy_type} canopy beams using {ic2_analysis_type} analysis.")

                        for entry in entries :
                            #print(f"Use <<{entry.name}>> ?")
                            #sys.exit()
                            if entry.name.startswith('ATL08_agg') and 'QC2' in entry.name and entry.name.endswith('.txt') and any(beam in entry.name.upper() for beam in beams_to_use) :
                                IC2_raw_input_file=open(entry,"r")
                                #print(f"Reading file {entry.name}")
                                
                                header_line=IC2_raw_input_file.readline()
                                line=IC2_raw_input_file.readline()
                                while (line !='') :
                                    # Line contains longitude_land, latitude_land, dem_h, h_te_median, h_te_mean, h_te_best_fit, h_te_interp, n_seg_ph, msw_flag, cloud_flag_atm, h_canopy, h_mean_canopy, h_median_canopy, canopy_flag, IC2_raw_input_file, track, beam, distance
                                    longitude_land=float(line.split(",")[0])
                                    latitude_land=float(line.split(",")[1])                            
                                        
                                    try :
                                        h_canopy=float(line.split(",")[10])
                                    except :
                                        h_canopy=float("nan")
                                    try :
                                        h_mean_canopy=float(line.split(",")[11])
                                    except :
                                        h_mean_canopy=float("nan")
                                    try :
                                        h_median_canopy=float(line.split(",")[12])
                                    except :
                                        h_median_canopy=float("nan")

                                    if ic2_canopy_type=='basic' :
                                        h_canopy_to_use=h_canopy
                                    elif ic2_canopy_type=='mean' :
                                        h_canopy_to_use=h_mean_canopy
                                    elif ic2_canopy_type=='median' :
                                        h_canopy_to_use=h_median_canopy
                                        
                                    if not math.isnan(h_canopy_to_use)  :
                                        which_xcell=int((longitude_land-tandemx_ulx)/tandemx_resampled_xres)  
                                        which_ycell=int((latitude_land-tandemx_uly)/tandemx_resampled_yres)
                                        #print(f"Point in cell {which_xcell},{which_ycell}")
                                        if which_xcell>=0 and which_ycell>=0 and which_xcell<tandemx_resampled_xcells and which_ycell<tandemx_resampled_ycells  :
                                            # Add to grid
                                            icesat_3d[which_ycell][which_xcell][icesat_3d_next[which_ycell][which_xcell]]=h_canopy_to_use
                                            icesat_3d_next[which_ycell][which_xcell]+=1
                                            if icesat_3d_next[which_ycell][which_xcell]>=icesat_points_per_cell :
                                                print(f"Too many ICESat-2 points per cell {icesat_3d_next[which_ycell][which_xcell]}")
                                                sys.exit()
                                            icesat_count+=1
                                            if (float(icesat_count)/100000).is_integer() :
                                                print(f"{icesat_count} points, eg. ({longitude_land}, {latitude_land}, {h_canopy_to_use}) into cell ({which_xcell},{which_ycell})")
                                        else :
                                            icesat_outside_bounds+=1
                                    line=IC2_raw_input_file.readline()
                                IC2_raw_input_file.close()
                        print(f"Have read in {icesat_count} ICESat-2 ({which_beams}) points, {icesat_outside_bounds} were outside bounds.")

                        print(f"Analysing {which_beams} beams, canopy type {ic2_canopy_type} using {ic2_analysis_type} analysis")

                        # Import intercept and slope for each class from file.
                        os.chdir(lidar_translation_class_directory)
                        entries = os.scandir(lidar_translation_class_directory)
                        intercept_array=np.full([7],-999.0)
                        slope_array=np.full([7],-999.0)
                        read_parameters=False
                        
                        for entry in entries :
                            #print(f"Use <<{entry.name}>> ?")
                            #sys.exit()
                            #print(f"LiDAR trans {entry.name}")
                            
                            
                            if entry.name.startswith('Results') and entry.name.endswith('.csv') and ( "canopy_type="+ic2_canopy_type in entry.name ) and ( "analysis="+ic2_analysis_type in entry.name) and ("beams="+which_beams in entry.name) and ("elev_dth="+str(max_ground_discontinuity_threshold)+"_" in entry.name) and ("class_pthr="+str(classification_purity_threshold)+"." in entry.name) :
                                cross_comparison_class_input_file=open(entry.name,"r")
                                header_line=cross_comparison_class_input_file.readline()
                                line=cross_comparison_class_input_file.readline()
                                while line!="" :
                                    class_input=int(line.split(",")[0])
                                    intercept_array[class_input]=float(line.split(",")[1])
                                    slope_array[class_input]=float(line.split(",")[2])
                                    line=cross_comparison_class_input_file.readline()
                                read_parameters=True
                        if -999 in intercept_array[1:] :
                            print("Couldn't get all parameters")
                            sys.exit()
                        else :
                            print(f"LiDAR translation Intercepts {intercept_array[1:]}, slopes {slope_array[1:]}")

                        output_file=open(output_directory+"Results_"+str(tandemx_resampled_cellsize)+"m_"+"canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+".txt","w")
                        output_file.write("lon_scan, lat_scan, max_ground_discontinuity, ic2_canopy_measure, ic2_point_count, canopy_height_estimate, local_tandemx_resampled_dsm, dtm_estimate, area_classification, area_classification_name, area_class_percentage, NoClass, Water, Savanna, Terra_Firme_Forest, Palm-dominated_swamp, Hardwood_swamp\n")

                        # raster scan of TanDEM-X DSM
                        
                        minimum_dtm=-999
                        maximum_dtm=-999
                        canopy_height_estimate_array=np.full([tandemx_resampled_ycells,tandemx_resampled_xcells],canopy_nodata, dtype=float)
                        canopy_height_estimate_array_per_class=np.full([6,tandemx_resampled_ycells,tandemx_resampled_xcells],canopy_nodata, dtype=float)
                        area_classification_array=np.full([tandemx_resampled_ycells,tandemx_resampled_xcells],0)
                        area_classification_percentage_array=np.full([tandemx_resampled_ycells,tandemx_resampled_xcells],-999)
                        area_classification_proportions=np.full([tandemx_resampled_ycells,tandemx_resampled_xcells,6],0.0)
                        ic2_point_count=np.full([tandemx_resampled_ycells,tandemx_resampled_xcells],0)
        
                        print("Running through TanDEM-X DSM array, calculating IC2 canopy, translating to LiDAR equivalent")
                        x_scan_ten_percent=int(tandemx_resampled_xcells/10)
                        for x_scan in range(tandemx_resampled_xcells) :
                            if (x_scan/x_scan_ten_percent).is_integer() :
                                print (f"{100*x_scan/tandemx_resampled_xcells:4.0f}% complete.")
                            for y_scan in range(tandemx_resampled_ycells) :
                                if tandemx_resampled_rasterArray[y_scan][x_scan]!=tandemx_nodata :
                                    # Check for discontinuities in ground
                                    lon_scan, lat_scan = coords_from_pixels(x_scan,y_scan,tandemx_ulx,tandemx_uly,tandemx_resampled_xres,tandemx_resampled_yres)
                                    max_ground_discontinuity=0
                                    if x_scan>0 :
                                        if (tandemx_resampled_rasterArray[y_scan][x_scan]!= tandemx_nodata) and (tandemx_resampled_rasterArray[y_scan][x_scan-1]!= tandemx_nodata) and abs(tandemx_resampled_rasterArray[y_scan][x_scan]-tandemx_resampled_rasterArray[y_scan][x_scan-1])>max_ground_discontinuity :
                                            max_ground_discontinuity=abs(tandemx_resampled_rasterArray[y_scan][x_scan]-tandemx_resampled_rasterArray[y_scan][x_scan-1])
                                    if x_scan<tandemx_resampled_xcells-1 :
                                        if (tandemx_resampled_rasterArray[y_scan][x_scan]!= tandemx_nodata) and (tandemx_resampled_rasterArray[y_scan][x_scan+1]!= tandemx_nodata) and abs(tandemx_resampled_rasterArray[y_scan][x_scan]-tandemx_resampled_rasterArray[y_scan][x_scan+1])>max_ground_discontinuity :
                                            max_ground_discontinuity=abs(tandemx_resampled_rasterArray[y_scan][x_scan]-tandemx_resampled_rasterArray[y_scan][x_scan+1])
                                    if y_scan>0 :
                                        if (tandemx_resampled_rasterArray[y_scan][x_scan]!= tandemx_nodata) and (tandemx_resampled_rasterArray[y_scan-1][x_scan]!= tandemx_nodata) and abs(tandemx_resampled_rasterArray[y_scan][x_scan]-tandemx_resampled_rasterArray[y_scan-1][x_scan])>max_ground_discontinuity :
                                            max_ground_discontinuity=abs(tandemx_resampled_rasterArray[y_scan][x_scan]-tandemx_resampled_rasterArray[y_scan-1][x_scan])
                                    if y_scan<tandemx_resampled_ycells-1 :
                                        if (tandemx_resampled_rasterArray[y_scan][x_scan]!= tandemx_nodata) and (tandemx_resampled_rasterArray[y_scan+1][x_scan]!= tandemx_nodata) and abs(tandemx_resampled_rasterArray[y_scan][x_scan]-tandemx_resampled_rasterArray[y_scan+1][x_scan])>max_ground_discontinuity :
                                            max_ground_discontinuity=abs(tandemx_resampled_rasterArray[y_scan][x_scan]-tandemx_resampled_rasterArray[y_scan+1][x_scan])
                                    # if max_ground_discontinuity>max_ground_discontinuity_threshold :
                                    if max_ground_discontinuity>1e6 : # V04 fudge to count all cells, then evaluate in Excel
                                        print(f"Ground elevation step {max_ground_discontinuity}")
                                    else :
                                        #print(f"TanDEM-X ground elevation step {max_ground_discontinuity:4.2f}")
                                        lon_scan, lat_scan = coords_from_pixels(x_scan,y_scan,tandemx_ulx,tandemx_uly,tandemx_resampled_xres,tandemx_resampled_yres)
                                        #print(f"Cell {x_scan}/{tandemx_resampled_xcells}, {y_scan}/{tandemx_resampled_ycells} - Window {lon_scan-tandemx_resampled_xres/2:6.3f} to {lon_scan+tandemx_resampled_xres/2:6.3f} , {lat_scan+tandemx_resampled_yres/2:6.3f} to {lat_scan-tandemx_resampled_yres/2:6.3f}, TanDEM-X ground elevation step {max_ground_discontinuity:4.2f}")
                                        # IC2 scan data for cell
                                        h_canopy_to_use_array=[]
                                        for shift in range(icesat_3d_next[y_scan][x_scan]) :
                                            h_canopy_to_use_array.append(icesat_3d[y_scan][x_scan][shift])
                                        
                                        # Calculate measures from raw IC2 data
                                        #print(f"{len(h_canopy_to_use_array)} IC2 points read")
                                        area_classification, area_class_percentage=area_most_common_class_lookup(lon_scan-tandemx_resampled_xres/2,lat_scan-tandemx_resampled_yres/2,lon_scan+tandemx_resampled_xres/2,lat_scan+tandemx_resampled_yres/2,classification_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)
                                        area_classification_array[y_scan][x_scan]=area_classification
                                        area_classification_percentage_array[y_scan][x_scan]=area_class_percentage
                                        classification_list=classification_proportions_percentage(lon_scan-tandemx_resampled_xres/2,lat_scan-tandemx_resampled_yres/2,lon_scan+tandemx_resampled_xres/2,lat_scan+tandemx_resampled_yres/2,classification_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)
                                        area_classification_proportions[y_scan][x_scan]=classification_list

                                        if len(h_canopy_to_use_array)>0 and area_classification>0:
                                            if ic2_analysis_type == 'mean' :
                                                ic2_canopy_measure=statistics.mean(h_canopy_to_use_array)
                                            elif ic2_analysis_type == 'median' :
                                                ic2_canopy_measure=statistics.mean(h_canopy_to_use_array)
                                            elif ic2_analysis_type == 'percentile75' :
                                                ic2_canopy_measure=np.percentile(h_canopy_to_use_array,75)
                                            elif ic2_analysis_type == 'percentile95' :
                                                ic2_canopy_measure=np.percentile(h_canopy_to_use_array,95)
                                            
                                            local_classification=lookup(lon_scan,lat_scan,classification_mostlikely_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)
                                            ic2_point_count[y_scan][x_scan]=len(h_canopy_to_use_array)

                                            #local_tandemx_resampled_dsm=lookup(lon_scan,lat_scan,tandemx_resampled_rasterArray,tandemx_resampled_xres,tandemx_resampled_yres,tandemx_resampled_ulx,tandemx_resampled_uly)

                                            #
                                            #
                                            canopy_height_estimate=slope_array[area_classification]*ic2_canopy_measure+intercept_array[area_classification] # This is for separate fit for each class
                                            #
                                            #
                                            #canopy_height_estimate_array[y_scan][x_scan]=canopy_height_estimate

                                            canopy_height_estimate_array[y_scan][x_scan]=canopy_height_estimate
                                            canopy_height_estimate_array_per_class[area_classification][y_scan][x_scan]=canopy_height_estimate
                                            if abs(canopy_height_estimate)>1000 or canopy_height_estimate<0 :
                                                print(f"Canopy est {canopy_height_estimate}")
                                                sys.exit()

                        if not 'skip' in verbose :
                            plt.imshow(canopy_height_estimate_array,interpolation='nearest',cmap=c,vmin=0,vmax=canopy_max_plot,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
                            plt.title(f"Canopy height, calculated from IC2, before interpolation, all classes")
                            plt.colorbar()    
                            if verbose == 'plot' :
                                plt.show()
                            else :
                                plt.savefig(output_directory+"02-Canopy_IC2-translated_before_interpolation_"+str(tandemx_resampled_cellsize)+"m_"+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+".png", format="png",dpi=1200)               
                            plt.close()
                        
                        print("Interpolating ICESat-2 canopy estimate")
 
                        for interpolation_class in range(6) :

                            # Output pre-interpolation images
                            outFileName_canopy=output_directory+"03-Canopy_IC2-translated_before_interpolation_"+str(tandemx_resampled_cellsize)+"m_"+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_class="+str(interpolation_class)+".tif"
                            driver = gdal.GetDriverByName("GTiff")
                            outdata = driver.Create(outFileName_canopy, tandemx_resampled_xcells, tandemx_resampled_ycells, 1, gdal.GDT_Float32)
                            geotransform = (tandemx_ulx, tandemx_xres*tandemx_resample_scale, 0, tandemx_uly, 0, tandemx_yres*tandemx_resample_scale)
                            outdata.SetGeoTransform(geotransform)
                            outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
                            outdata.GetRasterBand(1).WriteArray(canopy_height_estimate_array_per_class[interpolation_class])
                            outdata.GetRasterBand(1).SetNoDataValue(canopy_nodata)##if you want these values transparent
                            outdata.FlushCache() ##saves to disk!!

                            if not 'skip' in verbose :
                                plt.imshow(canopy_height_estimate_array_per_class[interpolation_class],interpolation='nearest',cmap=c,vmin=0,vmax=canopy_max_plot,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
                                plt.title(f"Canopy height, calculated from IC2, translated to LiDAR, before interpolation, class {interpolation_class}")
                                plt.colorbar()    
                                if verbose == 'plot' :
                                    plt.show()
                                else :
                                    plt.savefig(output_directory+"03-Canopy_IC2-translated_before_interpolation_"+str(tandemx_resampled_cellsize)+"m_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_class="+str(interpolation_class)+".png", format="png",dpi=1200)               
                                plt.close()

                            # Interpolate canopy array
                            
                            canopy_height_estimate_array_original=canopy_height_estimate_array_per_class[interpolation_class].copy() # Copy and preserve the original for processing
                            interpolated_outFileName_canopy=output_directory+"04-Canopy_IC2-translated_interpolated_"+str(tandemx_resampled_cellsize)+"m_"+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_water_excl="+str(water_percentage_limit)+"percent_class="+str(interpolation_class)+".tif"
                            shutil.copyfile(outFileName_canopy,interpolated_outFileName_canopy)
                            forinterp = gdal.Open(interpolated_outFileName_canopy, GA_Update)
                            interpband=forinterp.GetRasterBand(1)
                            interpolation_distance=int(canopy_interpolation_distance/tandemx_resampled_cellsize)
                            result = gdal.FillNodata(targetBand = interpband, maskBand = None, maxSearchDist = interpolation_distance, smoothingIterations = 0)
                            canopy_height_estimate_array_per_class[interpolation_class] = forinterp.ReadAsArray()
                            forinterp = None

                            if not 'skip' in verbose :
                                plt.imshow(canopy_height_estimate_array_per_class[interpolation_class],interpolation='nearest',cmap=c,vmin=0,vmax=canopy_max_plot,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
                                plt.title(f"Canopy height, calculated from IC2, translated to LiDAR, after interpolation, class {interpolation_class}")
                                plt.colorbar()    
                                if verbose == 'plot' :
                                    plt.show()
                                else :
                                    plt.savefig(output_directory+"04-Canopy_IC2-translated_after_interpolation_"+str(tandemx_resampled_cellsize)+"m_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_class="+str(interpolation_class)+".png", format="png",dpi=1200)               
                                plt.close()

                            outFileName_canopy=output_directory+"04-Canopy_IC2-translated_after_interpolation_"+str(tandemx_resampled_cellsize)+"m_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_class="+str(interpolation_class)+".tif"
                            driver = gdal.GetDriverByName("GTiff")
                            outdata = driver.Create(outFileName_canopy, tandemx_resampled_xcells, tandemx_resampled_ycells, 1, gdal.GDT_Float32)
                            geotransform = (tandemx_ulx, tandemx_xres*tandemx_resample_scale, 0, tandemx_uly, 0, tandemx_yres*tandemx_resample_scale)
                            outdata.SetGeoTransform(geotransform)
                            outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
                            outdata.GetRasterBand(1).WriteArray(canopy_height_estimate_array_per_class[interpolation_class])
                            outdata.GetRasterBand(1).SetNoDataValue(canopy_nodata)##if you want these values transparent
                            outdata.FlushCache() ##saves to disk!!

                        # Method 1 - paste in the relevant class canopy height to each cell, effectively saying that the canopy height in a cell is the same as the nearest cell of the same class
                        
                        
                        if canopy_technique=="majority" : # Use the majority class
                            for x_scan in range(tandemx_resampled_xcells) :
                                if (x_scan/x_scan_ten_percent).is_integer() :
                                    print (f"{100*x_scan/tandemx_resampled_xcells:4.0f}% complete.")
                                for y_scan in range(tandemx_resampled_ycells) :
                                    if area_classification_array[y_scan][x_scan]>1 :
                                        canopy_height_estimate_array[y_scan][x_scan]=canopy_height_estimate_array_per_class[area_classification_array[y_scan][x_scan]][y_scan][x_scan]
                                    else :
                                        canopy_height_estimate_array[y_scan][x_scan]=canopy_nodata
                        else : # Do weighted average of canopy elevation from type proportions 
                            if mix_diag=="yes" :
                                output_diag_file=open(output_directory+"Diagnostic.txt","w")
                            for x_scan in range(tandemx_resampled_xcells) :
                                if (x_scan/x_scan_ten_percent).is_integer() :
                                    print (f"{100*x_scan/tandemx_resampled_xcells:4.0f}% complete.")
                                for y_scan in range(tandemx_resampled_ycells) :
                                    lon_scan, lat_scan = coords_from_pixels(x_scan,y_scan,tandemx_ulx,tandemx_uly,tandemx_resampled_xres,tandemx_resampled_yres)
                                    classification_list=classification_proportions_percentage(lon_scan-tandemx_resampled_xres/2,lat_scan-tandemx_resampled_yres/2,lon_scan+tandemx_resampled_xres/2,lat_scan+tandemx_resampled_yres/2,classification_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)
                                    if area_classification_array[y_scan][x_scan]>1 :
                                        canopy_sum=0.0
                                        canopy_count=0.0
                                        for class_pick in range(2,6) :
                                            if canopy_height_estimate_array_per_class[class_pick][y_scan][x_scan]>0.0 :
                                                canopy_sum+=canopy_height_estimate_array_per_class[class_pick][y_scan][x_scan]*classification_list[class_pick]
                                                canopy_count+=classification_list[class_pick]
                                                if mix_diag=="yes" :
                                                    output_diag_file.write(f"{classification(class_pick)}, {canopy_height_estimate_array_per_class[class_pick][y_scan][x_scan]:4.1f} m, {classification_list[class_pick]:4.1f} %, ")
                                                #print(f"{classification(class_pick)} {canopy_height_estimate_array_per_class[class_pick][y_scan][x_scan]:4.1f} m, {classification_list[class_pick]:4.1f} %,",end="")
                                        if canopy_count>0 :
                                            canopy_height_estimate_array[y_scan][x_scan]=canopy_sum/canopy_count
                                        else :
                                            canopy_height_estimate_array[y_scan][x_scan]=canopy_nodata
                                        # Diagnostic
                                        if mix_diag=="yes" :
                                            output_diag_file.write(f"w-mean {canopy_height_estimate_array[y_scan][x_scan]:4.1f}, majority, {canopy_height_estimate_array_per_class[area_classification_array[y_scan][x_scan]][y_scan][x_scan]:4.1f}\n")
                                        #print(f"w-mean, {canopy_height_estimate_array[y_scan][x_scan]:4.1f}, majority {canopy_height_estimate_array_per_class[area_classification_array[y_scan][x_scan]][y_scan][x_scan]:4.1f}")
                                        #sys.exit()
                                    else :
                                        canopy_height_estimate_array[y_scan][x_scan]=canopy_nodata
                            if mix_diag=="yes" :
                                output_diag_file.close()            
                        # Output combined class map
                        if not 'skip' in verbose :
                            plt.imshow(canopy_height_estimate_array,interpolation='nearest',cmap=c,vmin=0,vmax=canopy_max_plot,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
                            plt.title(f"Canopy height, calculated from IC2, translated to LiDAR, after interpolation, combined classes")
                            plt.colorbar()    
                            if verbose == 'plot' :
                                plt.show()
                            else :
                                plt.savefig(output_directory+"05-Canopy_IC2-translated_after_interpolation_"+str(tandemx_resampled_cellsize)+"m_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_combined_classes.png", format="png",dpi=1200)               
                            plt.close()

                        outFileName_canopy=output_directory+"05-Canopy_IC2-translated_after_interpolation_"+str(tandemx_resampled_cellsize)+"m_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_combined_classes.tif"
                        driver = gdal.GetDriverByName("GTiff")
                        outdata = driver.Create(outFileName_canopy, tandemx_resampled_xcells, tandemx_resampled_ycells, 1, gdal.GDT_Float32)
                        geotransform = (tandemx_ulx, tandemx_xres*tandemx_resample_scale, 0, tandemx_uly, 0, tandemx_yres*tandemx_resample_scale)
                        outdata.SetGeoTransform(geotransform)
                        outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
                        outdata.GetRasterBand(1).WriteArray(canopy_height_estimate_array)
                        outdata.GetRasterBand(1).SetNoDataValue(canopy_nodata)##if you want these values transparent
                        outdata.FlushCache() ##saves to disk!!
                                                        
                        # Output class arrays

                        if not 'skip' in verbose :
                            plt.imshow(area_classification_array,interpolation='nearest',cmap=c,vmin=0,vmax=5,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
                            plt.title(f"Resampled classification")
                            plt.colorbar()
                            if verbose == 'plot' :
                                plt.show()
                            else :
                                plt.savefig(output_directory+"06-Classification_resampled_"+str(tandemx_resampled_cellsize)+"m.png", format="png",dpi=1200)               
                            plt.close()
                            
                        outFileName_class=output_directory+"06-Classification_resampled_"+str(tandemx_resampled_cellsize)+"m.tif"
                        driver = gdal.GetDriverByName("GTiff")
                        outdata = driver.Create(outFileName_class, tandemx_resampled_xcells, tandemx_resampled_ycells, 1, gdal.GDT_Float32)
                        geotransform = (tandemx_ulx, tandemx_xres*tandemx_resample_scale, 0, tandemx_uly, 0, tandemx_yres*tandemx_resample_scale)
                        outdata.SetGeoTransform(geotransform)
                        outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
                        outdata.GetRasterBand(1).WriteArray(area_classification_array)
                        outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
                        outdata.FlushCache() ##saves to disk!!

                        # Mask out water
                        #water_percentage_limit=100*(1.85*tandemx_resample_scale)**2
                        for y_scan in range(tandemx_resampled_ycells) :
                            for x_scan in range(tandemx_resampled_xcells) :
                                if area_classification_proportions[y_scan][x_scan][1]>=water_percentage_limit or area_classification_proportions[y_scan][x_scan][0]>0:
                                    canopy_height_estimate_array[y_scan][x_scan]=canopy_nodata

                        #output water-masked canopy to PNG
                        if not 'skip' in verbose :
                            plt.imshow(canopy_height_estimate_array,interpolation='nearest',cmap=c,vmin=0,vmax=canopy_max_plot,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
                            plt.title(f"Canopy height, interpreted from IC2, \ncalibrated from LiDAR, interpolated, water removed")
                            plt.colorbar()    
                            if verbose == 'plot' :
                                plt.show()
                            else :
                                plt.savefig(output_directory+"07-Canopy_interpolated_without_water_"+str(tandemx_resampled_cellsize)+"m_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+".png", format="png",dpi=1200)               
                            plt.close()

                        #output water-masked canopy to TIFF
                        outFileName=output_directory+"07-Canopy_interpolated_without_water_"+str(tandemx_resampled_cellsize)+"m_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+".tif"
                        driver = gdal.GetDriverByName("GTiff")
                        outdata = driver.Create(outFileName, tandemx_resampled_xcells, tandemx_resampled_ycells, 1, gdal.GDT_Float32)
                        geotransform = (tandemx_ulx, tandemx_xres*tandemx_resample_scale, 0, tandemx_uly, 0, tandemx_yres*tandemx_resample_scale)
                        outdata.SetGeoTransform(geotransform)
                        outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
                        outdata.GetRasterBand(1).WriteArray(canopy_height_estimate_array)
                        outdata.GetRasterBand(1).SetNoDataValue(tandemx_nodata)##if you want these values transparent
                        outdata.FlushCache() ##saves to disk!!
                        
                        # Subtract canopy from DSM

                        print("Subtracting canopy from TanDEM-X DSM to build DTM")                    
                        for x_scan in range(tandemx_resampled_xcells) :
                            #print (f"column {x_scan} of {lidar_canopy_xcells}")
                            for y_scan in range(tandemx_resampled_ycells) :
                                if tandemx_resampled_rasterArray[y_scan][x_scan]!=tandemx_nodata and canopy_height_estimate_array[y_scan][x_scan]!=canopy_nodata :
                                    local_tandemx_resampled_dsm=tandemx_resampled_rasterArray[y_scan][x_scan]
                                    dtm_estimate=local_tandemx_resampled_dsm-canopy_height_estimate_array[y_scan][x_scan]
                                    tandemx_resampled_dtm_rasterArray[y_scan][x_scan]=dtm_estimate
                                    if dtm_estimate>1000 :
                                        #print(f"DTM estimated as {dtm_estimate:4.2f}")
                                        sys.exit()
                                    line="{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(lon_scan, lat_scan, max_ground_discontinuity, ic2_canopy_measure, ic2_point_count[y_scan][x_scan], canopy_height_estimate, local_tandemx_resampled_dsm, dtm_estimate, area_classification_array[y_scan][x_scan], classification(area_classification_array[y_scan][x_scan]), area_classification_percentage_array[y_scan][x_scan],', '.join(map(str,area_classification_proportions[y_scan][x_scan])))
                                    #print(line)
                                    output_file.write(line+"\n")
                                    
                                    #icesat2_canopy_list=np.append(icesat2_canopy_list,ic2_canopy_measure)
                                    #if area_classification>0 :
                                    #    print(f"Cell {x_scan}/{tandemx_resampled_xcells}, {y_scan}/{tandemx_resampled_ycells} TanDEM-X-R {tandemx_resampled_rasterArray[y_scan][x_scan]:5.2f}, IC2 canopy {ic2_canopy_measure:5.2f} ({len(h_canopy_to_use_array)} pts), LiDAR-trans {canopy_height_estimate:4.2f}, DTM {dtm_estimate:4.2f}, Class {area_classification} {area_class_percentage:4.1f} %",', '.join(map(str,classification_list)))
                                    if minimum_dtm==-999 or dtm_estimate<minimum_dtm :
                                        minimum_dtm=dtm_estimate
                                    if maximum_dtm==-999 or dtm_estimate>maximum_dtm :
                                        maximum_dtm=dtm_estimate
                                else :
                                    #tandemx_resampled_dtm_rasterArray[y_scan][x_scan]=tandemx_resampled_rasterArray[y_scan][x_scan]
                                    tandemx_resampled_dtm_rasterArray[y_scan][x_scan]=tandemx_nodata

                        output_file.close()
                        if not 'skip' in verbose :
                            #plt.imshow(tandemx_resampled_dtm_rasterArray,interpolation='nearest',cmap=c,vmin=minimum_dtm,vmax=maximum_dtm,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
                            plt.imshow(tandemx_resampled_dtm_rasterArray,interpolation='nearest',cmap=c,vmin=250,vmax=400,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
                            plt.colorbar()
                            plt.title(f"TanDEM-X resampled, estimated DTM")
                            if verbose == 'plot' :
                                plt.show()
                            else :
                                plt.savefig(output_directory+"08-TanDEM-X_resampled_DTM_"+str(tandemx_resampled_cellsize)+"m_smoothed_"+str(tandemx_smoothing)+"_pixels_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_water_excl="+str(water_percentage_limit)+"percent.png", format="png",dpi=1200)               
                            plt.close()

                        #output resampled TanDEM-X data with canopy stripped off to TIFF
                        outFileName=output_directory+"08-TanDEM-X_resampled_DTM_"+str(tandemx_resampled_cellsize)+"m_smoothed_"+str(tandemx_smoothing)+"_pixels_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_water_excl="+str(water_percentage_limit)+"percent.tif"
                        driver = gdal.GetDriverByName("GTiff")
                        outdata = driver.Create(outFileName, tandemx_resampled_xcells, tandemx_resampled_ycells, 1, gdal.GDT_Float32)
                        geotransform = (tandemx_ulx, tandemx_xres*tandemx_resample_scale, 0, tandemx_uly, 0, tandemx_yres*tandemx_resample_scale)
                        outdata.SetGeoTransform(geotransform)
                        outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
                        outdata.GetRasterBand(1).WriteArray(tandemx_resampled_dtm_rasterArray)
                        outdata.GetRasterBand(1).SetNoDataValue(tandemx_nodata)##if you want these values transparent
                        outdata.FlushCache() ##saves to disk!!

                        # Do an interpolation to fill in missing water sections

                        tandemx_resampled_dtm_rasterArray_original=tandemx_resampled_dtm_rasterArray.copy() # Copy and preserve the original for processing
                        interpolated_outFileName_dtm=output_directory+"09-TanDEM-X_resampled_gapfilled_DTM_"+str(tandemx_resampled_cellsize)+"m_smoothed_"+str(tandemx_smoothing)+"_pixels_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_water_excl="+str(water_percentage_limit)+"percent.tif"
                        shutil.copyfile(outFileName,interpolated_outFileName_dtm)
                        forinterp = gdal.Open(interpolated_outFileName_dtm, GA_Update)
                        interpband=forinterp.GetRasterBand(1)
                        # How many pixels to interpolate?
                        #interpolation_distance=int(canopy_interpolation_distance/tandemx_resampled_cellsize)
                        interpolation_distance=20 # Just fill in one-pixel gaps
                        result = gdal.FillNodata(targetBand = interpband, maskBand = None, maxSearchDist = interpolation_distance, smoothingIterations = 0)
                        tandemx_resampled_dtm_gapfilled_rasterArray = forinterp.ReadAsArray()
                        forinterp = None


                        # Remove unclassified pixels

                        print(f"Removing unclassified points.")
                        y_unclassified_ten_percent=int(tandemx_resampled_ycells/10)
                        for y_unclassified in range(tandemx_resampled_ycells) :
                            if (y_unclassified/y_unclassified_ten_percent).is_integer() :
                                print (f"{100*y_unclassified/tandemx_xcells:4.0f}% complete.")
                            for x_unclassified in range(tandemx_resampled_xcells) :
                                lon_unclassified, lat_unclassified = coords_from_pixels(x_unclassified,y_unclassified,tandemx_ulx,tandemx_uly,tandemx_resampled_xres,tandemx_resampled_yres)
                                classification_list=classification_proportions_percentage(lon_unclassified-tandemx_resampled_xres/2,lat_unclassified-tandemx_resampled_yres/2,lon_unclassified+tandemx_resampled_xres/2,lat_unclassified+tandemx_resampled_yres/2,classification_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)        
                                if classification_list[0]>50 :
                                    tandemx_resampled_dtm_gapfilled_rasterArray[y_unclassified,x_unclassified]=tandemx_nodata

                        # Rewrite out to tiff
                        outFileName_dtm=output_directory+"09a-TanDEM-X_resampled_gapfilled_DTM_"+str(tandemx_resampled_cellsize)+"m_smoothed_"+str(tandemx_smoothing)+"_pixels_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_water_excl="+str(water_percentage_limit)+"percent.tif"
                        driver = gdal.GetDriverByName("GTiff")
                        outdata = driver.Create(outFileName_dtm, tandemx_resampled_xcells, tandemx_resampled_ycells, 1, gdal.GDT_Float32)
                        geotransform = (tandemx_ulx, tandemx_xres*tandemx_resample_scale, 0, tandemx_uly, 0, tandemx_yres*tandemx_resample_scale)
                        outdata.SetGeoTransform(geotransform)
                        outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
                        outdata.GetRasterBand(1).WriteArray(tandemx_resampled_dtm_gapfilled_rasterArray)
                        outdata.GetRasterBand(1).SetNoDataValue(tandemx_nodata)##if you want these values transparent
                        outdata.FlushCache() ##saves to disk!!


                        if not 'skip' in verbose :
                            plt.imshow(tandemx_resampled_dtm_gapfilled_rasterArray,interpolation='nearest',cmap=c,vmin=290,vmax=350,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
                            plt.title(f"DTM gap-filled")
                            plt.colorbar()    
                            if verbose == 'plot' :
                                plt.show()
                            else :
                                plt.savefig(output_directory+"09-TanDEM-X_resampled_gapfilled_DTM_"+str(tandemx_resampled_cellsize)+"m_smoothed_"+str(tandemx_smoothing)+"_pixels_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_water_excl="+str(water_percentage_limit)+"percent.png", format="png",dpi=1200)               
                            plt.close()
                        
                        print(f"Finished analysing {which_beams} {ic2_canopy_type} canopy beams using {ic2_analysis_type} analysis.")
                        print("Removing water areas...")
                        
                        # Now remove water areas
                        # raw tandem-x removed cells with 0.1 fraction water in the +/- 4 classification cells area
                        # For this, try neightbouring DTM cells, +/-1, check if 25% or greater
                        water_look_distance=900 # meters
                        
                        water_classification_removal_threshold=25 # percent limit, above this in the +/- 4 pixels area, remove cell
                        water_classification_buffer_degrees=(water_look_distance/40075000)*360
                        tandemx_resampled_dtm_gapfilled_water_removed_rasterArray=tandemx_resampled_dtm_gapfilled_rasterArray.copy()
                        y_crop_fraction=int(tandemx_resampled_ycells/10)
                            #print (f"column {x_scan} of {lidar_canopy_xcells}")
                        for y_scan in range(tandemx_resampled_ycells) :
                            if (y_scan/y_crop_fraction).is_integer() :
                                print (f"{100*y_scan/tandemx_resampled_ycells:4.0f}% complete.")
                            for x_scan in range(tandemx_resampled_xcells) :
                                # Decide if this cell is too close to a large body of water
                                lon_water_buffer, lat_water_buffer = coords_from_pixels(x_scan,y_scan,tandemx_ulx,tandemx_uly,tandemx_resampled_xres,tandemx_resampled_yres)
                                classification_list=classification_proportions_percentage(lon_water_buffer-water_classification_buffer_degrees,lat_water_buffer-water_classification_buffer_degrees,lon_water_buffer+water_classification_buffer_degrees,lat_water_buffer+water_classification_buffer_degrees,classification_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)        
                                if classification_list[1]>water_classification_removal_threshold :
                                    tandemx_resampled_dtm_gapfilled_water_removed_rasterArray[y_scan,x_scan]=tandemx_nodata
                        # Write to tiff, png

                        final_tiff_name="10-TanDEM-X_resampled_gapfilled_DTM_"+str(tandemx_resampled_cellsize)+"m_smoothed_"+str(tandemx_smoothing)+"_pixels_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_water_excl="+str(water_percentage_limit)+"percent_water_removed.tif"
                        outFileName=output_directory+final_tiff_name
                        driver = gdal.GetDriverByName("GTiff")
                        outdata = driver.Create(outFileName, tandemx_resampled_xcells, tandemx_resampled_ycells, 1, gdal.GDT_Float32)
                        geotransform = (tandemx_ulx, tandemx_xres*tandemx_resample_scale, 0, tandemx_uly, 0, tandemx_yres*tandemx_resample_scale)
                        outdata.SetGeoTransform(geotransform)
                        outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
                        outdata.GetRasterBand(1).WriteArray(tandemx_resampled_dtm_gapfilled_water_removed_rasterArray)
                        outdata.GetRasterBand(1).SetNoDataValue(tandemx_nodata)##if you want these values transparent
                        outdata.FlushCache() ##saves to disk!!

                        if not 'skip' in verbose :
                            plt.imshow(tandemx_resampled_dtm_gapfilled_water_removed_rasterArray,interpolation='nearest',cmap=c,vmin=290,vmax=350,extent=[tandemx_ulx,tandemx_lrx,tandemx_lry,tandemx_uly])
                            plt.title(f"DTM gap-filled, water-removed")
                            plt.colorbar()    
                            if verbose == 'plot' :
                                plt.show()
                            else :
                                plt.savefig(output_directory+"10-TanDEM-X_resampled_gapfilled_DTM_"+str(tandemx_resampled_cellsize)+"m_smoothed_"+str(tandemx_smoothing)+"_pixels_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+"_water_excl="+str(water_percentage_limit)+"percent_water_removed.png", format="png",dpi=1200)               
                            plt.close()
                        print("Completed DTM build")
                        print(final_tiff_name)

                        
print("Finishing up")

print(f"Garbage collection {gc.collect()}")
sys.exit()

