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
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
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
    for which_ycell in range(which_ycell_low,which_ycell_high+1,int(math.copysign(1,yres))) :
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
    all_values=[]
    for which_ycell in range(which_ycell_low,which_ycell_high+1,int(math.copysign(1,yres))) :
        for which_xcell in range(which_xcell_low,which_xcell_high+1) :
            all_values.append(class_data_array[which_ycell][which_xcell])
    #print(f"In most common fn {all_values}, {most_common(all_values)}")
    #sys.exit()
    #print(f"Values {all_values}")
    classification=most_common(all_values)
    count=0
    how_many=0
    for which_ycell in range(which_ycell_low,which_ycell_high+1,int(math.copysign(1,yres))) :
        for which_xcell in range(which_xcell_low,which_xcell_high+1) :
            count+=1
            if class_data_array[which_ycell][which_xcell]==classification :
                how_many+=1
    percentage=how_many*100/count
    #print (f"{how_many} of {count} = {percentage}")
    return(most_common(all_values)), percentage

def classification_proportions_percentage(lon_low,lat_low,lon_high,lat_high,class_data_array,xres,yres,ulx,uly) :
    which_xcell_low= int((lon_low-ulx)/xres)  
    which_xcell_high=int((lon_high-ulx)/xres)  
    which_ycell_low= int((lat_low-uly)/yres)
    which_ycell_high=int((lat_high-uly)/yres)
    all_sum=np.zeros(6)
    count=0
    for which_ycell in range(which_ycell_low,which_ycell_high+1,int(math.copysign(1,yres))) :
        for which_xcell in range(which_xcell_low,which_xcell_high+1) :
            classification=(class_data_array[which_ycell][which_xcell])
            all_sum[classification]+=1
            count+=1
            #print(f"All sum {all_sum}, count{count}")
    classification_list=np.zeros(6)
    for classification_count in range(6) :
        classification_list[classification_count]=100*all_sum[classification_count]/count
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
        3 :  "Terra_firme_forest",
        4 : "Palm-dominated_swamp",
        5 : "Hardwood swamp",
        }[number]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

pi=3.14159265
#os.getcwd()

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
verbose='skip' # plot for interactive, save for save to files, skip to produce no plots (avoid calls to test if matplotlib leaking)
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
version="Parallel_v05"
filled_tandemx="yes"
classification_convolution_pixels_radius=0
lidar_cellsize=500
number_in_colourbar=100

max_ground_discontinuity_threshold=2
ic2_subset_filter1='' # 'percentile95_Canopy_measure=basic'
ic2_subset_filter2='' # 'Widening=10_pixels'

atl08_version='4'
ICESat2ATL08_input_directory=input_directory_root+"Work - Edinburgh\\IceSat2\\ATL08 redo\\extractedv14_buffer20\\basin_Alienware\\percentile99_trim_1m\\ATL08v"+str(atl08_version)
tandemx_input_directory=input_directory_root+"Work - Edinburgh\\TanDEM-X\\Wider download 90m\\"
classification_input_directory=input_directory_root+"Work - Edinburgh\\Bart Classification Map\\"

if lidar_cellsize==1000 :
    lidar_input_directory=input_directory_root+"Work - Edinburgh\\DRC LiDAR\\Analysis\\Data\\Results_aggregated\\"
    version=version+"_1000m"
elif lidar_cellsize==500 :
    lidar_input_directory=input_directory_root+"Work - Edinburgh\\DRC LiDAR\\Analysis\\Data\\ResultsV18_aggregated_500m\\"
    version=version+"_500m"
else :
    print("Don't know lidar scale")
    sys.exit()

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
#nodata=tandemx_band.GetNoDataValue()
nodata=-32767
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
print(f"TanDEM-X raster covers longitude {tandemx_ulx:4.2f} to {tandemx_lrx:4.2f}, longitude {tandemx_uly:4.2f} to {tandemx_lry:7.2f}, min {tandemx_bandmin:7.2f}, max {tandemx_bandmax:4.2f}")
print(f"{tandemx_xcells} by {tandemx_ycells} pixels")
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
        plt.savefig(output_directory+"TanDEM-X.png", format="png",dpi=1200)               
    plt.close()

raster2=tandemx_rasterArray.copy()

#output copy of data raster2 to TIFF
#outFileName=output_directory+"TanDEM-X-nodataproc.tif"
#driver = gdal.GetDriverByName("GTiff")
#outdata = driver.Create(outFileName, tandemx_xcells, tandemx_ycells, 1, gdal.GDT_Float32)
#outdata.SetGeoTransform(tandemx_raster.GetGeoTransform())##sets same geotransform as input
#outdata.SetProjection(tandemx_raster.GetProjection())##sets same projection as input
#outdata.GetRasterBand(1).WriteArray(raster2)
#outdata.GetRasterBand(1).SetNoDataValue(nodata)##if you want these values transparent
#outdata.FlushCache() ##saves to disk!!

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
if not 'skip' in verbose :
    
    plt.imshow(classification_mostlikely_rasterArray,interpolation='nearest',cmap=c,vmin=0,vmax=6,extent=[classification_ulx,classification_lrx,classification_lry,classification_uly])   
    plt.colorbar()
    plt.title(f"Classes")
    if verbose == 'plot' :
        plt.show()
    else :
        plt.savefig(output_directory+"Classification.png", format="png",dpi=1200)               
    plt.close()

# Convolve classification to remove noise

classification_mostlikely_convolved_rasterArray=classification_mostlikely_rasterArray.copy()

log_step=int(classification_ycells/100)
if classification_convolution_pixels_radius>0 :
    print("Convolving classification map")
    log_step=int(classification_ycells/100)
    for class_y in range(classification_ycells) :
        if (class_y/log_step).is_integer() :
            print(f"Row {class_y} of {classification_ycells}, {(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))}")
        for class_x in range(classification_xcells) :
            class_list=[]
            for aggregate_y in range(class_y-classification_convolution_pixels_radius,class_y+classification_convolution_pixels_radius+1) :
                for aggregate_x in range(class_x-classification_convolution_pixels_radius,class_x+classification_convolution_pixels_radius+1) :
                    if aggregate_y>=0 and aggregate_y<classification_ycells and aggregate_x>=0 and aggregate_x<classification_xcells :
                        class_list.append(classification_mostlikely_rasterArray[aggregate_y][aggregate_x])
            classification_mostlikely_convolved_rasterArray[class_y][class_x]=most_common(class_list)
            
    if not 'skip' in verbose :
        
        plt.imshow(classification_mostlikely_convolved_rasterArray,interpolation='nearest',cmap=c,vmin=0,vmax=6,extent=[classification_ulx,classification_lrx,classification_lry,classification_uly])   
        plt.colorbar()
        plt.title(f"Classes")
        if verbose == 'plot' :
            plt.show()
        else :
            plt.savefig(output_directory+"Classification_convolved_"+str(classification_convolution_pixels_radius)+"pixels.png", format="png",dpi=1200)               
        plt.close()

    classification_mostlikely_rasterArray=classification_mostlikely_convolved_rasterArray.copy()

# Import all IC2 data

print("Reading ICESat-2 data")

array_size=int(2e6)
icesat2_lon_array=np.empty((array_size),float)
icesat2_lat_array=np.empty((array_size),float)
icesat2_h_canopy_array=np.empty((array_size),float)
icesat2_h_mean_canopy_array=np.empty((array_size),float)
icesat2_h_median_canopy_array=np.empty((array_size),float)
icesat2_beam_array=np.empty((array_size,4),str)
icesat2_beam_strength_array=np.empty((array_size),str) # 's' or 'w' for strong or weak
count=0
os.chdir(ICESat2ATL08_input_directory)
entries = os.scandir(ICESat2ATL08_input_directory)
for entry in entries :
    #print(f"Use <<{entry.name}>> ?")
    #sys.exit()
    if entry.name.startswith('ATL08_agg') and 'QC2' in entry.name and entry.name.endswith('.txt')  :
        IC2_raw_input_file=open(entry,"r")
        #print(f"Reading file {entry.name}")
        
        header_line=IC2_raw_input_file.readline()
        if "weak" in entry.name :
            icesat2_beam_strength="w"
        elif "strong" in entry.name :
            icesat2_beam_strength="s"
        else :
            print("Neither strong nor weak?")
            sys.exit()
            
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
            canopy_flag=int(line.split(",")[13])
            IC2_input_file=line.split(", ")[14]
            track=line.split(", ")[15]
            track_number=int(track)
            beam=line.split(", ")[16]
            beam=beam[:4]
            distance=line.split(", ")[17]

            if not math.isnan(h_canopy) and not math.isnan(h_mean_canopy) and not math.isnan(h_median_canopy) :
                #icesat2_lon_array=np.append(icesat2_lon_array,longitude_land)
                #icesat2_lat_array=np.append(icesat2_lat_array,latitude_land)
                #icesat2_h_canopy_array=np.append(icesat2_h_canopy_array,h_canopy)
                #icesat2_h_mean_canopy_array=np.append(icesat2_h_mean_canopy_array,h_mean_canopy)
                #icesat2_h_median_canopy_array=np.append(icesat2_h_median_canopy_array,h_median_canopy)
                #icesat2_beam_array=np.append(icesat2_beam_array,beam)

                icesat2_lon_array[count]=longitude_land
                icesat2_lat_array[count]=latitude_land
                icesat2_h_canopy_array[count]=h_canopy
                icesat2_h_mean_canopy_array[count]=h_mean_canopy
                icesat2_h_median_canopy_array[count]=h_median_canopy
                icesat2_beam_array[count,0]=beam[0]
                icesat2_beam_array[count,1]=beam[1]
                icesat2_beam_array[count,2]=beam[2]
                icesat2_beam_array[count,3]=beam[3]
                icesat2_beam_array[count]=icesat2_beam_strength
                #icesat2_beam_array[count]=beam
                
                count+=1
                if count>array_size :
                    print("Too much ICESat-2 data!")
                    sys.exit()
                if (count/100000).is_integer() :
                    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} Read in {count} points, eg {longitude_land:4.2f} {latitude_land:4.2f} {h_canopy:4.2f} {h_canopy:4.2f} {h_median_canopy:4.2f}) {beam} {IC2_input_file}")
            line=IC2_raw_input_file.readline()
        IC2_raw_input_file.close() 
number_of_icesat2_points=count
print(f"Imported {number_of_icesat2_points} IC2 points")
#sys.exit()
#
#
#

which_beams_list=['both','weak','strong']
ic2_canopy_type_list = ['mean','median','basic']
ic2_analysis_type_list = ['mean','median','percentile75','percentile95']

for class_heterogeneity_limit in [50] :
    for max_ground_discontinuity_threshold in [1e6] :
        #
        output_directory=input_directory_root+"Work - Edinburgh\\DRC LiDAR\\Analysis\\Cross-comparison\\"+version+"\\het_limit_"+str(class_heterogeneity_limit)+"pc\\ground_disc_limit_"+str(max_ground_discontinuity_threshold)+"m\\ATL08v"+str(atl08_version)+"\\"
        if not os.path.isdir(output_directory) :
            os.makedirs(output_directory)

        overall_results_filename=output_directory+"Overall_results_"+ic2_subset_filter1+"_"+ic2_subset_filter2+".txt"
        overall_results=open(overall_results_filename,"w")
        overall_results.write("ic2_canopy_type, ic2_analysis_type, which_beams, r-squared, intercept, slope, pairs, inverse_model_intercept, inverse_model_slope, mean_absolute_error, rmse_prediction, percentile95_error, max_error\n")



        #
        #for which_beams in which_beams_list : # ['strong','weak','both']
        #    if which_beams=="strong" :
        #        beams_to_use=["GT1L","GT2L","GT3L"] # L= strong
        #    elif which_beams=="weak" :
        #        beams_to_use=["GT1R","GT2R","GT3R"] # R=weak
        #    else :
        #        beams_to_use=["GT1L","GT2L","GT3L","GT1R","GT2R","GT3R"] # R=weak
        # 
        #    for ic2_canopy_type in ic2_canopy_type_list : # skip 'basic'
        #        for ic2_analysis_type in ic2_analysis_type_list :


        #print(f"Analysing {which_beams} beams, canopy type {ic2_canopy_type} using {ic2_analysis_type} analysis")
        print("Doing multi-parameter anakysis")
        lidar_canopy_list=np.array([])
        icesat2_canopy_list=np.array([])
        file_number = 0
        output_file_test=open(output_directory+"Test.txt","w")
        output_file_array=np.empty([100],type(output_file_test))
        number_of_output_files=0
        results_header="Lon, Lat, LiDAR_canopy, LiDAR_ground, max_LiDAR_ground_discontinuity, IC2_canopy, IC2_pts, TanDEM-X_local_DSM, TanDEM-X_area_DSM, local_classification, area_classification, area_classification_name, area_classification_percentage, LiDAR_file, NoClass, Water, Savanna, Terra_Firme_Forest, Palm-dominated_swamp, Hardwood_swamp"
        for which_beams in which_beams_list : # ['strong','weak','both']
            for ic2_canopy_type in ic2_canopy_type_list :
                for ic2_analysis_type in ic2_analysis_type_list :
                    output_file_array[file_number]=open(output_directory+"Results_canopy_type="+ic2_canopy_type+"_analysis="+ic2_analysis_type+"_beams="+which_beams+".csv","w")
                    output_file_array[file_number].write(results_header+"\n")
                    file_number+=1
        number_of_output_files=file_number

        # Read in the LiDAR canopy data, do comparison to IC2 canopy data
    
        for root, dirs, files in os.walk(lidar_input_directory, topdown=True):
            for name in files :
                if ("CanopyMap" in name) and (name.endswith('.tif')) :
                    print("Analysing "+name)
                    lidar_canopy_input_filename=os.path.join(root, name)
                    lidar_canopy_raster=gdal.Open(lidar_canopy_input_filename)
                    lidar_canopy_xcells=lidar_canopy_raster.RasterXSize
                    lidar_canopy_ycells=lidar_canopy_raster.RasterYSize
                    lidar_canopy_bands=lidar_canopy_raster.RasterCount
                    lidar_canopy_metadata=lidar_canopy_raster.GetMetadata()
                    lidar_canopy_band = lidar_canopy_raster.GetRasterBand(1)
                    lidar_canopy_stats=lidar_canopy_band.ComputeStatistics(0)
                    lidar_canopy_bandmin=lidar_canopy_band.GetMinimum()
                    lidar_canopy_bandmax=lidar_canopy_band.GetMaximum()
                    lidar_canopy_nodata=lidar_canopy_band.GetNoDataValue()
                    lidar_canopy_ulx, lidar_canopy_xres, lidar_canopy_xskew, lidar_canopy_uly, lidar_canopy_yskew, lidar_canopy_yres  = lidar_canopy_raster.GetGeoTransform()
                    lidar_canopy_lrx = lidar_canopy_ulx + (lidar_canopy_raster.RasterXSize * lidar_canopy_xres)
                    lidar_canopy_lry = lidar_canopy_uly + (lidar_canopy_raster.RasterYSize * lidar_canopy_yres)
                    lidar_canopy_rasterArray = lidar_canopy_raster.ReadAsArray()
                    # Get filename up to '_'
                    lidar_stem=''
                    stem_flag=1
                    for stem_build in range(len(name)) :
                        if name[stem_build]!='_' and stem_flag==1 :
                            lidar_stem+=name[stem_build]
                        else :
                            stem_flag=0
                    # Import ground map
                    filename_groundmap=''
                    for root_groundmap, dirs_groundmap, files_groundmap in os.walk(lidar_input_directory, topdown=True):
                        for name_groundmap in files_groundmap :
                            if (name_groundmap.startswith(lidar_stem)) and ("GroundMap" in name_groundmap) and (name_groundmap.endswith('.tif')) :
                                filename_groundmap=name_groundmap
                    
                    if filename_groundmap=='' :
                        print("No ground map found")
                        sys.exit()
                    else :
                        print(f"Ground map is {filename_groundmap}")    
                    
                    lidar_ground_input_filename=os.path.join(root_groundmap, filename_groundmap)
                    lidar_ground_raster=gdal.Open(lidar_ground_input_filename)
                    lidar_ground_xcells=lidar_ground_raster.RasterXSize
                    lidar_ground_ycells=lidar_ground_raster.RasterYSize
                    lidar_ground_bands=lidar_ground_raster.RasterCount
                    lidar_ground_metadata=lidar_ground_raster.GetMetadata()
                    lidar_ground_band = lidar_ground_raster.GetRasterBand(1)
                    lidar_ground_stats=lidar_ground_band.ComputeStatistics(0)
                    lidar_ground_bandmin=lidar_ground_band.GetMinimum()
                    lidar_ground_bandmax=lidar_ground_band.GetMaximum()
                    lidar_ground_nodata=lidar_ground_band.GetNoDataValue()
                    lidar_ground_ulx, lidar_ground_xres, lidar_ground_xskew, lidar_ground_uly, lidar_ground_yskew, lidar_ground_yres  = lidar_ground_raster.GetGeoTransform()
                    lidar_ground_lrx = lidar_ground_ulx + (lidar_ground_raster.RasterXSize * lidar_ground_xres)
                    lidar_ground_lry = lidar_ground_uly + (lidar_ground_raster.RasterYSize * lidar_ground_yres)
                    lidar_ground_rasterArray = lidar_ground_raster.ReadAsArray()
                    if lidar_ground_xcells != lidar_canopy_xcells or lidar_ground_ycells != lidar_canopy_ycells :
                        print("Canopy and ground files don't match")
                        sys.exit()
                    # End import ground map
                    
                    #print(f"lidar_canopy raster covers longitude {lidar_canopy_ulx:4.2f} to {lidar_canopy_lrx:4.2f}, longitude {lidar_canopy_uly:4.2f} to {lidar_canopy_lry:7.2f}, min {lidar_canopy_bandmin:7.2f}, max {lidar_canopy_bandmax:4.2f}")
                    #print(f"{lidar_canopy_xcells} by {lidar_canopy_ycells} pixels")
                    if not 'skip' in verbose :    
                        plotmax=50
                        plotmin=0
                        plt.imshow(lidar_canopy_rasterArray,interpolation='nearest',cmap=c,vmin=plotmin,vmax=plotmax,extent=[lidar_canopy_ulx,lidar_canopy_lrx,lidar_canopy_lry,lidar_canopy_uly])
                        plt.colorbar()
                        plt.title(f"LiDAR-derived canopy")
                        if verbose == 'plot' :
                            plt.show()
                        else :
                            plt.savefig(output_directory+name+".png", format="png",dpi=1200)               
                        plt.close()
                    
                    for x_scan in range(lidar_canopy_xcells) :
                        #print (f"column {x_scan} of {lidar_canopy_xcells}")
                        for y_scan in range(lidar_canopy_ycells) :
                            h_canopy_to_use_array=np.array([])
                            if lidar_canopy_rasterArray[y_scan][x_scan]!=lidar_canopy_nodata :
                                # Check for discontinuities in ground
                                max_ground_discontinuity=0
                                if x_scan>0 :
                                    if (lidar_ground_rasterArray[y_scan][x_scan]!= lidar_ground_nodata) and (lidar_ground_rasterArray[y_scan][x_scan-1]!= lidar_ground_nodata) and abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan][x_scan-1])>max_ground_discontinuity :
                                        max_ground_discontinuity=abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan][x_scan-1])
                                if x_scan<lidar_canopy_xcells-1 :
                                    if (lidar_ground_rasterArray[y_scan][x_scan]!= lidar_ground_nodata) and (lidar_ground_rasterArray[y_scan][x_scan+1]!= lidar_ground_nodata) and abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan][x_scan+1])>max_ground_discontinuity :
                                        max_ground_discontinuity=abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan][x_scan+1])
                                if y_scan>0 :
                                    if (lidar_ground_rasterArray[y_scan][x_scan]!= lidar_ground_nodata) and (lidar_ground_rasterArray[y_scan-1][x_scan]!= lidar_ground_nodata) and abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan-1][x_scan])>max_ground_discontinuity :
                                        max_ground_discontinuity=abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan-1][x_scan])
                                if y_scan<lidar_canopy_ycells-1 :
                                    if (lidar_ground_rasterArray[y_scan][x_scan]!= lidar_ground_nodata) and (lidar_ground_rasterArray[y_scan+1][x_scan]!= lidar_ground_nodata) and abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan+1][x_scan])>max_ground_discontinuity :
                                        max_ground_discontinuity=abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan+1][x_scan])
                                # if max_ground_discontinuity>max_ground_discontinuity_threshold :
                                if max_ground_discontinuity>max_ground_discontinuity_threshold : # V04 fudge to count all cells, then evaluate in Excel
                                    print(f"File {filename_groundmap} Ground elevation step {max_ground_discontinuity}, centre {lidar_ground_rasterArray[y_scan][x_scan]}, canopy estimate ignored")
                                else :                                       
                                    lon_scan, lat_scan = coords_from_pixels(x_scan,y_scan,lidar_canopy_ulx,lidar_canopy_uly,lidar_canopy_xres,lidar_canopy_yres)

                                    #
                                    #
                                    # Pull all relevant variables into new arrays
                                    h_canopy_to_use_basic_array=np.array([])
                                    h_canopy_to_use_mean_array=np.array([])
                                    h_canopy_to_use_median_array=np.array([])
                                    beam_type_use_array=np.array([])
                                    beam_strength_use_array=np.array([])
                                    
                                    for ic2_scan in range(number_of_icesat2_points) :
                                        if ( icesat2_lon_array[ic2_scan]>lon_scan-lidar_canopy_xres/2 ) and ( icesat2_lon_array[ic2_scan]<lon_scan+lidar_canopy_xres/2 ) and ( icesat2_lat_array[ic2_scan]>lat_scan-lidar_canopy_yres/2 ) and ( icesat2_lat_array[ic2_scan]<lat_scan+lidar_canopy_yres/2 ) :
                                            h_canopy_to_use_basic_array=np.append(h_canopy_to_use_basic_array, icesat2_h_canopy_array[ic2_scan])
                                            h_canopy_to_use_mean_array=np.append(h_canopy_to_use_mean_array, icesat2_h_mean_canopy_array[ic2_scan])
                                            h_canopy_to_use_median_array=np.append(h_canopy_to_use_median_array, icesat2_h_mean_canopy_array[ic2_scan])
                                            beam_type_use_array=np.append(beam_type_use_array, ''.join(icesat2_beam_array[ic2_scan]).upper())
                                            beam_strength_use_array=np.append(beam_strength_use_array,icesat2_beam_array[ic2_scan])
                                    if len(h_canopy_to_use_basic_array>0) :
                                        print("Found points")
                                        file_number=0
                                        for which_beams in which_beams_list : # ['strong','weak','both']
                                            if which_beams=="strong" :
                                                beams_to_use=["s"] # s= strong
                                            elif which_beams=="weak" :
                                                beams_to_use=["w"] # w=weak
                                            else :
                                                beams_to_use=["s","w"] # R=weak
                                            
                                            # Extract relevant beam types/analyses/canopy measures
                                            h_canopy_to_use_array=np.array([])
                                            for ic2_canopy_type in ic2_canopy_type_list : # skip 'basic'
                                                for beam_extract in range(len(h_canopy_to_use_basic_array)) :
                                                    if any (beam in (beam_strength_use_array[beam_extract]) for beam in beams_to_use) :
                                                        if ic2_canopy_type=='basic' :
                                                            h_canopy_to_use_array=np.append(h_canopy_to_use_array,h_canopy_to_use_basic_array[beam_extract])
                                                        elif ic2_canopy_type=='mean' :
                                                            h_canopy_to_use_array=np.append(h_canopy_to_use_array,h_canopy_to_use_mean_array[beam_extract])
                                                        elif ic2_canopy_type=='median' :
                                                            h_canopy_to_use_array=np.append(h_canopy_to_use_array,h_canopy_to_use_median_array[beam_extract])
                                                        else :
                                                            print("Unknown canopy measure")
                                                            sys.exit()
                                                #sys.exit()            
                                                for ic2_analysis_type in ic2_analysis_type_list :
                                                    
                                                    # Calculate measures from raw IC2 data
                                                    if len(h_canopy_to_use_array)>0 :
                                                        if ic2_analysis_type == 'mean' :
                                                            ic2_canopy_measure=statistics.mean(h_canopy_to_use_array)
                                                        elif ic2_analysis_type == 'median' :
                                                            ic2_canopy_measure=statistics.mean(h_canopy_to_use_array)
                                                        elif ic2_analysis_type == 'percentile75' :
                                                            ic2_canopy_measure=np.percentile(h_canopy_to_use_array,75)
                                                        elif ic2_analysis_type == 'percentile95' :
                                                            ic2_canopy_measure=np.percentile(h_canopy_to_use_array,95)
                                                        
                                                        local_classification=lookup(lon_scan,lat_scan,classification_mostlikely_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)
                                                        area_classification, area_class_percentage=area_most_common_class_lookup(lon_scan-lidar_canopy_xres/2,lat_scan-lidar_canopy_yres/2,lon_scan+lidar_canopy_xres/2,lat_scan+lidar_canopy_yres/2,classification_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)
                                                        classification_list=classification_proportions_percentage(lon_scan-lidar_canopy_xres/2,lat_scan-lidar_canopy_yres/2,lon_scan+lidar_canopy_xres/2,lat_scan+lidar_canopy_yres/2,classification_rasterArray,classification_xres,classification_yres,classification_ulx,classification_uly)
                                                        local_tandemx_dsm=lookup(lon_scan,lat_scan,tandemx_rasterArray,tandemx_xres,tandemx_yres,tandemx_ulx,tandemx_uly)
                                                        area_tandemx_dsm=area_mean_lookup(lon_scan-lidar_canopy_xres/2,lat_scan-lidar_canopy_yres/2,lon_scan+lidar_canopy_xres/2,lat_scan+lidar_canopy_yres/2,tandemx_rasterArray,tandemx_xres,tandemx_yres,tandemx_ulx,tandemx_uly, nodata) 
                                                        line="{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(lon_scan, lat_scan, lidar_canopy_rasterArray[y_scan][x_scan],lidar_ground_rasterArray[y_scan][x_scan], max_ground_discontinuity, ic2_canopy_measure, len(h_canopy_to_use_array), local_tandemx_dsm, area_tandemx_dsm, local_classification, area_classification, classification(area_classification), area_class_percentage, name,', '.join(map(str,classification_list)))
                                                        #print(line)
                                                        output_file_array[file_number].write(line+"\n")
                                                        #print(f"Written file {file_number}")
                                                        file_number+=1
                                                        lidar_canopy_list=np.append(lidar_canopy_list,lidar_canopy_rasterArray[y_scan][x_scan])
                                                        icesat2_canopy_list=np.append(icesat2_canopy_list,ic2_canopy_measure)
                                                        print(f"LiDAR {lidar_canopy_rasterArray[y_scan][x_scan]:5.2f}, LiDAR_grd_disc {max_ground_discontinuity:5.2f} IC2 {ic2_canopy_measure:5.2f} (from {len(h_canopy_to_use_array)} pts), Class {area_classification} {area_class_percentage:4.1f} %",', '.join(map(str,classification_list)))
                                                    else:
                                                        file_number+=1
                    lidar_canopy_raster=None            
        for file_number in range(number_of_output_files) :
            output_file_array[file_number].close()
        
        
print("Finishing up")
overall_results.close()            
print(f"Garbage collection {gc.collect()}")
sys.exit()

