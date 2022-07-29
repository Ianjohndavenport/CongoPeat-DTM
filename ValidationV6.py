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
version="v6"
filled_tandemx="yes"
classification_convolution_pixels_radius=0
lidar_cellsize=1000
number_in_colourbar=100
max_ground_discontinuity_threshold=2
aircraft_lidar_offset=-5.3 #metres derived from previous analysis
uav_lidar_offset=-15.75 # metres derived from previous analysis


atl08_version='4'
ICESat2ATL08_input_directory=input_directory_root+"Work - Edinburgh\\IceSat2\\ATL08 redo\\extractedv14_buffer20\\basin_Alienware\\percentile99_trim_1m\\ATL08v"+str(atl08_version)
tandemx_input_directory=input_directory_root+"Work - Edinburgh\\TanDEM-X\\Wider download 90m\\"
classification_input_directory=input_directory_root+"Work - Edinburgh\\Bart Classification Map\\"
source='hawker' # 'tandem-x' or 'hawker'

for source in 'hawker','tandem-x' :
    
    if source=='tandem-x' :
        dtm_directory=input_directory_root+"Work - Edinburgh\\TanDEM-X\\Wider download 90m\\CanopyRemoval\\AltAltAlt_v15\\"
        output_directory=input_directory_root+"Work - Edinburgh\\DRC LiDAR\\Validation\\"+version+"\\"
    elif source=='hawker' :
        dtm_directory=input_directory_root+"Work - Edinburgh\\Paper2\\Hawker\\"
        output_directory=input_directory_root+"Work - Edinburgh\\Paper2\\Hawker\\LiDARcomparison\\"
    else :
        print("Don't know source file")
        sys.exit()
    if not os.path.isdir(output_directory) :
        os.makedirs(output_directory)

        
    if lidar_cellsize==1000 :
        lidar_input_directory=input_directory_root+"Work - Edinburgh\\DRC LiDAR\\Analysis\\Data\\Results_aggregated\\Ground"
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

    # Inner loop - LiDAR ground estimates

    overall_results_filename=output_directory+"Validation_detailed_results_"+version+".csv"
    overall_results=open(overall_results_filename,"w")
    overall_results.write(f"DTM_file,LiDAR_file,lidar_ground,TanDEM_X_dtm, LIDAR-DTM\n")
    # Stats on dtm level, covering all lidar files
    summary_dtm_results_filename=output_directory+"Validation_dtm_summary_results_"+version+".csv"
    summary_dtm_results=open(summary_dtm_results_filename,"w")
    summary_dtm_results.write(f"Directory,DTM_file,mean_difference,mean_abs_difference,stdev,RMSD,npoints,mean_difference_aircraft,mean_abs_difference_aircraft,stdev_aircraft,RMSD_aircraft,npoints_aircraft,mean_difference_uav,mean_abs_difference_uav,stdev_uav,RMSD_uav,npoints_uav\n")
    # Finer grain stats, showing error per LiDAR file
    summary_lidar_results_filename=output_directory+"Validation_lidar_summary_results_"+version+".csv"
    summary_lidar_results=open(summary_lidar_results_filename,"w")
    summary_lidar_results.write(f"Directory,DTM_file,LiDAR_file,mean_difference,mean_abs_difference,stdev,RMSD,npoints,lon,lat\n")
    for unspecified_change in [50] :
        # Outer loop - DTMs
        os.chdir(dtm_directory)

        #entries = os.scandir(dtm_directory)
        #for entry in entries :
        for root_dtm, dirs_dtm, files_dtm in os.walk(dtm_directory, topdown=True):
            for name_dtm in files_dtm :
                #print(f"Checking {entry.name}")
                if (source=='hawker' and name_dtm.startswith('Hawker') and name_dtm.endswith('.tif') ) or (source=='tandem-x' and name_dtm.startswith('09-TanDEM-X_resampled_gapfilled_DTM') and name_dtm.endswith('.tif') ):
                    dtm_input_filename=os.path.join(root_dtm,name_dtm)
                    # Read in DTM
                    print(f"Analysing {name_dtm}")
                    
                    # DTM stats
                    dtm_difference_sum=0
                    dtm_difference_sq_sum=0
                    dtm_difference_count=0
                    dtm_diff_array=np.array([])
                    dtm_sq_diff_array=np.array([])
                    dtm_sq_diff_uav_array=np.array([])
                    dtm_sq_diff_aircraft_array=np.array([])
                    dtm_abs_diff_array=np.array([])
                    dtm_diff_aircraft_array=np.array([])
                    dtm_abs_diff_aircraft_array=np.array([])
                    dtm_diff_uav_array=np.array([])
                    dtm_abs_diff_uav_array=np.array([])
                    
                    
                    dtm_raster=gdal.Open(dtm_input_filename)
                    dtm_xcells=dtm_raster.RasterXSize
                    dtm_ycells=dtm_raster.RasterYSize
                    dtm_bands=dtm_raster.RasterCount
                    dtm_metadata=dtm_raster.GetMetadata()
                    dtm_band = dtm_raster.GetRasterBand(1)
                    dtm_stats=dtm_band.ComputeStatistics(0)
                    #nodata=dtm_band.GetNoDataValue()
                    nodata=-32767
                    dtm_bandmin=dtm_band.GetMinimum()
                    dtm_bandmax=dtm_band.GetMaximum()
                    dtm_ulx, dtm_xres, dtm_xskew, dtm_uly, dtm_yskew, dtm_yres  = dtm_raster.GetGeoTransform()
                    dtm_lrx = dtm_ulx + (dtm_raster.RasterXSize * dtm_xres)
                    dtm_lry = dtm_uly + (dtm_raster.RasterYSize * dtm_yres)
                    dtm_rasterArray = dtm_raster.ReadAsArray()
                    dtm_bandmin=9999
                    for x in range(dtm_xcells) :
                        for y in range(dtm_ycells) :
                            if (dtm_rasterArray[y][x] < dtm_bandmin and dtm_rasterArray[y][x] >-32700) :
                                dtm_bandmin=dtm_rasterArray[y][x]
                    if not 'skip' in verbose :
                        print(f"DTM raster {dtm_input_filename} covers longitude {dtm_ulx:4.2f} to {dtm_lrx:4.2f}, longitude {dtm_uly:4.2f} to {dtm_lry:7.2f}, min {dtm_bandmin:7.2f}, max {dtm_bandmax:4.2f}")
                        print(f"{dtm_xcells} by {dtm_ycells} pixels")
                    
                    # Read in the LiDAR canopy data, do comparison to IC2 canopy data
                
                    for root, dirs, files in os.walk(lidar_input_directory, topdown=True):
                        for name in files :
                            if ("GroundMap" in name) and (name.endswith('.tif')) :
                                if not 'skip' in verbose :
                                    print("Analysing "+name)
                                lidar_stats_diff_sum=0
                                lidar_stats_abs_diff_sum=0
                                lidar_stats_diff_sq_sum=0
                                lidar_stats_diff_count=0
                                lidar_diff_array=np.array([])
                                lidar_square_diff_array=np.array([])
                                lidar_abs_diff_array=np.array([])
                                
                                lidar_ground_input_filename=os.path.join(lidar_input_directory, name)
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

                                if not 'skip' in verbose :
                                    print(f"lidar_ground raster covers longitude {lidar_ground_ulx:4.2f} to {lidar_ground_lrx:4.2f}, longitude {lidar_ground_uly:4.2f} to {lidar_ground_lry:7.2f}, min {lidar_ground_bandmin:7.2f}, max {lidar_ground_bandmax:4.2f}")
                                    print(f"{lidar_ground_xcells} by {lidar_ground_ycells} pixels")
                                #sys.exit()
                                if not 'skip' in verbose :    
                                    plotmax=290
                                    plotmin=350
                                    plt.imshow(lidar_ground_rasterArray,interpolation='nearest',cmap=c,vmin=plotmin,vmax=plotmax,extent=[lidar_ground_ulx,lidar_ground_lrx,lidar_ground_lry,lidar_ground_uly])
                                    plt.colorbar()
                                    plt.title(f"LiDAR-derived ground")
                                    if verbose == 'plot' :
                                        plt.show()
                                    else :
                                        plt.savefig(output_directory+name+".png", format="png",dpi=1200)               
                                    plt.close()
                                
                                for x_scan in range(lidar_ground_xcells) :
                                    #print (f"column {x_scan} of {lidar_ground_xcells}")
                                    for y_scan in range(lidar_ground_ycells) :
                                        h_ground_to_use_array=np.array([])
                                        if lidar_ground_rasterArray[y_scan][x_scan]!=lidar_ground_nodata :
                                            # Check for discontinuities in ground
                                            max_ground_discontinuity=0
                                            if x_scan>0 :
                                                if (lidar_ground_rasterArray[y_scan][x_scan]!= lidar_ground_nodata) and (lidar_ground_rasterArray[y_scan][x_scan-1]!= lidar_ground_nodata) and abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan][x_scan-1])>max_ground_discontinuity :
                                                    max_ground_discontinuity=abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan][x_scan-1])
                                            if x_scan<lidar_ground_xcells-1 :
                                                if (lidar_ground_rasterArray[y_scan][x_scan]!= lidar_ground_nodata) and (lidar_ground_rasterArray[y_scan][x_scan+1]!= lidar_ground_nodata) and abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan][x_scan+1])>max_ground_discontinuity :
                                                    max_ground_discontinuity=abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan][x_scan+1])
                                            if y_scan>0 :
                                                if (lidar_ground_rasterArray[y_scan][x_scan]!= lidar_ground_nodata) and (lidar_ground_rasterArray[y_scan-1][x_scan]!= lidar_ground_nodata) and abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan-1][x_scan])>max_ground_discontinuity :
                                                    max_ground_discontinuity=abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan-1][x_scan])
                                            if y_scan<lidar_ground_ycells-1 :
                                                if (lidar_ground_rasterArray[y_scan][x_scan]!= lidar_ground_nodata) and (lidar_ground_rasterArray[y_scan+1][x_scan]!= lidar_ground_nodata) and abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan+1][x_scan])>max_ground_discontinuity :
                                                    max_ground_discontinuity=abs(lidar_ground_rasterArray[y_scan][x_scan]-lidar_ground_rasterArray[y_scan+1][x_scan])
                                            # if max_ground_discontinuity>max_ground_discontinuity_threshold :
                                            if max_ground_discontinuity>max_ground_discontinuity_threshold : # V04 fudge to count all cells, then evaluate in Excel
                                                if not 'skip' in verbose :
                                                    print(f"File {name} Ground elevation step {max_ground_discontinuity}, centre {lidar_ground_rasterArray[y_scan][x_scan]}, ground estimate ignored")
                                            else :                                       
                                                lon_scan, lat_scan = coords_from_pixels(x_scan,y_scan,lidar_ground_ulx,lidar_ground_uly,lidar_ground_xres,lidar_ground_yres)
                                                # Find DTM point, compare
                                                local_dtm=lookup(lon_scan,lat_scan,dtm_rasterArray,dtm_xres,dtm_yres,dtm_ulx,dtm_uly)
                                                difference=lidar_ground_rasterArray[y_scan][x_scan]-local_dtm
                                                if name.startswith("P") :
                                                    difference-=aircraft_lidar_offset
                                                else :
                                                    difference-=uav_lidar_offset
                                                if not 'skip' in verbose :
                                                    print(f"LiDAR ground {lidar_ground_rasterArray[y_scan][x_scan]:5.2f}, DTM {local_dtm:5.2f}, difference {difference:5.2f},{lon_scan:4.2f},{lat_scan:4.2f}")
                                                overall_results.write(f"{root_dtm},{name_dtm},{name},{lidar_ground_rasterArray[y_scan][x_scan]:5.2f},{local_dtm:5.2f},{difference:5.2f},{lon_scan:4.2f},{lat_scan:4.2f}\n")
                                                lidar_stats_diff_sum+=difference
                                                lidar_stats_abs_diff_sum+=abs(difference)
                                                lidar_stats_diff_sq_sum+=difference*difference
                                                lidar_stats_diff_count+=1
                                                lidar_diff_array=np.append(lidar_diff_array,difference)
                                                lidar_square_diff_array=np.append(lidar_square_diff_array,difference*difference)
                                                lidar_abs_diff_array=np.append(lidar_abs_diff_array,abs(difference))
                                                dtm_diff_array=np.append(dtm_diff_array,difference)
                                                dtm_abs_diff_array=np.append(dtm_abs_diff_array,abs(difference))
                                                dtm_sq_diff_array=np.append(dtm_sq_diff_array,difference*difference)
                                                if name.startswith("P") :
                                                    dtm_diff_aircraft_array=np.append(dtm_diff_aircraft_array,difference)
                                                    dtm_abs_diff_aircraft_array=np.append(dtm_abs_diff_aircraft_array,abs(difference))
                                                    dtm_sq_diff_aircraft_array=np.append(dtm_sq_diff_aircraft_array,difference*difference)
                                                else :
                                                    dtm_diff_uav_array=np.append(dtm_diff_uav_array,difference)
                                                    dtm_abs_diff_uav_array=np.append(dtm_abs_diff_uav_array,abs(difference))
                                                    dtm_sq_diff_uav_array=np.append(dtm_sq_diff_uav_array,difference*difference)
                                                    
                                # Output stats for LiDAR area
                                lidar_mean_diff=np.average(lidar_diff_array)
                                lidar_mean_abs_diff=np.average(lidar_abs_diff_array)
                                lidar_rmse_diff=np.sqrt(np.average(lidar_square_diff_array))
                                lidar_std_diff=np.std(lidar_diff_array)
                                if not 'skip' in verbose :
                                    print(f"{root_dtm},{name_dtm},{name},{lidar_mean_diff},{lidar_mean_abs_diff},{lidar_std_diff},{lidar_rmse_diff},{lidar_ground_ulx:4.2f},{lidar_ground_uly:4.2f}")
                                summary_lidar_results.write(f"{root_dtm},{name_dtm},{name},{lidar_mean_diff},{lidar_mean_abs_diff},{lidar_std_diff},{lidar_rmse_diff},{len(lidar_diff_array)},{lidar_ground_ulx:4.2f},{lidar_ground_uly:4.2f}\n")
                                # Close LiDAR TIFF
                                lidar_canopy_raster=None            
                    dtm_mean_diff=np.average(dtm_diff_array)
                    dtm_mean_abs_diff=np.average(dtm_abs_diff_array)
                    dtm_std_diff=np.std(dtm_diff_array)
                    dtm_rmse_diff=np.sqrt(np.average(dtm_sq_diff_array))
                    dtm_rmse_aircraft_diff=np.sqrt(np.average(dtm_sq_diff_aircraft_array))
                    dtm_rmse_uav_diff=np.sqrt(np.average(dtm_sq_diff_uav_array))
                    dtm_mean_diff_aircraft=np.average(dtm_diff_aircraft_array)
                    dtm_mean_abs_diff_aircraft=np.average(dtm_abs_diff_aircraft_array)
                    dtm_std_diff_aircraft=np.std(dtm_diff_aircraft_array)
                    dtm_mean_diff_uav=np.average(dtm_diff_uav_array)
                    dtm_mean_abs_diff_uav=np.average(dtm_abs_diff_uav_array)
                    dtm_std_diff_uav=np.std(dtm_diff_uav_array)
                    
                    if not 'skip' in verbose :
                        print(f"{dtm_input_filename},{dtm_mean_diff},{dtm_mean_abs_diff},{dtm_std_diff},{dtm_rmse_diff}")
                    summary_dtm_results.write(f"{root_dtm},{name_dtm},{dtm_mean_diff},{dtm_mean_abs_diff},{dtm_std_diff},{dtm_rmse_diff},{len(dtm_diff_array)},{dtm_mean_diff_aircraft},{dtm_mean_abs_diff_aircraft},{dtm_std_diff_aircraft},{dtm_rmse_aircraft_diff},{len(dtm_diff_aircraft_array)},{dtm_mean_diff_uav},{dtm_mean_abs_diff_uav},{dtm_std_diff_uav},{dtm_rmse_uav_diff},{len(dtm_diff_uav_array)}\n")
                    summary_dtm_results.flush()
                    summary_lidar_results.flush()
                    overall_results.flush()
    print("Finishing up")
    summary_dtm_results.close()
    summary_lidar_results.close()
    overall_results.close()
del dtm_rasterArray
print(f"Garbage collection {gc.collect()}")
sys.exit()

