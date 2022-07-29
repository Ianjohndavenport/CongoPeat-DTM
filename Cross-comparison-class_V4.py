# Second stage analysis after AltCrossComparison to look at the relations within different classes, mainly to distinguish palm from hardwood swamp.

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

def classification(number) :
    return {
        0 : "No_data",
        1 : "Water",
        2 : "Savanna",
        3 :  "Terra_firme_forest",
        4 : "Palm-dominated_swamp",
        5 : "Hardwood swamp",
        6 : "All_non-water"
        }[number]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

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
version="v04"
verbose='skip' # plot for interactive, save for save to files, skip to produce no plots (avoid calls to test if matplotlib leaking)
atl08_version='4' # v4 or v4 analysis

for lidar_cellsize in [1000] : # also 500

    if lidar_cellsize==1000 :
        cross_comparison_input_directory=input_directory_root+"Work - Edinburgh\\DRC LiDAR\\Analysis\\Cross-comparison\\Parallel_v04_1000m\\het_limit_50pc\\ground_disc_limit_1000000.0m\\ATL08v"+str(atl08_version)+"\\"
    elif lidar_cellsize==500 :
        cross_comparison_input_directory=input_directory_root+"Work - Edinburgh\\DRC LiDAR\\Analysis\\Cross-comparison\\Parallel_v03_500m\\het_limit_50pc\\ground_disc_limit_1000000.0m\\ATL08v"+str(atl08_version)+"\\"
    else :
        print("Don't know lidar scale")
        sys.exit()

    output_directory=cross_comparison_input_directory+"cross-comparison-class_"+version+"\\"
    if not os.path.isdir(output_directory) :
        os.makedirs(output_directory)
    os.chdir(cross_comparison_input_directory)


    summary_filename=output_directory+"Cross-comparison_summary_"+str(lidar_cellsize)+"m_"+version+".csv"
    summary_file=open(summary_filename,"w")
    summary_file.write("File,discontinuity_threshold(m),classification_purity_threshold(%),")
    for filter_class in range(7) :
        summary_file.write(f"{classification(filter_class)}_RMSE,{classification(filter_class)}_pairs,{classification(filter_class)}_r-squared,")
    summary_file.write("mean_non_water_rmse\n")

    for classification_purity_threshold in [0,50,75] :
        for max_ground_discontinuity_threshold in [0.5,1.0,1.5,2.0] :
            entries = os.scandir(cross_comparison_input_directory)
            for entry in entries :
                #print(f"Found file {entry.name}")
                if entry.name.startswith("Results") :
                    print(f"Opening file {entry.name}, ground discontinuity threshold {max_ground_discontinuity_threshold}")
                    
                    # Open input file
                    cross_comparison_input_file_name=entry.name
                    cross_comparison_input_file=open(cross_comparison_input_directory+cross_comparison_input_file_name)
                    summary_file.write(cross_comparison_input_file_name+",")
                    summary_file.write(f"{max_ground_discontinuity_threshold},{classification_purity_threshold},")
                    # Open output analysis file
                    output_filename=entry.name[:-4]+"_elev_dth="+str(max_ground_discontinuity_threshold)+"_class_pthr="+str(classification_purity_threshold)+".csv"
                    output_file=open(output_directory+output_filename,"w")
                    output_file.write("Class,intercept,slope,pairs,RMSE,99th_percentile,Max_error\n")

                    # Read in data

                    lidar_canopy_list=np.array([])
                    max_ground_discontinuity_list=np.array([])
                    icesat2_canopy_list=np.array([])
                    classification_area_list=np.array([],dtype=int)
                    classification_purity_percentage_list=np.array([])

                    header_line=cross_comparison_input_file.readline()
                    line=cross_comparison_input_file.readline()
                    
                    while (line !='' and not line.startswith("IC2")) :
                        lidar_canopy=float(line.split(",")[2])
                        max_ground_discontinuity=float(line.split(",")[3])
                        icesat2_canopy=float(line.split(",")[4])
                        classification_area=int(line.split(",")[8])
                        classification_area_name=(line.split(",")[9]).strip()
                        classification_purity_percentage=float(line.split(",")[10])

                        lidar_canopy_list=np.append(lidar_canopy_list,lidar_canopy)
                        max_ground_discontinuity_list=np.append(max_ground_discontinuity_list,max_ground_discontinuity)
                        icesat2_canopy_list=np.append(icesat2_canopy_list,icesat2_canopy)
                        classification_area_list=np.append(classification_area_list,classification_area)
                        classification_purity_percentage_list=np.append(classification_purity_percentage_list,classification_purity_percentage)
                        #print(f"Read in {lidar_canopy:4.2f} LiDAR, {max_ground_discontinuity:4.2f} ground discontinuity, {icesat2_canopy:4.2f} IC2, {classification_area} class, {classification_area_name}, {classification_purity_percentage} class percentage")
                        line=cross_comparison_input_file.readline()
                        
                # Filter for classes
                    rmse_class=np.zeros([7])
                    pairs_class=np.zeros([7])
                    for filter_class in range(7) : # class 6 is to do all classes but water, 2-5
                        
                        lidar_canopy_class_filtered_list=np.array([])
                        icesat2_canopy_class_filtered_list=np.array([])
                        for scan_element in range(len(lidar_canopy_list)) :
                            if max_ground_discontinuity_list[scan_element]<max_ground_discontinuity_threshold and classification_purity_percentage_list[scan_element]>=classification_purity_threshold and ( classification_area_list[scan_element]==filter_class or ( classification_area_list[scan_element]>1 and filter_class==6 ) ) :
                                lidar_canopy_class_filtered_list=np.append(lidar_canopy_class_filtered_list,lidar_canopy_list[scan_element])
                                icesat2_canopy_class_filtered_list=np.append(icesat2_canopy_class_filtered_list,icesat2_canopy_list[scan_element])
                                #print(f"Using {max_ground_discontinuity_list[scan_element]} {classification_area_list[scan_element]} {lidar_canopy_list[scan_element]:4.2f} {icesat2_canopy_list[scan_element]:4.2f}")
                            #else :
                            #    print(f"Not   {max_ground_discontinuity_list[scan_element]} {classification_area_list[scan_element]} {lidar_canopy_list[scan_element]:4.2f} {icesat2_canopy_list[scan_element]:4.2f}")
                                
                    
                    
                        # Do analyses
                        if len(lidar_canopy_class_filtered_list)>1 :
                            flipped_icesat2_canopy_class_filtered_list=icesat2_canopy_class_filtered_list.reshape((-1, 1))
                            inverse_model=LinearRegression().fit(flipped_icesat2_canopy_class_filtered_list, lidar_canopy_class_filtered_list)
                            lidar_prediction_from_icesat2_and_regression=inverse_model.predict(flipped_icesat2_canopy_class_filtered_list)
                            error_per_prediction=lidar_prediction_from_icesat2_and_regression-lidar_canopy_class_filtered_list
                            absolute_error_per_prediction = list(map(abs, error_per_prediction))
                            mean_absolute_error=statistics.mean(absolute_error_per_prediction)
                            std_dev_prediction=np.std(absolute_error_per_prediction)
                            rmse_prediction=rmse(lidar_prediction_from_icesat2_and_regression,lidar_canopy_class_filtered_list)
                            percentile95_error=np.percentile(absolute_error_per_prediction,95)
                            percentile99_error=np.percentile(absolute_error_per_prediction,99)
                            max_error=max(absolute_error_per_prediction)

                            rmse_class[filter_class]=rmse_prediction
                            pairs_class[filter_class]=len(lidar_canopy_class_filtered_list)

                            intercept=inverse_model.intercept_
                            slope=inverse_model.coef_[0]
                            r_squared = inverse_model.score(flipped_icesat2_canopy_class_filtered_list, lidar_canopy_class_filtered_list)
                            output_file.write(f"{filter_class},{intercept},{slope},{len(lidar_canopy_class_filtered_list)},{rmse_prediction},{percentile99_error},{max_error}\n")
                            summary_file.write(f"{rmse_prediction},{len(lidar_canopy_class_filtered_list)},{r_squared},")
                            print (f"Class {filter_class}, {classification(filter_class)}, Pairs {len(flipped_icesat2_canopy_class_filtered_list)} Intercept {intercept:4.2f}, Slope {slope:4.2f},  RMSE {rmse_prediction:4.2f}, r-sq {r_squared:4.2f}")
                            if not 'skip' in verbose :
                                plt.scatter(icesat2_canopy_class_filtered_list,lidar_canopy_class_filtered_list,s=2,label='ICESat-2 vs LiDAR',c=[[1,0,0]])
                                plt.plot(icesat2_canopy_class_filtered_list,lidar_prediction_from_icesat2_and_regression,'b',linewidth=0.5,label='Fit')
                                if filter_class<6 :
                                    class_name_display=classification(filter_class)
                                else :
                                    class_name_display="All"
                                plt.title(f"{cross_comparison_input_file_name}, class {filter_class} {class_name_display} \nr-squared {r_squared:4.2f} , intercept {inverse_model.intercept_:4.1f}m, Slope {inverse_model.coef_[0]:4.2f}, pairs {len(icesat2_canopy_class_filtered_list)}\nPred RMSE {rmse_prediction:4.2f}m 99th percentile error {percentile99_error:4.2f}m Max error {max_error:4.2f}m\n")
                                plt.ylabel('LiDAR-derived canopy height (m)')
                                plt.xlabel('ICESat-2 derived canopy height (m)')
                                if verbose == 'plot' :
                                    plt.show()
                                else :
                                    plt.savefig(output_directory+cross_comparison_input_file_name[:-4]+"_class_"+class_name_display+"_fit.png", format="png",dpi=1200)
                                plt.close()
                        else :
                            summary_file.write(f"999,0,999,")
                    # Output to textfile
                    mean_non_water_rmse=(rmse_class[2]*pairs_class[2]+rmse_class[3]*pairs_class[3]+rmse_class[4]*pairs_class[4]+rmse_class[5]*pairs_class[5])/(pairs_class[2]+pairs_class[3]+pairs_class[4]+pairs_class[5])
                    summary_file.write(f"{mean_non_water_rmse}\n")
                    #summary_file.write(f"\n")
                    output_file.close()
    summary_file.close()        
