from netCDF4 import Dataset
import os
import numpy as np
import sys
import operator
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import statistics
import math

location='Alienware'
region_runs=['basin'] # 'basin','test','uav'
version='v15'
buffer=20
upper_limit_percentile=75
upper_limit_percentile_big=99
verbose='save' # plot for interactive
do_plots=0
atl08_version='4' # v4 or v4 analysis
exclude_problem_tracks='no'

if location=='Alienware' :
    input_directory="D:\\Dropbox\\Work - Edinburgh\\IceSat2\\ATL08 redo\\"
elif location=='xps8930' :
    input_directory="D:\\Hard Disk User Folder\\Dropbox\\Work - Edinburgh\\IceSat2\\ATL08 redo\\"
elif location=='work' :
    input_directory="D:\\Dropbox\\Work - Edinburgh\\IceSat2\\ATL08 redo\\"
else :
    print("Don't know location")
    sys.exit()

exclusion_filelist_filename=input_directory+"ExclusionList.txt"
exclusion_filelist_file=open(exclusion_filelist_filename)
excluded_files=[]
if exclude_problem_tracks=='yes' :
    print("Reading in track exclusion list")
    line=exclusion_filelist_file.readline()
    while (line !='') :
        excluded_files.append(line.split("\n")[0])
        line=exclusion_filelist_file.readline()
else :
    print("Not excluding tracks")
    #sys.exit()

for top_trim in [1] : # Distance between 99th canopy percentile and point removal threshold
    for region in region_runs :
        if region=='basin' :
            west_edge=15.7
            east_edge=21.4
            south_edge=-2.8
            north_edge=2.8
        elif region=='test' :
            west_edge=18.70
            east_edge=18.81
            south_edge=0.90
            north_edge=1.30
        elif region=='uav' :
            west_edge=17.50
            east_edge=17.80
            south_edge=1.24
            north_edge=1.27
            
        else :
            print("Which site to process?")
            sys.exit()
        
        output_directory=input_directory+"extracted"+version+"_buffer"+str(buffer)+"\\"+region+"_"+location+"\\percentile"+str(upper_limit_percentile_big)+"_trim_"+str(top_trim)+"m\\ATL08v"+str(atl08_version)+"exclusion="+exclude_problem_tracks+"\\"
        allpoints_filename=output_directory+"allpoints.txt"
        allpoints_QC_filename=output_directory+"allpoints_QC.txt"
        
        if not os.path.isdir(output_directory) :
            os.makedirs(output_directory)
        allpoints_file=open(allpoints_filename,"w")
        allpoints_file.write('Long, Lat, DEM, h_te_median, h_te_mean, h_te_bestfit, h_te_interp, number_of_photons(n_seg_ph), multiple_scattering_flag(msw_flag), cloud_flag_atm,  h_canopy, h_canopy_mean, h_canopy_median, canopy_flag, src_file, track, beam\n')
        allpoints_QC_file=open(allpoints_QC_filename,"w")
        allpoints_QC_file.write('Long, Lat, DEM, h_te_median, h_te_mean, h_te_bestfit, h_te_interp, number_of_photons(n_seg_ph), multiple_scattering_flag(msw_flag), cloud_flag_atm,  h_canopy, h_canopy_mean, h_canopy_median, canopy_flag, src_file, track, beam\n')

        beams=["gt1l","gt1r","gt2l","gt2r","gt3l","gt3r"]
        beam_strength_list=["strong","weak"]
        canopy_worry_threshold=200
        os.chdir(input_directory)
        
        for track in range(0,1388) :  # diagnostic, was (0,1388)
            #print(f"Track {track}, beam ", end='')
            track_string="_{:04d}".format(track)

            for beam in beams :
                for beam_strength_scan in beam_strength_list:                    
                    output_filename=output_directory+"ATL08_aggregated_"+version+"_"+track_string+"_"+beam+"_"+beam_strength_scan+".txt"
                    #sys.exit()
                    output_file=open(output_filename,"w")
                    output_file.write('Long, Lat, DEM, h_te_median, h_te_mean, h_te_bestfit, h_te_interp, number_of_photons(n_seg_ph), multiple_scattering_flag(msw_flag), cloud_flag_atm,  h_canopy, h_canopy_mean, h_canopy_median, canopy_flag, src_file, track, beam\n')


                    print(f"Track {track}, beam {beam}, strength {beam_strength_scan}")
                    maximum_canopy_elevation=-999          
                    entries = os.scandir(input_directory)
                    extracted_points_per_track=0
                    combined_list=[]
                    longitude_land_array_big=[]
                    latitude_land_array_big=[]
                    dem_h_array_big=[]
                    h_te_median_array_big=[]
                    h_te_mean_array_big=[]
                    h_te_best_fit_array_big=[]
                    h_te_interp_array_big=[]
                    n_seg_ph_array_big=[]
                    msw_flag_array_big=[]
                    cloud_flag_atm_array_big=[]
                    h_canopy_array_big=[]
                    h_mean_canopy_array_big=[]
                    h_median_canopy_array_big=[]
                    canopy_flag_array_big=[]
                    input_file_array_big=[]         
                    distance_m_array_big=[]
                    h_te_best_fit_step_array_big=[]
                    h_te_best_fit_slope_array_big=[]
                    h_te_best_fit_minus_DEM_array_big=[]

                    # array for removed elevations
                    h_mean_canopy_removed_big_array=[]
                    
                    for entry in entries:
                        #print(f"version <{entry.name[-4:-3]}>")
                        if entry.name.endswith('h5') and (entry.name[-7:-6]==atl08_version) and (track_string in entry.name) and not any(entry.name[:29] in filenames for filenames in excluded_files) : # diagnostic
                            input_file=entry.name
                            rootgrp=Dataset(input_file,"r")
                            print(f"Data file {input_file} ",end='')            

                            sc_orient="/orbit_info/sc_orient"
                            latitude_land="/"+beam+"/land_segments/latitude"
                            longitude_land="/"+beam+"/land_segments/longitude"
                            h_dif_ref="/"+beam+"/land_segments/h_dif_ref"
                            dem_h="/"+beam+"/land_segments/dem_h"
                            h_te_best_fit="/"+beam+"/land_segments/terrain/h_te_best_fit"
                            h_te_mean="/"+beam+"/land_segments/terrain/h_te_mean"
                            h_te_median="/"+beam+"/land_segments/terrain/h_te_median"
                            h_te_interp="/"+beam+"/land_segments/terrain/h_te_interp"
                            msw_flag="/"+beam+"/land_segments/msw_flag"
                            n_seg_ph="/"+beam+"/land_segments/n_seg_ph"
                            cloud_flag_atm="/"+beam+"/land_segments/cloud_flag_atm"
                            h_canopy="/"+beam+"/land_segments/canopy/h_canopy"
                            h_mean_canopy="/"+beam+"/land_segments/canopy/h_mean_canopy"
                            h_median_canopy="/"+beam+"/land_segments/canopy/h_median_canopy"
                            canopy_flag="/"+beam+"/land_segments/canopy/canopy_flag"
                            direction=rootgrp[sc_orient][0] # 1 = fwd, R=strong, 0=backward, R=weak
                            if (direction==0 and beam[3]=='r') or (direction==1 and beam[3]=='l') :
                                beam_strength='weak'
                            else :
                                beam_strength='strong'
                            print("beam "+beam[3]+", direction "+str(direction)+", beam strength "+beam_strength+",",end='')
                            
                            if beam_strength==beam_strength_scan :
                                
                                try :
                                    elements=len(rootgrp[latitude_land])
                                except :
                                    print(f"Data missing in file {input_file}")
                                else :
                                    print(f'{elements:5d} elements, ', end='')
                                    
                                    output_step=int(elements/5)
                                    used_point_count=0;                    
                                    #print('Long, Lat, DEM+corr, DEM, corr, number_of_photons(n_seg_ph), multiple_scattering_flag(msw_flag), cloud_flag_atm')
                                    longitude_land_array=[]
                                    latitude_land_array=[]
                                    dem_h_array=[]
                                    h_te_median_array=[]
                                    h_te_mean_array=[]
                                    h_te_best_fit_array=[]
                                    h_te_interp_array=[]
                                    n_seg_ph_array=[]
                                    msw_flag_array=[]
                                    cloud_flag_atm_array=[]
                                    h_canopy_array=[]
                                    h_mean_canopy_array=[]
                                    h_median_canopy_array=[]
                                    canopy_flag_array=[]
                                    input_file_array=[]
                                    distance_m_array=[]
                                    h_te_best_fit_step_array=[]
                                    h_te_best_fit_slope_array=[]
                                    h_te_best_fit_minus_DEM_array=[]
                                    maximum_canopy_elevation_per_file=-999
                                    j=0
                                    for i in range(elements):
                                        if (rootgrp[latitude_land][i] > south_edge) and (rootgrp[latitude_land][i] < north_edge) \
                                        and (rootgrp[longitude_land][i] > west_edge) and (rootgrp[longitude_land][i] < east_edge) :
                                        #and not (rootgrp[h_dif_ref][i].mask or rootgrp[h_te_median][i].mask or rootgrp[h_te_mean][i].mask or rootgrp[h_te_best_fit][i].mask or rootgrp[h_canopy][i].mask):
                                            #print(f'DATA {rootgrp[longitude_land][i]:12.6f}, {rootgrp[latitude_land][i]:12.6f}, {(rootgrp[dem_h][i]):8.2f}, {rootgrp[h_te_median][i]:8.2f}, {rootgrp[h_te_mean][i]:8.2f}, {rootgrp[h_te_best_fit][i]:8.2f}, {rootgrp[n_seg_ph][i]:4d}, {rootgrp[msw_flag][i]:2d}, {rootgrp[cloud_flag_atm][i]:2d}')                    
                                            output_file.write(f'{rootgrp[longitude_land][i]:12.6f}, {rootgrp[latitude_land][i]:12.6f}, {(rootgrp[dem_h][i]):8.2f}, {rootgrp[h_te_median][i] if not rootgrp[h_te_median][i].mask else float("NaN") :8.2f}, {rootgrp[h_te_mean][i] if not rootgrp[h_te_mean][i].mask else float("NaN") :8.2f}, {rootgrp[h_te_best_fit][i] if not rootgrp[h_te_best_fit][i].mask else float("NaN"):8.2f}, {rootgrp[h_te_interp][i] if not rootgrp[h_te_interp][i].mask else float("NaN"):8.2f}, {rootgrp[n_seg_ph][i]:4d}, {rootgrp[msw_flag][i]:2d}, {rootgrp[cloud_flag_atm][i]:2d}, {rootgrp[h_canopy][i] if not rootgrp[h_canopy][i].mask else float("NaN"):8.2f}, {rootgrp[h_mean_canopy][i] if not rootgrp[h_mean_canopy][i].mask else float("NaN"):8.2f}, {rootgrp[h_median_canopy][i] if not rootgrp[h_median_canopy][i].mask else float("NaN"):8.2f}, {rootgrp[canopy_flag][i]}, {input_file}, {input_file[21:25]}, {beam}\n')
                                            allpoints_file.write(f'{rootgrp[longitude_land][i]:12.6f}, {rootgrp[latitude_land][i]:12.6f}, {(rootgrp[dem_h][i]):8.2f}, {rootgrp[h_te_median][i] if not rootgrp[h_te_median][i].mask else float("NaN") :8.2f}, {rootgrp[h_te_mean][i] if not rootgrp[h_te_mean][i].mask else float("NaN") :8.2f}, {rootgrp[h_te_best_fit][i] if not rootgrp[h_te_best_fit][i].mask else float("NaN"):8.2f}, {rootgrp[h_te_interp][i] if not rootgrp[h_te_interp][i].mask else float("NaN"):8.2f}, {rootgrp[n_seg_ph][i]:4d}, {rootgrp[msw_flag][i]:2d}, {rootgrp[cloud_flag_atm][i]:2d}, {rootgrp[h_canopy][i] if not rootgrp[h_canopy][i].mask else float("NaN"):8.2f}, {rootgrp[h_mean_canopy][i] if not rootgrp[h_mean_canopy][i].mask else float("NaN"):8.2f}, {rootgrp[h_median_canopy][i] if not rootgrp[h_median_canopy][i].mask else float("NaN"):8.2f}, {rootgrp[canopy_flag][i]}, {input_file}, {input_file[21:25]}, {beam}\n')
                                            used_point_count+=1
                                            extracted_points_per_track+=1
                                            combined_list.append([rootgrp[longitude_land][i], rootgrp[latitude_land][i], rootgrp[dem_h][i], rootgrp[h_te_median][i] if not rootgrp[h_te_median][i].mask else float("NaN"), rootgrp[h_te_mean][i] if not rootgrp[h_te_mean][i].mask else float("NaN"), rootgrp[h_te_best_fit][i] if not rootgrp[h_te_best_fit][i].mask else float("NaN"), rootgrp[h_te_interp][i] if not rootgrp[h_te_interp][i].mask else float("NaN"), rootgrp[n_seg_ph][i], rootgrp[msw_flag][i], rootgrp[cloud_flag_atm][i], rootgrp[h_canopy][i] if not rootgrp[h_canopy][i].mask else float("NaN"), rootgrp[h_mean_canopy][i] if not rootgrp[h_mean_canopy][i].mask else float("NaN"), rootgrp[h_median_canopy][i] if not rootgrp[h_median_canopy][i].mask else float("NaN"), rootgrp[canopy_flag][i], input_file, input_file[21:25], beam])
                                            # Build arrays for filtering odd canopy elevations

                                            longitude_land_array.append(rootgrp[longitude_land][i])
                                            latitude_land_array.append(rootgrp[latitude_land][i])
                                            dem_h_array.append(rootgrp[dem_h][i])
                                            h_te_median_array.append(rootgrp[h_te_median][i] if not rootgrp[h_te_median][i].mask else float("NaN"))
                                            h_te_mean_array.append(rootgrp[h_te_mean][i] if not rootgrp[h_te_mean][i].mask else float("NaN"))
                                            h_te_best_fit_array.append(rootgrp[h_te_best_fit][i] if not rootgrp[h_te_best_fit][i].mask else float("NaN"))
                                            h_te_best_fit_minus_DEM_array.append(h_te_best_fit_array[j]-dem_h_array[j])
                                            h_te_interp_array.append(rootgrp[h_te_interp][i])
                                            n_seg_ph_array.append(rootgrp[n_seg_ph][i])
                                            msw_flag_array.append(rootgrp[msw_flag][i])
                                            cloud_flag_atm_array.append(rootgrp[cloud_flag_atm][i])
                                            h_canopy_array.append(rootgrp[h_canopy][i] if not rootgrp[h_canopy][i].mask else float("NaN"))
                                            h_mean_canopy_array.append(rootgrp[h_mean_canopy][i] if not rootgrp[h_mean_canopy][i].mask else float("NaN"))
                                            h_median_canopy_array.append(rootgrp[h_median_canopy][i] if not rootgrp[h_median_canopy][i].mask else float("NaN"))
                                            canopy_flag_array.append(rootgrp[canopy_flag][i])
                                            input_file_array.append(input_file)
                                            distance_m_array.append(haversine([longitude_land_array[0],latitude_land_array[0]],[longitude_land_array[j],latitude_land_array[j]],unit='m'))                
                                            if j==0 :
                                                h_te_best_fit_step_array.append(0)
                                                h_te_best_fit_slope_array.append(0)
                                            else :
                                                h_te_best_fit_step_array.append(h_te_best_fit_array[j]-h_te_best_fit_array[j-1])
                                                h_te_best_fit_slope_array.append((h_te_best_fit_array[j]-h_te_best_fit_array[j-1])/distance_m_array[j])

                                            j+=1
                                            #if int(i/output_step) == float(i)/output_step:
                                            #print(f'INFO {rootgrp[longitude_land][i]:12.6f}, {rootgrp[latitude_land][i]:12.6f}, {(rootgrp[dem_h][i]):8.2f}, {rootgrp[h_te_median][i] if not rootgrp[h_te_median][i].mask else float("NaN") :8.2f}, {rootgrp[h_te_mean][i] if not rootgrp[h_te_mean][i].mask else float("NaN") :8.2f}, {rootgrp[h_te_best_fit][i] if not rootgrp[h_te_best_fit][i].mask else float("NaN"):8.2f}, {rootgrp[h_te_interp][i] if not rootgrp[h_te_interp][i].mask else float("NaN"):8.2f}, {rootgrp[n_seg_ph][i]:4d}, {rootgrp[msw_flag][i]:2d}, {rootgrp[cloud_flag_atm][i]:2d}, {rootgrp[h_canopy][i] if not rootgrp[h_canopy][i].mask else float("NaN"):8.2f}, {rootgrp[h_mean_canopy][i] if not rootgrp[h_mean_canopy][i].mask else float("NaN"):8.2f}, {rootgrp[h_median_canopy][i] if not rootgrp[h_median_canopy][i].mask else float("NaN"):8.2f}, {rootgrp[canopy_flag][i]}, {input_file}, {input_file[21:25]}, {beam}\n')
                                            
                                            if rootgrp[h_mean_canopy][i] > maximum_canopy_elevation :
                                                maximum_canopy_elevation = rootgrp[h_mean_canopy][i]
                                            if rootgrp[h_mean_canopy][i] > maximum_canopy_elevation_per_file :
                                                maximum_canopy_elevation_per_file = rootgrp[h_mean_canopy][i]

                                    print(f'{used_point_count} points used, max canopy {maximum_canopy_elevation_per_file:4.0f}')
                                    #plt.plot(distance_m_array,h_mean_canopy_array)

                                    #output plot for initial study, not used now
                                    if maximum_canopy_elevation_per_file > canopy_worry_threshold :
                                        plot_type='simple'
                                        if plot_type=='simple' :
                                            plt.scatter(distance_m_array,h_mean_canopy_array,s=2,label='h_mean_canopy',c=[[1,0,0]])
                                            plt.plot(distance_m_array,h_te_best_fit_step_array,'b',linewidth=0.5,label='h_te_best_fit_step')
                                            #plt.plot(distance_m_array,h_te_best_fit_minus_DEM_array,'g',label='Elev diff')
                                            
                                            plt.legend()
                                        else:
                                            fig, ax1 = plt.subplots()
                                            red_color = 'tab:red'
                                            blue_color = 'tab:blue'
                                            green_color = 'tab:green'
                                            ax1.set_xlabel('Distance (m)')
                                            ax1.set_ylabel('Elevation (m)', color=red_color)
                                            ax1.scatter(distance_m_array,h_mean_canopy_array,s=2,label='h_mean_canopy',c=[[1,0,0]])
                                            ax1.plot(distance_m_array,h_te_best_fit_step_array, color=green_color)
                                            ax1.tick_params(axis='y', labelcolor=red_color)

                                            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                                            ax2.set_ylabel('slope', color=blue_color)  # we already handled the x-label with ax1
                                            ax2.plot(distance_m_array, h_te_best_fit_slope_array, color=blue_color)
                                            ax2.tick_params(axis='y', labelcolor=blue_color)

                                            fig.tight_layout()  # otherwise the right y-label is slightly clipped
                                        mean_step=statistics.mean([element for element in h_te_best_fit_step_array if not math.isnan(element)])
                                        mean_step_mag=statistics.mean([abs(element) for element in h_te_best_fit_step_array if not math.isnan(element)])
                                        max_step_mag=max([abs(element) for element in h_te_best_fit_step_array if not math.isnan(element)])
                                        plt.title(f"{input_file}\n {mean_step:4.1f}m mean, {mean_step_mag:4.1f}m mean mag, {max_step_mag:4.1f}m max step\nCanopy max {maximum_canopy_elevation_per_file:4.1f}m")
                                        if verbose == 'plot' :
                                            plt.show()
                                        else :
                                            plt.savefig(output_directory+input_file+"_"+beam+"_analysis.png", format="png",dpi=1200)
                                            print("Saving plot "+input_file+"_"+beam+"_analysis.png")
                                        plt.close()
                                    # Output preQC file
                                    #output_preQC_filename=output_directory+input_file+"_"+beam+"_preQC.txt"
                                    #output_preQC_file=open(output_preQC_filename,"w")
                                    #output_preQC_file.write('Long, Lat, DEM, h_te_median, h_te_mean, h_te_bestfit, h_te_interp, number_of_photons(n_seg_ph), multiple_scattering_flag(msw_flag), cloud_flag_atm,  h_canopy, h_canopy_mean, h_canopy_median, canopy_flag, src_file, track, beam\n')
                                    #for i in range(len(distance_m_array)) :
                                    #    output_preQC_file.write(f'{longitude_land_array[i]:12.6f}, {latitude_land_array[i]:12.6f}, {dem_h_array[i]:8.2f}, {h_te_median_array[i]:8.2f}, {h_te_mean_array[i]:8.2f}, {h_te_best_fit_array[i]:8.2f}, {h_te_interp_array[i]:8.2f}, {n_seg_ph_array[i]:4d}, {msw_flag_array[i]:2d}, {cloud_flag_atm_array[i]:2d}, {h_canopy_array[i]:8.2f}, {h_mean_canopy_array[i]:8.2f}, {h_median_canopy_array[i]:8.2f}, {canopy_flag_array[i]}, {input_file}, {input_file[21:25]}, {beam}\n')
                                    #output_preQC_file.close()
                                    #print("Pre QC file saved")
                                    #sys.exit()

                                    # Do QC on array

                                    # Build a warning array based on elevation variation
                                    
                                    # count actual non-nan h_canopy_mean points
                                    non_nan_used = len([actual for actual in h_mean_canopy_array if not math.isnan(actual)])
                                    if non_nan_used > buffer*2 :
                                        cumul_change=[]
                                        median_local_height=[]
                                        upper_canopy_height_measure=[]
                                        max_local_change=[]
                                        h_te_best_fit_step_abs_array=[]
                                        for abloop in range(len(h_te_best_fit_step_array)) :
                                            h_te_best_fit_step_abs_array.append(abs(h_te_best_fit_step_array[abloop]))
                                        if len(distance_m_array) > buffer :
                                            for exploreloop in range(len(h_te_best_fit_step_array)) :
                                                if exploreloop<=int(buffer/2) :
                                                    cumul_change.append(sum(h_te_best_fit_step_array[0:buffer]))
                                                    max_local_change.append(max(h_te_best_fit_step_abs_array[0:buffer]))
                                                    median_local_height.append(statistics.median(h_te_best_fit_step_abs_array[0:buffer]))
                                                    non_nan_canopy=[actual for actual in h_mean_canopy_array[0:buffer] if not math.isnan(actual)]
                                                    if len(non_nan_canopy)>0 :
                                                        upper_canopy_height_measure.append(np.percentile(non_nan_canopy,upper_limit_percentile))
                                                    else :
                                                        upper_canopy_height_measure.append(0.0)
                                                elif exploreloop>=len(distance_m_array)-int(buffer)/2 :
                                                    cumul_change.append(sum(h_te_best_fit_step_array[-buffer:]))
                                                    max_local_change.append(max(h_te_best_fit_step_abs_array[-buffer:]))
                                                    median_local_height.append(statistics.median(h_te_best_fit_step_abs_array[-buffer:]))
                                                    non_nan_canopy=[actual for actual in h_mean_canopy_array[-buffer:] if not math.isnan(actual)]
                                                    if len(non_nan_canopy)>0 :
                                                        upper_canopy_height_measure.append(np.percentile(non_nan_canopy,upper_limit_percentile))
                                                    else :
                                                        upper_canopy_height_measure.append(0.0)
                                                else :
                                                    cumul_change.append(sum(h_te_best_fit_step_array[exploreloop-int(buffer/2):exploreloop+int(buffer/2)]))
                                                    max_local_change.append(max(h_te_best_fit_step_abs_array[exploreloop-int(buffer/2):exploreloop+int(buffer/2)]))
                                                    median_local_height.append(statistics.median(h_te_best_fit_step_abs_array[exploreloop-int(buffer/2):exploreloop+int(buffer/2)]))
                                                    non_nan_canopy=[actual for actual in h_mean_canopy_array[exploreloop-int(buffer/2):exploreloop+int(buffer/2)] if not math.isnan(actual)]
                                                    if len(non_nan_canopy)>0 :
                                                        upper_canopy_height_measure.append(np.percentile(non_nan_canopy,upper_limit_percentile))
                                                    else :
                                                        upper_canopy_height_measure.append(0.0)
                                            #sys.exit()
                                            
                                        distance_m_array_original=distance_m_array.copy()
                                        h_te_best_fit_step_array_original=h_te_best_fit_step_array.copy()
                                        max_local_change_original=max_local_change.copy()
                                        cumul_change_original=cumul_change.copy()
                                        median_local_height_original=median_local_height.copy()
                                        upper_canopy_height_measure_original=upper_canopy_height_measure.copy()
                                        canopy_removal_threshold=50
                                        removed_distance_array=[]
                                        removed_h_mean_canopy_array=[]
                                        loop_end=len(longitude_land_array)
                                        i=0
                                        while i < loop_end :
                                            #print(f"testing {i}")
                                            #if (h_mean_canopy_array[i]>canopy_removal_threshold) or (h_mean_canopy_array[i]>canopy_removal_threshold/2 and abs(h_te_best_fit_step_array[i])>canopy_removal_threshold/2) :
                                            if ( (h_mean_canopy_array[i]>canopy_removal_threshold) or
                                                (h_mean_canopy_array[i]>canopy_removal_threshold/2 and abs(max_local_change[i])>h_mean_canopy_array[i]/2) or
                                                (h_mean_canopy_array[i]>canopy_removal_threshold/2 and (i<buffer or i>loop_end-buffer)) ) :
                                                #print(f"Removing {distance_m_array[i]:4.0f},{h_mean_canopy_array[i]:4.0f}")
                                                removed_distance_array.append(distance_m_array[i])
                                                removed_h_mean_canopy_array.append(h_mean_canopy_array[i])
                                                h_mean_canopy_removed_big_array.append(h_mean_canopy_array[i])
                                                longitude_land_array.pop(i)
                                                latitude_land_array.pop(i)
                                                dem_h_array.pop(i)
                                                h_te_median_array.pop(i)
                                                h_te_mean_array.pop(i)
                                                h_te_best_fit_array.pop(i)
                                                h_te_best_fit_minus_DEM_array.pop(i)
                                                h_te_best_fit_step_array.pop(i)
                                                h_te_interp_array.pop(i)
                                                n_seg_ph_array.pop(i)
                                                msw_flag_array.pop(i)
                                                cloud_flag_atm_array.pop(i)
                                                h_canopy_array.pop(i)
                                                h_mean_canopy_array.pop(i)
                                                h_median_canopy_array.pop(i)
                                                canopy_flag_array.pop(i)
                                                input_file_array.pop(i)
                                                distance_m_array.pop(i)

                                                max_local_change.pop(i)
                                                cumul_change.pop(i)
                                                loop_end-=1
                                            else :
                                                #print(f"Leaving {distance_m_array[i]:4.0f},{h_mean_canopy_array[i]:4.0f}")
                                                i+=1
                                        if do_plots :
                                            # plot kept and deleted points
                                            plt.scatter(distance_m_array,h_mean_canopy_array,s=2,label='h_mean_canopy',c=[[0,1,0]])
                                            plt.scatter(removed_distance_array,removed_h_mean_canopy_array,s=2,label='canopy points removed',c=[[1,0,0]])
                                            plt.plot(distance_m_array_original,h_te_best_fit_step_array_original,'b',linewidth=0.5,label='h_te_best_fit_step')
                                            plt.plot(distance_m_array_original,max_local_change_original,'y:',linewidth=0.5,label='max local change')
                                            plt.plot(distance_m_array_original,upper_canopy_height_measure_original,'m:',linewidth=0.5,label='upper_canopy_height_measure_original, percentile '+str(upper_limit_percentile))
                                            #plt.plot(distance_m_array_original,cumul_change_original,'c:',linewidth=0.5,label='cumul change')
                                            
                                            plt.legend()
                                            plt.title(f"{input_file} {beam}\n{non_nan_used} canopy points, {len(h_mean_canopy_array)} used,{len(removed_h_mean_canopy_array)} removed")
                                            if verbose == 'plot' :
                                                plt.show()
                                            else :
                                                plt.savefig(output_directory+input_file+"_"+beam+"_before_and_after.png", format="png",dpi=1200)
                                                print("Saving plot "+input_file+"_"+beam+"_before_and_after.png")
                                            plt.close()
                                    else :
                                            print(f"Not used, only {non_nan_used} points")
                                    # Only append if substantial number of non-nan canopy elevations
                                    # Append QC elements to main array, adding input_file_array_big from input_file 
                                    if non_nan_used > buffer*2 :
                                        for i in range(len(distance_m_array)) :
                                            if not math.isnan(h_mean_canopy_array[i]) :
                                                longitude_land_array_big.append(longitude_land_array[i])
                                                latitude_land_array_big.append(latitude_land_array[i])
                                                dem_h_array_big.append(dem_h_array[i])
                                                h_te_median_array_big.append(h_te_median_array[i])
                                                h_te_mean_array_big.append(h_te_mean_array[i])
                                                h_te_best_fit_array_big.append(h_te_best_fit_array[i])
                                                h_te_best_fit_minus_DEM_array_big.append(h_te_best_fit_minus_DEM_array[i])
                                                h_te_interp_array_big.append(h_te_interp_array[i])
                                                n_seg_ph_array_big.append(n_seg_ph_array[i])
                                                msw_flag_array_big.append(msw_flag_array[i])
                                                cloud_flag_atm_array_big.append(cloud_flag_atm_array[i])
                                                h_canopy_array_big.append(h_canopy_array[i])
                                                h_mean_canopy_array_big.append(h_mean_canopy_array[i])
                                                h_median_canopy_array_big.append(h_median_canopy_array[i])
                                                canopy_flag_array_big.append(canopy_flag_array[i])
                                                input_file_array_big.append(input_file_array[i])
                                                distance_m_array_big.append(distance_m_array[i])                
                                rootgrp.close()
                                #sys.exit()
                                # plot up
                            else:
                                print("") # Not used beam strength
                    #print("\n")
                    output_file.close()
                    #print(f'Checking {output_filename}')
                    #sys.exit()
                    if extracted_points_per_track==0 :
                        try :
                            os.remove(output_filename)
                        except :
                            print("Failed to delete output file.")
                        else :
                            print("Deleted output file.")
                    else :
                        # Plot big track canopy points
         
                        if len(distance_m_array_big) > buffer and do_plots:
                            plt.scatter(distance_m_array_big,h_mean_canopy_array_big,s=2,label='h_mean_canopy_big',c=[[0,1,0]])
                            plt.legend()
                            plt.title(f"Track {track_string} Beam {beam}")
                            if verbose == 'plot' :
                                plt.show()
                            else :
                                plt.savefig(output_directory+"Track"+track_string+"_beam_"+beam+".png", format="png",dpi=1200)
                            plt.close()
                            
                        # Output QC big array data
                        output_filename_QC=output_directory+"ATL08_aggregated_"+version+"_"+track_string+"_"+beam+"_"+beam_strength_scan+"_QC.txt"
                        output_QC_file=open(output_filename_QC,"w")
                        output_QC_file.write('Long, Lat, DEM, h_te_median, h_te_mean, h_te_bestfit, h_te_interp, number_of_photons(n_seg_ph), multiple_scattering_flag(msw_flag), cloud_flag_atm,  h_canopy, h_canopy_mean, h_canopy_median, canopy_flag, src_file, track, beam, distance\n')
                        for i in range(len(distance_m_array_big)) :
                            output_QC_file.write(f'{longitude_land_array_big[i]:12.6f}, {latitude_land_array_big[i]:12.6f}, {dem_h_array_big[i]:8.2f}, {h_te_median_array_big[i]:8.2f}, {h_te_mean_array_big[i]:8.2f}, {h_te_best_fit_array_big[i]:8.2f}, {h_te_interp_array_big[i]:8.2f}, {n_seg_ph_array_big[i]:4d}, {msw_flag_array_big[i]:2d}, {cloud_flag_atm_array_big[i]:2d}, {h_canopy_array_big[i]:8.2f}, {h_mean_canopy_array_big[i]:8.2f}, {h_median_canopy_array_big[i]:8.2f}, {canopy_flag_array_big[i]}, {input_file_array_big[i]}, {input_file_array_big[i][21:25]}, {beam}, {distance_m_array_big[i]}\n')
                        output_QC_file.close()
                        print("QC file saved")
                        #sys.exit()
                        print('Raw data saved. Sorting...')
                        combined_list_lat_sorted=sorted(combined_list,key=operator.itemgetter(1))

                        # Do a QC2 run on the entire track after latsorting

                        if maximum_canopy_elevation > canopy_worry_threshold :
                            print("Flagging")
                            sorted_output_filename=output_directory+"ATL08_aggregated_latsorted"+track_string+"_"+beam+"_"+beam_strength_scan+"_FLAG_"+str(int(maximum_canopy_elevation))+"m.txt"
                        else:
                            print("Canopy in range")
                            sorted_output_filename=output_directory+"ATL08_aggregated_latsorted"+track_string+"_"+beam+"_"+beam_strength_scan+".txt"
                        sorted_output_file=open(sorted_output_filename,"w")
                        sorted_output_file.write('Long, Lat, DEM, h_te_median, h_te_mean, h_te_bestfit, h_te_interp, number_of_photons(n_seg_ph), multiple_scattering_flag(msw_flag), cloud_flag_atm,  h_canopy, h_canopy_mean, h_canopy_median, canopy_flag, src_file, track, beam, distance(m)\n')
                        for i in range(len(combined_list_lat_sorted)) :
                            distance=haversine([combined_list_lat_sorted[0][1],combined_list_lat_sorted[0][0]],[combined_list_lat_sorted[i][1],combined_list_lat_sorted[i][0]],unit='m')
                            sorted_output_file.write(f'{combined_list_lat_sorted[i][0]:12.6f}, {combined_list_lat_sorted[i][1]:12.6f}, {combined_list_lat_sorted[i][2]:8.2f}, {combined_list_lat_sorted[i][3]:8.2f}, {combined_list_lat_sorted[i][4]:8.2f}, {combined_list_lat_sorted[i][5]:8.2f}, {combined_list_lat_sorted[i][6]:8.2f}, {combined_list_lat_sorted[i][7]:4d}, {combined_list_lat_sorted[i][8]:2d}, {combined_list_lat_sorted[i][9]:2d}, {combined_list_lat_sorted[i][10]:8.2f}, {combined_list_lat_sorted[i][11]:8.2f}, {combined_list_lat_sorted[i][12]:8.2f}, {combined_list_lat_sorted[i][13]}, {combined_list_lat_sorted[i][14]}, {combined_list_lat_sorted[i][15]}, {combined_list_lat_sorted[i][16]}, {distance:10.2f}\n')
                        sorted_output_file.close()
                        
                        # Output a QC latsorted list
                        print("LatSorting QC data")
                        combined_list_QC=[]
                        for i in range(len(longitude_land_array_big)) :
                            combined_list_QC.append([longitude_land_array_big[i], latitude_land_array_big[i], dem_h_array_big[i], h_te_median_array_big[i], h_te_mean_array_big[i], h_te_best_fit_array_big[i], h_te_interp_array_big[i], n_seg_ph_array_big[i], msw_flag_array_big[i], cloud_flag_atm_array_big[i], h_canopy_array_big[i], h_mean_canopy_array_big[i], h_median_canopy_array_big[i], canopy_flag_array_big[i], input_file_array_big[i], input_file_array_big[i][21:25], beam])
                        combined_list_lat_sorted_QC=sorted(combined_list_QC,key=operator.itemgetter(1))
                        sorted_QC_output_filename=output_directory+"ATL08_aggregated_latsorted_QC"+track_string+"_"+beam+"_"+beam_strength_scan+".txt"
                        sorted_QC_output_file=open(sorted_QC_output_filename,"w")
                        sorted_QC_output_file.write('Long, Lat, DEM, h_te_median, h_te_mean, h_te_bestfit, h_te_interp, number_of_photons(n_seg_ph), multiple_scattering_flag(msw_flag), cloud_flag_atm,  h_canopy, h_canopy_mean, h_canopy_median, canopy_flag, src_file, track, beam, distance(m)\n')
                        for i in range(len(combined_list_lat_sorted_QC)) :
                            distance=haversine([combined_list_lat_sorted_QC[0][1],combined_list_lat_sorted_QC[0][0]],[combined_list_lat_sorted_QC[i][1],combined_list_lat_sorted_QC[i][0]],unit='m')
                            sorted_QC_output_file.write(f'{combined_list_lat_sorted_QC[i][0]:12.6f}, {combined_list_lat_sorted_QC[i][1]:12.6f}, {combined_list_lat_sorted_QC[i][2]:8.2f}, {combined_list_lat_sorted_QC[i][3]:8.2f}, {combined_list_lat_sorted_QC[i][4]:8.2f}, {combined_list_lat_sorted_QC[i][5]:8.2f}, {combined_list_lat_sorted_QC[i][6]:8.2f}, {combined_list_lat_sorted_QC[i][7]:4d}, {combined_list_lat_sorted_QC[i][8]:2d}, {combined_list_lat_sorted_QC[i][9]:2d}, {combined_list_lat_sorted_QC[i][10]:8.2f}, {combined_list_lat_sorted_QC[i][11]:8.2f}, {combined_list_lat_sorted_QC[i][12]:8.2f}, {combined_list_lat_sorted_QC[i][13]}, {combined_list_lat_sorted_QC[i][14]}, {combined_list_lat_sorted_QC[i][15]}, {combined_list_lat_sorted_QC[i][16]}, {distance:10.2f}\n')
                        sorted_QC_output_file.close()

                        # final filter for last QC level
                        for i in range(len(combined_list_lat_sorted_QC)) :
                            distance=haversine([combined_list_lat_sorted_QC[0][1],combined_list_lat_sorted_QC[0][0]],[combined_list_lat_sorted_QC[i][1],combined_list_lat_sorted_QC[i][0]],unit='m')
                            combined_list_lat_sorted_QC[i].append(distance)

                        #
                        # build sorted canopy array - element 17 is distance, 11 is h_mean
                        buffer_big=200
                        combined_list_lat_sorted_QC=sorted(combined_list_lat_sorted_QC,key=operator.itemgetter(17))
                        sorted_canopy_mean=[]
                        sorted_distance=[]
                        for extract_element in range(len(combined_list_lat_sorted_QC)) :
                            sorted_canopy_mean.append(combined_list_lat_sorted_QC[extract_element][11])
                            sorted_distance.append(combined_list_lat_sorted_QC[extract_element][17])
                        big_upper_canopy_height_measure=[]
                        if len(sorted_canopy_mean) > buffer_big :
                            for exploreloop in range(len(sorted_canopy_mean)) :
                                if exploreloop<=int(buffer_big/2) :
                                    non_nan_canopy=[actual for actual in sorted_canopy_mean[0:buffer_big] if not math.isnan(actual)]
                                elif exploreloop>=len(sorted_canopy_mean)-int(buffer_big)/2 :
                                    non_nan_canopy=[actual for actual in sorted_canopy_mean[-buffer_big:] if not math.isnan(actual)]
                                else :
                                    non_nan_canopy=[actual for actual in sorted_canopy_mean[exploreloop-int(buffer_big/2):exploreloop+int(buffer_big/2)] if not math.isnan(actual)]
                                if len(non_nan_canopy)>0 :
                                    big_upper_canopy_height_measure.append(np.percentile(non_nan_canopy,upper_limit_percentile_big))
                                else :
                                    big_upper_canopy_height_measure.append(0.0)
                            if do_plots :
                                plt.scatter(sorted_distance,sorted_canopy_mean,s=2,label='h_mean_canopy_big',c=[[0,1,0]])
                                plt.plot(sorted_distance,big_upper_canopy_height_measure,'m:',linewidth=0.5,label='upper_canopy_height_measure, percentile '+str(upper_limit_percentile_big))
                                plt.legend()
                                plt.title(f"Track {track_string} Beam {beam}")
                                if verbose == 'plot' :
                                    plt.show()
                                else :
                                    plt.savefig(output_directory+"TrackV2"+track_string+"_beam_"+beam+"_percentile_"+str(upper_limit_percentile_big)+".png", format="png",dpi=1200)
                                plt.close()

                            #
                            # Filtering
                            sorted_distance_rejects=[]
                            sorted_canopy_mean_rejects=[]
                            loop_end=len(sorted_distance)
                            i=0
                            while i < loop_end :
                                if sorted_canopy_mean[i] > big_upper_canopy_height_measure[i] + top_trim :
                                    sorted_distance_rejects.append(sorted_distance[i])
                                    sorted_canopy_mean_rejects.append(sorted_canopy_mean[i])
                                    sorted_distance.pop(i)
                                    sorted_canopy_mean.pop(i)
                                    big_upper_canopy_height_measure.pop(i)
                                    combined_list_lat_sorted_QC.pop(i)
                                    loop_end-=1
                                else :
                                    i+=1
                            #
                            if do_plots :
                                plt.scatter(sorted_distance,sorted_canopy_mean,s=2,label='h_mean_canopy_big',c=[[0,1,0]])
                                plt.scatter(sorted_distance_rejects,sorted_canopy_mean_rejects,s=2,label='Canopy rejects',c=[[1,0,0]])
                                plt.plot(sorted_distance,big_upper_canopy_height_measure,'m:',linewidth=0.5,label='upper_canopy_height_measure, percentile '+str(upper_limit_percentile_big))
                                plt.legend()
                                plt.title(f"Track {track_string} Beam {beam}, trim {top_trim}m above percentile {upper_limit_percentile_big}, \nremoved {len(sorted_distance_rejects)} points of {len(sorted_distance)}")
                                if verbose == 'plot' :
                                    plt.show()
                                else :
                                    plt.savefig(output_directory+"TrackV3"+track_string+"_beam_"+beam+"_percentile_"+str(upper_limit_percentile_big)+"_trim_"+str(top_trim)+"m.png", format="png",dpi=1200)
                                plt.close()

                            #
                            # Output final QC data
                            sorted_QC2_output_filename=output_directory+"ATL08_aggregated_latsorted_QC2"+track_string+"_"+beam+"_"+beam_strength_scan+"_trim_"+str(top_trim)+"m.txt"
                            sorted_QC2_output_file=open(sorted_QC2_output_filename,"w")
                            sorted_QC2_output_file.write('Long, Lat, DEM, h_te_median, h_te_mean, h_te_bestfit, h_te_interp, number_of_photons(n_seg_ph), multiple_scattering_flag(msw_flag), cloud_flag_atm,  h_canopy, h_canopy_mean, h_canopy_median, canopy_flag, src_file, track, beam, distance(m)\n')
                            for i in range(len(combined_list_lat_sorted_QC)) :                            
                                sorted_QC2_output_file.write(f'{combined_list_lat_sorted_QC[i][0]:12.6f}, {combined_list_lat_sorted_QC[i][1]:12.6f}, {combined_list_lat_sorted_QC[i][2]:8.2f}, {combined_list_lat_sorted_QC[i][3]:8.2f}, {combined_list_lat_sorted_QC[i][4]:8.2f}, {combined_list_lat_sorted_QC[i][5]:8.2f}, {combined_list_lat_sorted_QC[i][6]:8.2f}, {combined_list_lat_sorted_QC[i][7]:4d}, {combined_list_lat_sorted_QC[i][8]:2d}, {combined_list_lat_sorted_QC[i][9]:2d}, {combined_list_lat_sorted_QC[i][10]:8.2f}, {combined_list_lat_sorted_QC[i][11]:8.2f}, {combined_list_lat_sorted_QC[i][12]:8.2f}, {combined_list_lat_sorted_QC[i][13]}, {combined_list_lat_sorted_QC[i][14]}, {combined_list_lat_sorted_QC[i][15]}, {combined_list_lat_sorted_QC[i][16]}, {combined_list_lat_sorted_QC[i][17]:10.2f}\n')
                                allpoints_QC_file.write(f'{combined_list_lat_sorted_QC[i][0]:12.6f}, {combined_list_lat_sorted_QC[i][1]:12.6f}, {combined_list_lat_sorted_QC[i][2]:8.2f}, {combined_list_lat_sorted_QC[i][3]:8.2f}, {combined_list_lat_sorted_QC[i][4]:8.2f}, {combined_list_lat_sorted_QC[i][5]:8.2f}, {combined_list_lat_sorted_QC[i][6]:8.2f}, {combined_list_lat_sorted_QC[i][7]:4d}, {combined_list_lat_sorted_QC[i][8]:2d}, {combined_list_lat_sorted_QC[i][9]:2d}, {combined_list_lat_sorted_QC[i][10]:8.2f}, {combined_list_lat_sorted_QC[i][11]:8.2f}, {combined_list_lat_sorted_QC[i][12]:8.2f}, {combined_list_lat_sorted_QC[i][13]}, {combined_list_lat_sorted_QC[i][14]}, {combined_list_lat_sorted_QC[i][15]}, {combined_list_lat_sorted_QC[i][16]}, {combined_list_lat_sorted_QC[i][17]:10.2f}\n')
                                
                            sorted_QC2_output_file.close()

                            
                        
                        
        allpoints_file.close()
        allpoints_QC_file.close()
        
        if len(h_mean_canopy_array_big)>0 :
            print(f"Overall canopy stats\n{len(h_mean_canopy_array_big)} used, {len(h_mean_canopy_removed_big_array)} removed")
            print(f"Used stats\nMin {min(h_mean_canopy_array_big):4.1f} max {max(h_mean_canopy_array_big):4.1f}")
            print(f"Removed stats Min {min(h_mean_canopy_removed_big_array):4.1f}\nMax {max(h_mean_canopy_removed_big_array):4.1f}")
            print(f"Means {statistics.mean(ma.masked_values(h_mean_canopy_array_big,-9999)):4.2f} kept, {statistics.mean(ma.masked_values(h_mean_canopy_removed_big_array,-9999)):4.2f} removed")
            summary_filename=output_directory+"Summary.txt"
            summary_file=open(summary_filename,"w")
            summary_file.write(f"Overall canopy stats\n{len(h_mean_canopy_array_big)} used, {len(h_mean_canopy_removed_big_array)} removed")
            summary_file.write(f"Used stats\nmin {min(h_mean_canopy_array_big):4.1f} max {max(h_mean_canopy_array_big):4.1f}")
            summary_file.write(f"Removed statsmin {min(h_mean_canopy_removed_big_array):4.1f}\nMax {max(h_mean_canopy_removed_big_array):4.1f}")
            summary_file.write(f"Means {statistics.mean(ma.masked_values(h_mean_canopy_array_big,-9999)):4.2f} kept, {statistics.mean(ma.masked_values(h_mean_canopy_removed_big_array,-9999)):4.2f} removed")
            summary_file.close()
