# Analyse DRC data

from netCDF4 import Dataset
import os
import sys
from haversine import haversine, Unit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import statistics
import math
import datetime
import gdal
from osgeo import gdal, osr, osr
from gdalconst import *

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

nodata=10000
ground_difference_threshold=200 # Set variation from mean/median ground in meters to indicate a problem.
version='V20_'+str(ground_difference_threshold)+"GD_threshold"

pc='Alienware'
verbose='skip' # skip or plot or save
if pc=='home' :
    input_folder="D:\\Hard Disk User Folder\\Dropbox\\Work - Edinburgh\\DRC LiDAR\\Analysis\\Data\\"
elif pc=='Alienware' :
    input_folder="D:\\Dropbox\\Work - Edinburgh\\DRC LiDAR\\Analysis\\Data\\"    
else :
    input_folder="D:\\Dropbox\\Work - Edinburgh\\DRC LiDAR\\Analysis\\Data\\"

height_type = 'percentile95'
estimate_canopy='yes' # 'no' or 'yes'

os.chdir(input_folder)
entries = os.scandir(input_folder)
print(f"{datetime.datetime.now().time()}")
for entry in entries:
    if entry.name.endswith('LL.txt') and ('to' in entry.name ) :
        input_filename=entry.name
        if "Plot136" in input_filename :
            elev_lower_threshold=334
        else :
            elev_lower_threshold=200

        if "190217_143637" in input_filename :
            elev_upper_threshold=500
        else :
            elev_upper_threshold=1000


        lon_array=np.array([],float)
        lat_array=np.array([],float)
        elev_array=np.array([],float)
        
        input_file=open(input_filename,"r")
        if "to" in input_filename :
            output_summary_folder=input_folder+"Results"+version+"\\Ferry\\"+input_filename[:-4]+"\\"
        else :
            output_summary_folder=input_folder+"Results"+version+"\\Site\\"+input_filename[:-4]+"\\"
        if not os.path.isdir(output_summary_folder) :
            os.makedirs(output_summary_folder)
        output_summary_filename=output_summary_folder+input_filename[:-4]+"_summary.txt"
        output_summary_file=open(output_summary_filename,"w")
        output_summary_file.write("File Skip Cellsize Ground_range\n")
        print(f"Reading from file {input_filename}")
        
        line=input_file.readline()
        count=0
        westmost=-999
        eastmost=-999
        southmost=-999
        northmost=-999
        min_elev=-999
        max_elev=-999
        while (line !='') :
            if line =='' :
                print("Odd")
                sys.exit()
            lon=float(line.split(" ")[0])
            lat=float(line.split(" ")[1])
            elev=float(line.split(" ")[2])
            #if elev<elev_lower_threshold :
            #    print(f"{count} {lat} {lon} {elev}")
            if lon > -30 and lat > -30 and lon < 30 and lat < 30 and elev > elev_lower_threshold and elev < elev_upper_threshold :
                if (lon<westmost) or  (westmost<-900)  :
                    westmost=lon
                if (lon>eastmost) or  (eastmost<-900) :
                    eastmost=lon
                if (lat<southmost) or (southmost<-900) :
                    southmost=lat
                if (lat>northmost) or (northmost<-900) :
                    northmost=lat
                if (elev<min_elev) or (min_elev<-900) :
                    min_elev=elev
                if (elev>max_elev) or (max_elev<-900) :
                    max_elev=elev
                count+=1
                if (count/1e5).is_integer() :
                    print(f"Read in {count/1e6:4.1f} million points, {datetime.datetime.now().time()}")
                lon_array=np.append(lon_array,lon)
                lat_array=np.append(lat_array,lat)
                elev_array=np.append(elev_array,elev)
                #print(f"{count} {lat} {lon} {elev}")
            line=input_file.readline()
            
        print(f"{count} points read into array.")
        number_of_lidar_points_in_array=count
        print(f"{datetime.datetime.now().time()}")
        ew_extent_m=haversine([southmost,westmost],[southmost,eastmost],unit='m')
        ns_extent_m=haversine([southmost,westmost],[northmost,westmost],unit='m')

        for x_cell_width_m in [500] :
            y_cell_width_m=x_cell_width_m
            print(f"Cell width {x_cell_width_m}m")
            x_cell_lon_width=x_cell_width_m*(eastmost-westmost)/(ew_extent_m)
            y_cell_lat_width=y_cell_width_m*(northmost-southmost)/(ns_extent_m)
            number_of_x_cells=int(ew_extent_m/x_cell_width_m+2)
            number_of_y_cells=int(ns_extent_m/y_cell_width_m+2)

            ground_elev_estimate_array=np.empty((number_of_x_cells,number_of_y_cells,10),float)
            for i_fill in range(number_of_x_cells) :
                for j_fill in range(number_of_y_cells) :
                    for k_fill in range(10) :
                        lidar_3d[current_cell_x][current_cell_y].sort()
            
            print(f"Second stage - Estimating ground elevation on grid from file {input_filename}")
            for array_point in range(number_of_lidar_points_in_array) :
                lon=lon_array[array_point]
                lat=lat_array[array_point]
                elev=elev_array[array_points]
                current_cell_x=int((lon-westmost)/x_cell_lon_width)
                current_cell_y=int((lat-southmost)/y_cell_lat_width)
                
                if (current_cell_x>=0) and (current_cell_x<number_of_x_cells) and (current_cell_y>=0) and (current_cell_y<number_of_y_cells) and elev > elev_lower_threshold and elev < elev_upper_threshold :
                    if elev<ground_elev_estimate_array[current_cell_x][current_cell_y][9] :
                        ground_elev_estimate_array[current_cell_x][current_cell_y][9]=elev
                        #sys.exit()
                        ground_elev_estimate_array[current_cell_x][current_cell_y].sort()
                                       
                        #print(f"Updating cell {current_cell_x}, {current_cell_y} to {ground_elev_estimate_array[current_cell_x][current_cell_y]}")

            for skip_level in [2] :
                #calc stats
                print(f"Using point {skip_level} from lowest")
                minimum_ground=-999
                maximum_ground=-999
                for y in range(number_of_y_cells) :
                    for x in range(number_of_x_cells) :
                        if (minimum_ground<-900 or minimum_ground > ground_elev_estimate_array[x][y][skip_level]) and ground_elev_estimate_array[x][y][skip_level] <9999 :
                            minimum_ground=ground_elev_estimate_array[x][y][skip_level]
                        if (maximum_ground<-900 or maximum_ground < ground_elev_estimate_array[x][y][skip_level]) and ground_elev_estimate_array[x][y][skip_level] <9999 :
                            maximum_ground=ground_elev_estimate_array[x][y][skip_level]
                #plt.(ground_elev_estimate_array,vmin=290)
                #plt.show()
                #plt.close()
                output_summary_file.write(f"{input_filename} {skip_level} {x_cell_width_m} {maximum_ground-minimum_ground:4.2f}\n")
                print(f"{input_filename} {skip_level} {x_cell_width_m} {maximum_ground-minimum_ground:4.2f}")

                # Filter skipped_ground_elev_estimate_array to remove outliers
                
                ground_sum=0
                ground_count=0
                removed_cells=0
                data_list=[]
                for y in range(number_of_y_cells) :
                    for x in range(number_of_x_cells) :
                        if ground_elev_estimate_array[x][y][skip_level]<9999 :
                            ground_sum+=ground_elev_estimate_array[x][y][skip_level]
                            ground_count+=1
                            data_list.append(ground_elev_estimate_array[x][y][skip_level])

                mean_ground_level=ground_sum/ground_count
                median_ground_level=statistics.median(data_list)
                for y in range(number_of_y_cells) :
                    for x in range(number_of_x_cells) :
                        if abs(ground_elev_estimate_array[x][y][skip_level]-median_ground_level)>ground_difference_threshold and ground_elev_estimate_array[x][y][skip_level]<9999:
                            print(f"Removed ground estimate, {ground_elev_estimate_array[x][y][skip_level]} vs median {median_ground_level} (mean {mean_ground_level})")
                            removed_cells+=1
                            ground_elev_estimate_array[x][y][skip_level]=nodata
                        
                k=100
                c = cmap_discretize('jet', k)
                
                #plt.subplot(121)
                #skipped_ground_elev_estimate_array=[[ground_elev_estimate_array[i][j][skip_level] for j in range(number_of_y_cells)] for i in range(number_of_x_cells)]
                skipped_ground_elev_estimate_array=np.empty((number_of_x_cells,number_of_y_cells),float)
                for j in range(number_of_y_cells) :
                    for i in range(number_of_x_cells) :
                        skipped_ground_elev_estimate_array[i][j]=ground_elev_estimate_array[i][j][skip_level]

                max_ground_elevation_step=nodata
                for j in range(number_of_y_cells-1) :
                    for i in range(number_of_x_cells-1) :
                        if skipped_ground_elev_estimate_array[i][j]!=nodata and skipped_ground_elev_estimate_array[i+1][j]!=nodata :
                            if abs(skipped_ground_elev_estimate_array[i][j]-skipped_ground_elev_estimate_array[i+1][j])>max_ground_elevation_step or max_ground_elevation_step==nodata :
                                   max_ground_elevation_step=abs(skipped_ground_elev_estimate_array[i][j]-skipped_ground_elev_estimate_array[i+1][j])
                        if skipped_ground_elev_estimate_array[i][j]!=nodata and skipped_ground_elev_estimate_array[i][j+1]!=nodata :
                            if abs(skipped_ground_elev_estimate_array[i][j]-skipped_ground_elev_estimate_array[i][j+1])>max_ground_elevation_step or max_ground_elevation_step==nodata :
                                   max_ground_elevation_step=abs(skipped_ground_elev_estimate_array[i][j]-skipped_ground_elev_estimate_array[i][j+1])
                max_ground_elevation_step_per_km=max_ground_elevation_step*(1000/x_cell_width_m)
                output_summary_file.write(f"For cell width {x_cell_width_m}m, max step {max_ground_elevation_step:4.1f}m, {max_ground_elevation_step_per_km:4.1f}m per km\n")
                vertical_plot_extent=maximum_ground-minimum_ground # 10
                if "to" in input_filename :
                    output_folder=input_folder+"Results"+version+"\\Ferry\\"+input_filename[:-4]+"\\"
                else :
                    output_folder=input_folder+"Results"+version+"\\Site\\"+input_filename[:-4]+"\\"
                if not os.path.isdir(output_folder) :
                    os.makedirs(output_folder)
                if verbose !='skip' :
                    plt.imshow(np.transpose(skipped_ground_elev_estimate_array),interpolation='nearest',cmap=c,origin='lower',vmin=minimum_ground,vmax=minimum_ground+vertical_plot_extent,extent=[westmost,eastmost,southmost,northmost])
                    plt.colorbar()
                    plt.title(f"{input_filename}\nCell size {x_cell_width_m:4.0f},{y_cell_width_m}m, Skip level {skip_level}\n Ground min/max/diff {minimum_ground} {maximum_ground} {maximum_ground-minimum_ground:4.2f}\nMax Step {max_ground_elevation_step:4.1f}m, {max_ground_elevation_step_per_km:4.1f}m per km")
                        
                    #plt.subplot(122)
                    #plt.imshow(ground_elev_estimate_array,interpolation='nearest',cmap=c,vmin=minimum_ground)

                    #cb = plt.colorbar()
                    #labels = np.arange(0,k,1)
                    #loc    = labels + .5
                    #cb.set_ticks(loc)
                    #cb.set_ticklabels(labels)

                                    

                    if verbose == 'plot' :
                        plt.show()
                    else :
                        if "to" in input_filename :
                            output_folder=input_folder+"Results"+version+"\\Ferry\\"+input_filename[:-4]+"\\"
                        else :
                            output_folder=input_folder+"Results"+version+"\\Site\\"+input_filename[:-4]+"\\"
                        if not os.path.isdir(output_folder) :
                            os.makedirs(output_folder)
                        plt.savefig(output_folder+input_filename[:-4]+"_GroundMap_"+str(x_cell_width_m)+"m_skip"+str(skip_level)+"_"+pc+".png", format="png",dpi=1200)
                    plt.close()

                # Output to geotiff
                geotransform = (westmost, x_cell_lon_width, 0, southmost, 0, y_cell_lat_width)
                outFileName=output_folder+input_filename[:-4]+"_GroundMap_"+str(x_cell_width_m)+"m_skip"+str(skip_level)+"_"+pc+".tif"
                driver = gdal.GetDriverByName("GTiff").Create(outFileName, number_of_x_cells, number_of_y_cells, 1, gdal.GDT_Float32)
                driver.SetGeoTransform(geotransform)    # specify coords
                srs = osr.SpatialReference()            # establish encoding
                srs.ImportFromEPSG(3857)                # WGS84 lat/long
                driver.GetRasterBand(1).WriteArray(np.transpose(skipped_ground_elev_estimate_array))
                driver.GetRasterBand(1).SetNoDataValue(nodata)##if you want these values transparent
                driver.FlushCache() ##saves to disk!!

                # Output as ASCII
                ascii_output_filename=output_folder+input_filename[:-4]+"_GroundMap_"+str(x_cell_width_m)+"m_skip"+str(skip_level)+"_"+pc+".txt"
                ascii_output_file=open(ascii_output_filename,"w")
                ascii_output_file.write("Lon, Lat, Ground_elev\n")
                for y_cell_ascii in range(number_of_y_cells) :
                    for x_cell_ascii in range(number_of_x_cells) :
                        lon_ascii=x_cell_ascii*x_cell_lon_width+westmost
                        lat_ascii=y_cell_ascii*y_cell_lat_width+southmost
                        if skipped_ground_elev_estimate_array[x_cell_ascii][y_cell_ascii] != nodata :
                            ascii_output_file.write(f"{lon_ascii}, {lat_ascii}, {skipped_ground_elev_estimate_array[x_cell_ascii][y_cell_ascii]}\n")
                ascii_output_file.close()
                
                for layer_depth_m in [0.25] :
                    print(f"Third stage - Building elevation return histogram from file {input_filename}")
                    
                    elev_range=max_elev-min_elev
                    number_of_layers=int(elev_range/layer_depth_m+1)
                    
                    print(f"{number_of_layers} layers, between {min_elev} and {max_elev}m")

                    return_count_per_layer= [0 for i in range(number_of_layers)]
                    elevations_list=np.arange(0.0,(number_of_layers-0.5)*layer_depth_m,layer_depth_m)
                    
                    
                    #height_above_ground_list=[]
                    fails=0
                    for array_point in range(number_of_lidar_points_in_array) :
                        lon=lon_array[array_point]
                        lat=lat_array[array_point]
                        elev=elev_array[array_points]
                        current_cell_x=int((lon-westmost)/x_cell_lon_width)
                        current_cell_y=int((lat-southmost)/y_cell_lat_width)
                        
                        if (current_cell_x>=0) and (current_cell_x<number_of_x_cells) and (current_cell_y>=0) and (current_cell_y<number_of_y_cells) and elev > elev_lower_threshold and elev < elev_upper_threshold :
                            if (skipped_ground_elev_estimate_array[current_cell_x][current_cell_y]>9999) :
                                #print("No ground estimate!")
                                fails=fails+1
                                #sys.exit()
                            else :
                                height_above_ground=elev-skipped_ground_elev_estimate_array[current_cell_x][current_cell_y]
                                height_bin=int(height_above_ground/layer_depth_m)
                                #height_above_ground_list.append(height_above_ground)
                                return_count_per_layer[height_bin]+=1        
                    
                    #plot histogram
                    #plt.hist(height_above_ground_list, density=True, bins=number_of_layers)  # `density=False` would make counts
                    if verbose!='skip' :
                        fig=plt.figure()
                        ax=fig.add_subplot(111)
                        area = np.pi*3
                        ax.plot(elevations_list, return_count_per_layer)
                        ax.set_xlim(0,50)
                        plt.ylabel('Frequency')
                        plt.xlabel('Elevation')
                        plt.title(f"{input_filename}\nCell size {x_cell_width_m:4.0f},{y_cell_width_m}m, Skip level {skip_level}. layer depth {layer_depth_m}m, {fails} fails\n Ground min/max/dif {minimum_ground} {maximum_ground} {maximum_ground-minimum_ground:4.2f}")
                        #plt.show()
                        if verbose == 'plot' :
                            plt.show()
                        else :
                            if "to" in input_filename :
                                output_folder=input_folder+"Results"+version+"\\Ferry\\"+input_filename[:-4]+"\\"
                            else :
                                output_folder=input_folder+"Results"+version+"\\Site\\"+input_filename[:-4]+"\\"
                            if not os.path.isdir(output_folder) :
                                os.makedirs(output_folder)
                            png_filename=output_folder+input_filename[:-4]+"_cell_"+str(x_cell_width_m)+"m_layerdepth_"+str(layer_depth_m)+"m_skip"+str(skip_level)+"_"+pc+".png"
                            plt.savefig(png_filename, format="png",dpi=600)
                        plt.close()
                    
                    # Now build CHP using elevations_list, return_count_per_layer lists
                    ground_return_elevation=1.0 # metres from ground to assume is ground return
                    ground_layers=int(ground_return_elevation/layer_depth_m)
                    print("Fourth stage. Building canopy height profile")
                    chp=[0.0 for k in range(len(elevations_list))]
                    for layer in range(ground_layers,len(elevations_list)) :
                        chp[layer]=(-np.log(1-return_count_per_layer[layer]/sum(return_count_per_layer[0:layer+1])))
                    if verbose!='skip' :
                        fig=plt.figure()
                        ax=fig.add_subplot(111)
                        area = np.pi*3
                        ax.plot(elevations_list, chp)
                        ax.set_xlim(0,50)
                        ax.set_ylim(0,max(chp))
                        #ax.set_xlim(0,50)
                        plt.ylabel('CHP')
                        plt.xlabel('Elevation')
                        plt.title(f"{input_filename}\nCell size {x_cell_width_m:4.0f},{y_cell_width_m}m, Skip level {skip_level}. layer depth {layer_depth_m}m, {fails} fails\n Ground min/max/dif {minimum_ground} {maximum_ground} {maximum_ground-minimum_ground:4.2f}")
                        if verbose == 'plot' :
                            plt.show()
                        else :
                            if "to" in input_filename :
                                output_folder=input_folder+"Results"+version+"\\Ferry\\"+input_filename[:-4]+"\\"
                            else :
                                output_folder=input_folder+"Results"+version+"\\Site\\"+input_filename[:-4]+"\\"
                            if not os.path.isdir(output_folder) :
                                os.makedirs(output_folder)
                            png_filename=output_folder+input_filename[:-4]+"_CHP_cell_"+str(x_cell_width_m)+"m_layerdepth_"+str(layer_depth_m)+"m_skip"+str(skip_level)+"_"+pc+".png"
                            plt.savefig(png_filename, format="png",dpi=600)
                        plt.close()

                    # LAI cumulative back to layer 1
                    lai_cumul=[0.0 for k in range(len(elevations_list))]
                    for layer in range(ground_layers,len(elevations_list)) :
                        lai_cumul[layer]=sum(chp[layer:])
                    if verbose!='skip' :
                        fig=plt.figure()
                        ax=fig.add_subplot(111)
                        area = np.pi*3
                        ax.plot(elevations_list, lai_cumul)
                        ax.set_xlim(0,50)
                        ax.set_ylim(0,max(lai_cumul))
                        #ax.set_xlim(0,50)
                        plt.ylabel('LAI cumulative')
                        plt.xlabel('Elevation')
                        plt.title(f"{input_filename}\nCell size {x_cell_width_m:4.0f},{y_cell_width_m}m, Skip level {skip_level}. layer depth {layer_depth_m}m, {fails} fails\n Ground min/max/dif {minimum_ground} {maximum_ground} {maximum_ground-minimum_ground:4.2f}\nEstd LAI {lai_cumul[ground_layers]:4.2f}")
                        if verbose == 'plot' :
                            plt.show()
                        else :
                            if "to" in input_filename :
                                output_folder=input_folder+"Results"+version+"\\Ferry\\"+input_filename[:-4]+"\\"
                            else :
                                output_folder=input_folder+"Results"+version+"\\Site\\"+input_filename[:-4]+"\\"
                            if not os.path.isdir(output_folder) :
                                os.makedirs(output_folder)
                            png_filename=output_folder+input_filename[:-4]+"_LAI_cumul_cell_"+str(x_cell_width_m)+"m_layerdepth_"+str(layer_depth_m)+"m_skip"+str(skip_level)+"_"+pc+".png"
                            plt.savefig(png_filename, format="png",dpi=600)
                        plt.close()

                    # Output all to text file
                    output_filename=output_summary_folder+input_filename[:-4]+"_profile_"+str(x_cell_width_m)+"m.txt"
                    output_file=open(output_filename,"w")
                    output_file.write("Elevation, Return_frequency, CHP, Cumul_LAI\n")
                    for layer in range(0,len(elevations_list)) :
                        output_file.write(f"{elevations_list[layer]}, {return_count_per_layer[layer]}, {chp[layer]}, {lai_cumul[layer]}\n")
                    output_file.close()
                    #sys.exit()

                    #
                    #
                    if estimate_canopy=='yes' :

                        
                        # Map CHP/LAI per cell
                        print(f"Fifth stage - Building elevation return histogram from file cell {input_filename}")
                        canopy_estimate_array=np.empty((number_of_x_cells,number_of_y_cells),float)
                        canopy_maximum=0
                        for canopy_cell_y in range(number_of_y_cells) :
                            for canopy_cell_x in range(number_of_x_cells) :
                                #print(f"Cell ({canopy_cell_x} of {number_of_x_cells}, {canopy_cell_y} of {number_of_y_cells})")
                                if (skipped_ground_elev_estimate_array[canopy_cell_x][canopy_cell_y]==nodata) :
                                    canopy_estimate_array[canopy_cell_x][canopy_cell_y]=nodata
                                else :
                                    return_count_per_layer= [0 for i in range(number_of_layers)]
                                    elevations_list=np.arange(0.0,(number_of_layers-0.5)*layer_depth_m,layer_depth_m)
                                    
                                    #height_above_ground_list=[]
                                    elevations_in_cell=[]
                                    min_current_cell_x=nodata
                                    max_current_cell_x=nodata
                                    min_current_cell_y=nodata
                                    max_current_cell_y=nodata
                                    for array_point in range(number_of_lidar_points_in_array) :
                                        lon=lon_array[array_point]
                                        lat=lat_array[array_point]
                                        elev=elev_array[array_points]
                                        current_cell_x=int((lon-westmost)/x_cell_lon_width)
                                        current_cell_y=int((lat-southmost)/y_cell_lat_width)
                                        current_cell_x_float=((lon-westmost)/x_cell_lon_width)
                                        current_cell_y_float=((lat-southmost)/y_cell_lat_width)
                                        #if current_cell_x<min_current_cell_x or min_current_cell_x==nodata :
                                        #    min_current_cell_x=current_cell_x
                                        #    print(f"({current_cell_x},{current_cell_y}), range ({min_current_cell_x}-{max_current_cell_x}),({min_current_cell_y},{max_current_cell_y})")
                                        #if current_cell_y<min_current_cell_y or min_current_cell_y==nodata :
                                        #    min_current_cell_y=current_cell_y
                                        #    print(f"({current_cell_x},{current_cell_y}), range ({min_current_cell_x}-{max_current_cell_x}),({min_current_cell_y},{max_current_cell_y})")
                                        #if current_cell_x>max_current_cell_x or max_current_cell_x==nodata :
                                        #    max_current_cell_x=current_cell_x
                                        #    print(f"({current_cell_x},{current_cell_y}), range ({min_current_cell_x}-{max_current_cell_x}),({min_current_cell_y},{max_current_cell_y})")
                                        #if current_cell_y>max_current_cell_y or max_current_cell_y==nodata :
                                        #    max_current_cell_y=current_cell_y
                                        #    print(f"({current_cell_x},{current_cell_y}), range ({min_current_cell_x}-{max_current_cell_x}),({min_current_cell_y},{max_current_cell_y})")
                                       
                                        if lon > -30 and lat > -30 and lon < 30 and lat < 30 and elev > elev_lower_threshold and elev < elev_upper_threshold :
                                            if (current_cell_x==canopy_cell_x) and (current_cell_y==canopy_cell_y) and elev > elev_lower_threshold and elev < elev_upper_threshold :
                                                # Estimate ground elevation in cell, using average x-x and y-y interpolation
                                                if current_cell_x_float-current_cell_x<0.5 :
                                                    if current_cell_x>0 :
                                                        if skipped_ground_elev_estimate_array[current_cell_x-1][current_cell_y] != nodata :
                                                            cell_weight=0.5-(current_cell_x_float-current_cell_x)
                                                            improved_ground_estimate_x=cell_weight*skipped_ground_elev_estimate_array[current_cell_x][current_cell_y] + (1-cell_weight) * skipped_ground_elev_estimate_array[current_cell_x-1][current_cell_y]
                                                        else :
                                                            improved_ground_estimate_x=nodata
                                                    else :
                                                            improved_ground_estimate_x=nodata
                                                else :
                                                    if current_cell_x<number_of_x_cells-1 :
                                                        if skipped_ground_elev_estimate_array[current_cell_x+1][current_cell_y] != nodata :
                                                            cell_weight=1.5-(current_cell_x_float-current_cell_x)
                                                            improved_ground_estimate_x=cell_weight*skipped_ground_elev_estimate_array[current_cell_x][current_cell_y] + (1-cell_weight) * skipped_ground_elev_estimate_array[current_cell_x+1][current_cell_y]
                                                        else :
                                                            improved_ground_estimate_x=nodata
                                                    else :
                                                        improved_ground_estimate_x=nodata
                                                        
                                                if current_cell_y_float-current_cell_y<0.5 :
                                                    if current_cell_y>0 :
                                                        if skipped_ground_elev_estimate_array[current_cell_x][current_cell_y-1] != nodata :
                                                            cell_weight=0.5-(current_cell_y_float-current_cell_y)
                                                            improved_ground_estimate_y=cell_weight*skipped_ground_elev_estimate_array[current_cell_x][current_cell_y] + (1-cell_weight) * skipped_ground_elev_estimate_array[current_cell_x][current_cell_y-1]
                                                        else :
                                                            improved_ground_estimate_y=nodata
                                                    else :
                                                        improved_ground_estimate_y=nodata
                                                else :
                                                    if current_cell_y<number_of_y_cells-1 :
                                                        if skipped_ground_elev_estimate_array[current_cell_x][current_cell_y+1] != nodata :
                                                            cell_weight=1.5-(current_cell_y_float-current_cell_y)
                                                            improved_ground_estimate_y=cell_weight*skipped_ground_elev_estimate_array[current_cell_x][current_cell_y] + (1-cell_weight) * skipped_ground_elev_estimate_array[current_cell_x][current_cell_y+1]
                                                        else :
                                                            improved_ground_estimate_y=nodata
                                                    else :
                                                        improved_ground_estimate_y=nodata


                                                if (improved_ground_estimate_x != nodata) and (improved_ground_estimate_y != nodata) :
                                                    improved_estimate=(improved_ground_estimate_x+improved_ground_estimate_y)/2
                                                else :
                                                    if improved_ground_estimate_x != nodata :
                                                        improved_estimate=improved_ground_estimate_x
                                                    else :
                                                        improved_estimate=improved_ground_estimate_y
                                                #print (f"Original estimate {skipped_ground_elev_estimate_array[current_cell_x][current_cell_y]:4.2f}, improved {improved_estimate:4.2f}")


                                                if improved_estimate!=nodata :
                                                    ground_level_estimate=improved_estimate
                                                    #if abs(improved_estimate-skipped_ground_elev_estimate_array[current_cell_x][current_cell_y])>10 :
                                                    #    print(f"Big 'improvement' of {improved_estimate-skipped_ground_elev_estimate_array[current_cell_x][current_cell_y]:4.2f}m, from {skipped_ground_elev_estimate_array[current_cell_x][current_cell_y]:4.2f} to {improved_estimate:4.2f}")
                                                    #    sys.exit()
                                                else :
                                                    ground_level_estimate=skipped_ground_elev_estimate_array[current_cell_x][current_cell_y]
                                                height_above_ground=elev-ground_level_estimate
                                                height_bin=int(height_above_ground/layer_depth_m)
                                                #height_above_ground_list.append(height_above_ground)
                                                return_count_per_layer[height_bin]+=1
                                                elevations_in_cell.append(height_above_ground)
                                        

                                    # Estimate canopy height from returns
                                    if len(elevations_in_cell)>0 :
                                        if height_type=='percentile75' :
                                            canopy_estimate_array[canopy_cell_x][canopy_cell_y]=np.percentile(elevations_in_cell,75)
                                        elif height_type=='percentile95' :
                                            canopy_estimate_array[canopy_cell_x][canopy_cell_y]=np.percentile(elevations_in_cell,75)
                                            
                                        if canopy_estimate_array[canopy_cell_x][canopy_cell_y] > canopy_maximum :
                                            canopy_maximum=canopy_estimate_array[canopy_cell_x][canopy_cell_y]
                                        #print(f"Canopy estimate {canopy_estimate_array[canopy_cell_x][canopy_cell_y]}m")

                                        #plot histogram
                                        #plt.hist(height_above_ground_list, density=True, bins=number_of_layers)  # `density=False` would make counts
                                        if verbose!='skip' :
                                            fig=plt.figure()
                                            ax=fig.add_subplot(111)
                                            area = np.pi*3
                                            ax.plot(elevations_list, return_count_per_layer)
                                            ax.set_xlim(0,50)
                                            plt.ylabel('Frequency')
                                            plt.xlabel('Elevation')
                                            plt.title(f"{input_filename}\nCell size {x_cell_width_m:4.0f},{y_cell_width_m}m, Skip level {skip_level}. layer depth {layer_depth_m}m\n Ground min/max/dif {minimum_ground} {maximum_ground} {maximum_ground-minimum_ground:4.2f}")
                                            #plt.show()
                                            if verbose == 'plot' :
                                                plt.show()
                                            else :
                                                if "to" in input_filename :
                                                    output_folder=input_folder+"Results"+version+"\\Ferry\\"+input_filename[:-4]+"\\"
                                                else :
                                                    output_folder=input_folder+"Results"+version+"\\Site\\"+input_filename[:-4]+"\\"
                                                if not os.path.isdir(output_folder) :
                                                    os.makedirs(output_folder)
                                                png_filename=output_folder+input_filename[:-4]+"_cell_"+str(canopy_cell_x)+"_"+str(canopy_cell_y)+"_cellwidth_"+str(x_cell_width_m)+"m_layerdepth_"+str(layer_depth_m)+"m_skip"+str(skip_level)+"_"+pc+".png"
                                                plt.savefig(png_filename, format="png",dpi=600)
                                            plt.close()
                                        
                                        # Now build CHP using elevations_list, return_count_per_layer lists
                                        ground_return_elevation=1.0 # metres from ground to assume is ground return
                                        ground_layers=int(ground_return_elevation/layer_depth_m)
                                        #print("Sixth stage. Building canopy height profile")
                                        chp=[0.0 for k in range(len(elevations_list))]
                                        for layer in range(ground_layers,len(elevations_list)) :
                                            if sum(return_count_per_layer[0:layer+1])>0 :
                                                chp[layer]=(-np.log(1-return_count_per_layer[layer]/sum(return_count_per_layer[0:layer+1])))
                                            else :
                                                chp[layer]=-999
                                        if verbose!='skip' :
                                            fig=plt.figure()
                                            ax=fig.add_subplot(111)
                                            area = np.pi*3
                                            ax.plot(elevations_list, chp)
                                            ax.set_xlim(0,50)
                                            ax.set_ylim(0,max(chp))
                                            #ax.set_xlim(0,50)
                                            plt.ylabel('CHP')
                                            plt.xlabel('Elevation')
                                            plt.title(f"{input_filename}\nCell size {x_cell_width_m:4.0f},{y_cell_width_m}m, Skip level {skip_level}. layer depth {layer_depth_m}m\n Ground min/max/dif {minimum_ground} {maximum_ground} {maximum_ground-minimum_ground:4.2f}")
                                            if verbose == 'plot' :
                                                plt.show()
                                            else :
                                                if "to" in input_filename :
                                                    output_folder=input_folder+"Results"+version+"\\Ferry\\"+input_filename[:-4]+"\\"
                                                else :
                                                    output_folder=input_folder+"Results"+version+"\\Site\\"+input_filename[:-4]+"\\"
                                                if not os.path.isdir(output_folder) :
                                                    os.makedirs(output_folder)
                                                png_filename=output_folder+input_filename[:-4]+"_CHP_cell_"+str(x_cell_width_m)+"m_layerdepth_"+str(layer_depth_m)+"m_skip"+str(skip_level)+"_"+pc+".png"
                                                plt.savefig(png_filename, format="png",dpi=600)
                                            plt.close()

                                        # LAI cumulative back to layer 1
                                        lai_cumul=[0.0 for k in range(len(elevations_list))]
                                        for layer in range(ground_layers,len(elevations_list)) :
                                            lai_cumul[layer]=sum(chp[layer:])
                                        if verbose!='skip' :
                                            fig=plt.figure()
                                            ax=fig.add_subplot(111)
                                            area = np.pi*3
                                            ax.plot(elevations_list, lai_cumul)
                                            ax.set_xlim(0,50)
                                            ax.set_ylim(0,max(lai_cumul))
                                            #ax.set_xlim(0,50)
                                            plt.ylabel('LAI cumulative')
                                            plt.xlabel('Elevation')
                                            plt.title(f"{input_filename}\nCell size {x_cell_width_m:4.0f},{y_cell_width_m}m, Skip level {skip_level}. layer depth {layer_depth_m}m\n Ground min/max/dif {minimum_ground} {maximum_ground} {maximum_ground-minimum_ground:4.2f}\nEstd LAI {lai_cumul[ground_layers]:4.2f}")
                                            if verbose == 'plot' :
                                                plt.show()
                                            else :
                                                if "to" in input_filename :
                                                    output_folder=input_folder+"Results"+version+"\\Ferry\\"+input_filename[:-4]+"\\"
                                                else :
                                                    output_folder=input_folder+"Results"+version+"\\Site\\"+input_filename[:-4]+"\\"
                                                if not os.path.isdir(output_folder) :
                                                    os.makedirs(output_folder)
                                                png_filename=output_folder+input_filename[:-4]+"_LAI_cumul_cell_"+str(x_cell_width_m)+"m_layerdepth_"+str(layer_depth_m)+"m_skip"+str(skip_level)+"_"+pc+".png"
                                                plt.savefig(png_filename, format="png",dpi=600)
                                            plt.close()
                                        print(f"Cell ({canopy_cell_x} of {number_of_x_cells}, {canopy_cell_y} of {number_of_y_cells}) = {canopy_estimate_array[canopy_cell_x][canopy_cell_y]:4.2f}m")
                                    else :
                                        print(f"Cell ({canopy_cell_x} of {number_of_x_cells}, {canopy_cell_y} of {number_of_y_cells}) = -")
                                        canopy_estimate_array[canopy_cell_x][canopy_cell_y]=nodata

                        # plot grid
                        if verbose!='skip' :
                            plt.close()
                            plt.imshow(np.transpose(canopy_estimate_array),interpolation='nearest',cmap=c,vmin=0,origin='lower',vmax=canopy_maximum,extent=[westmost,eastmost,southmost,northmost])
                            plt.colorbar()
                            plt.title(f"{input_filename}\nCell size {x_cell_width_m:4.0f},{y_cell_width_m}m, Skip level {skip_level}\n Ground min/max/diff {minimum_ground} {maximum_ground} {maximum_ground-minimum_ground:4.2f}\nRemoved cells {removed_cells}")

                            if verbose == 'plot' :
                                plt.show()
                            else :
                                if "to" in input_filename :
                                    output_folder=input_folder+"Results"+version+"\\Ferry\\"+input_filename[:-4]+"\\"
                                else :
                                    output_folder=input_folder+"Results"+version+"\\Site\\"+input_filename[:-4]+"\\"
                                if not os.path.isdir(output_folder) :
                                    os.makedirs(output_folder)
                                plt.savefig(output_folder+input_filename[:-4]+"_CanopyMap_"+str(x_cell_width_m)+"m_skip"+str(skip_level)+"_"+height_type+"_"+pc+".png", format="png",dpi=1200)
                            plt.close()
                        # Output tiff
                        geotransform = (westmost, x_cell_lon_width, 0, southmost, 0, y_cell_lat_width)
                        outFileName=output_folder+input_filename[:-4]+"_CanopyMap_"+str(x_cell_width_m)+"m_skip"+str(skip_level)+"_"+height_type+"_"+pc+".tif"
                        driver = gdal.GetDriverByName("GTiff").Create(outFileName, number_of_x_cells, number_of_y_cells, 1, gdal.GDT_Float32)
                        driver.SetGeoTransform(geotransform)    # specify coords
                        srs = osr.SpatialReference()            # establish encoding
                        srs.ImportFromEPSG(3857)                # WGS84 lat/long
                        driver.GetRasterBand(1).WriteArray(np.transpose(canopy_estimate_array))
                        driver.GetRasterBand(1).SetNoDataValue(nodata)##if you want these values transparent
                        driver.FlushCache() ##saves to disk!!

                        # Output as ASCII
                        ascii_output_filename=output_folder+input_filename[:-4]+"_CanopyMap_"+str(x_cell_width_m)+"m_skip"+str(skip_level)+"_"+height_type+"_"+pc+".txt"
                        ascii_output_file=open(ascii_output_filename,"w")
                        ascii_output_file.write("Lon, Lat, Ground_elev\n")
                        for y_cell_ascii in range(number_of_y_cells) :
                            for x_cell_ascii in range(number_of_x_cells) :
                                lon_ascii=x_cell_ascii*x_cell_lon_width+westmost
                                lat_ascii=y_cell_ascii*y_cell_lat_width+southmost
                                if canopy_estimate_array[x_cell_ascii][y_cell_ascii] != nodata :
                                    ascii_output_file.write(f"{lon_ascii}, {lat_ascii}, {canopy_estimate_array[x_cell_ascii][y_cell_ascii]}\n")
                        ascii_output_file.close()

        input_file.close()
        print(f"Completed {input_filename}")            
        output_summary_file.close()

        
        
