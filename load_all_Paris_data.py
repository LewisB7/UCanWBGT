
"""
  LOAD_ALL_PARIS_DATA: does the loading step for both observation and model data from Paris.


Notes from Lewis...
# notes on updates that I have realised need to be made to the main branch through doing this work:
# - add "if isinstance(array_or_float, int): array_or_float = float(array_or_float)" in adjust_array_shape
# - add "print(f"!!! {100 * num_H_less_than_Z_and_gn_0_true / num_H_less_than_Z_true}% of Z < H points had H > 0 !!!")"
# - add "for arr in enumerate(array_or_float_list): array_or_float_list[i][mask] = np.nan"
# - recommend 8 not 9 for urban

#%%

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import iris
import pvlib
import sys
sys.path.append("/home/h04/lblunn/Documents/Projects/UCanWBGT")  
    # Add the directory containing UCanWBGT.py to the Python path
import UCanWBGT
import importlib
importlib.reload(UCanWBGT)

#--- Import parameters from the "parameters.py" file.
from parameters import (tzinfo, WBGT_model_choice, nref, gamma_choice, WBGT_equation_choice,
                        Twb_method, gamma, elevation, Z, 
                        alb_grnd_tiles, alb_wall, emiss_grnd_tiles, emiss_wall,
                        emiss_g, a_g, d, a_SW, a_LW)

#--- Read in the observation data from file.
print("*** Reading in observation data... ")
obs_frame = pd.read_pickle("/data/users/joshonk/data/U23/BGTfiles/Paris_BGT_all-sites_2023-09-07_to_2023-09-08.pkl")
print("Data Frame: \n",obs_frame)

#--- Specify names and morphological properties at the six urban BGT sites.
sites_obs = ["Rue_de_Rivoli", "Bvd_St_Germain", "Rue_Jacques_Callot",
             "Avenue_Marechal_Galieni_Invalides", "Place_Vendome",
             "Quai_Anatole_France"]
X_obs                 = [  0.0,      0.0,      0.0,      0.0,      0.0,      0.0    ]
canyon_orient_deg_obs = [115.0,    110.0,     90.0,      4.0,     33.0,    115.0    ]
tile_number_obs       = [  8,        8,        8,        8,        8,        8      ]
lat_obs               = [ 48.85936, 48.85381, 48.85527, 48.86121, 48.86742, 48.86111] 
lon_obs               = [  2.34568,  2.33358,  2.33708,  2.31300,  2.32943,  2.32504]
H_obs                 = [ 32.0,     28.0,     30.0,     24.0,     16.0,      8.0    ]
W_obs                 = [ 35.0,     20.0,     10.0,    260.0,    115.0,     25.0    ]

nsite = len(sites_obs)

#--- Extract necessary variables into Numpy arrays.
BGT_arrays = [obs_frame.loc[site, "BGT"].to_numpy() for site in sites_obs]
BGT_obs = np.vstack(BGT_arrays).astype(np.float64)
wind_arrays = [obs_frame.loc[site, "Wind_Speed"].to_numpy() for site in sites_obs]
wind_obs = np.vstack(wind_arrays).astype(np.float64)
temp_arrays = [obs_frame.loc[site, "Temperature"].to_numpy() for site in sites_obs]
temp_obs = np.vstack(temp_arrays).astype(np.float64)
RH_arrays = [obs_frame.loc[site, "Humidity"].to_numpy() for site in sites_obs]
RH_obs = np.vstack(RH_arrays).astype(np.float64)

#--- Read in model data from file.
print("*** Reading in model data...")
cubel = iris.load('/data/users/lblunn/Heat_Stress/Vinod_RNS_Data/paris_20230906T1200Z_48hrfcst.nc')
print("cubel original:\n",cubel)

#--- Cut out the central part of the domain from the cube using Iris constraints.
def spatial_constraint(cube,max_longitude,min_longitude,max_latitude,min_latitude):

    longitude_constraint = iris.Constraint(longitude=lambda cell: min_longitude <= cell <= max_longitude)
    latitude_constraint = iris.Constraint(latitude=lambda cell: min_latitude <= cell <= max_latitude)
    constrained_cube = cube.extract(longitude_constraint & latitude_constraint)

    return constrained_cube

clon = 2.34     # central longitude
clat = 48.87    # central latitude
delta = 0.3     # half-width of cut-out section in degrees
max_longitude = clon + delta
min_longitude = clon - delta
max_latitude = clat + delta
min_latitude = clat - delta

cubel = spatial_constraint(cubel,max_longitude,min_longitude,max_latitude,min_latitude)
print("cubel spatial constraint:\n",cubel)

#--- Extract the cube data on half-hourly time steps to match the observations.
#    Again, we use Iris constraints.
def interpolate_times(cubel,start_time,end_time,dt):

    cubel_out = iris.cube.CubeList([])
    for cube in cubel:
        coordinate_names = [coord.name() for coord in cube.coords()]
        has_time_dimension = "time" in coordinate_names
        if has_time_dimension:
            if cube.name() == 'lai':
                cube = cube.extract(iris.Constraint(time=lambda cell: cell.point.month == start_time.month))
            else:
                existing_times = cube.coord('time').points
                start_constraint = iris.Constraint(time=lambda cell: cell.point == start_time)
                end_constraint = iris.Constraint(time=lambda cell: cell.point == end_time)
                start_time_pnt = cube.extract(start_constraint).coord('time').points
                end_time_pnt = cube.extract(end_constraint).coord('time').points
                times = np.arange(start_time_pnt,end_time_pnt,dt)
                missing_times = []
                for time in times:
                    if time not in existing_times:
                        missing_times.append(time)
                if len(missing_times) > 0:
                    cube = cube.interpolate([('time', times)], iris.analysis.Linear())
                else:
                    time_contraint = iris.Constraint(time=lambda cell: start_time <= cell < end_time)
                    cube = cube.extract(time_contraint)
        else:
            pass
        cubel_out.append(cube)

    return cubel_out

step_time = 0.5  # output time step in hours
start_time_dt = datetime.strptime('20230907T0000Z','%Y%m%dT%H%MZ')  # start time
end_time_dt = datetime.strptime('20230908T0000Z','%Y%m%dT%H%MZ') # end time
start_time = iris.time.PartialDateTime(year=start_time_dt.year, month=start_time_dt.month, \
    day=start_time_dt.day, hour=start_time_dt.hour, minute=start_time_dt.minute)
end_time = iris.time.PartialDateTime(year=end_time_dt.year, month=end_time_dt.month, \
    day=end_time_dt.day, hour=end_time_dt.hour, minute=end_time_dt.minute)
cubel = interpolate_times(cubel,start_time,end_time,dt=step_time)
print("cubel 24 hours at half hourly interval:\n",cubel)

#--- Now extract all of the fields we need from the cube list and convert them to Numpy arrays.
print("*** Extracting variables to individual Iris cubes...")
T_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s03i236'))[0]
print("T_cube:\n",T_cube)
RH_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s03i245'))[0]
print("RH_cube:\n",RH_cube)
Ld_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s02i207'))[0]
print("Ld_cube:\n",Ld_cube)
Id_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s01i215'))[0]
print("Id_cube:\n",Id_cube)
Kd_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s01i216'))[0]
print("Kd_cube:\n",Kd_cube)  # 
P_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i409'))[0]
print("P_cube:\n",P_cube)  # pressure
WS_X_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i002'))[0] # data is 1 m not 1.5 m
print("WS_X_cube:\n",WS_X_cube)  # zonal wind speed
WS_Y_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i003'))[0] # data is 1 m not 1.5 m
print("WS_Y_cube:\n",WS_Y_cube) # meridional wind speed
T_surf_tiles_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s03i316'))[0]
print("T_surf_tiles_cube:\n",T_surf_tiles_cube)  # surface temperature on tiles
H_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i494'))[0]
print("H_cube:\n",H_cube)   # building height
HWR_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i495'))[0]
print("HWR_cube:\n",HWR_cube)  # canyon height-to-width ratio
LAI_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i217'))[0]
print("LAI_cube:\n",LAI_cube)  # leaf area index
LCF_tiles_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i216'))[0]
print("LCF_tiles_cube:\n",LCF_tiles_cube)  # land fraction on tiles

#--- Calculate wind speed.
WS_cube = WS_X_cube.copy()
WS_cube.data = (WS_X_cube.data**2 + WS_Y_cube.data**2)**0.5

#--- Calculate canyon width W from the height-to-width ratio.
W_cube = HWR_cube.copy()
W_cube.data = H_cube.data / HWR_cube.data

#--- Extract a grid of latitude and longitude
longitude = T_cube.coord('longitude').points
latitude = T_cube.coord('latitude').points
lon_grid, lat_grid = np.meshgrid(longitude, latitude)
print("lon:",np.shape(lon_grid))
print("lat:",np.shape(lat_grid))

#--- Build a time co-ordinate as a date-time object.
current_time = start_time_dt
times = []
while current_time < end_time_dt:
    times.append(current_time.strftime('%Y%m%dT%H%MZ'))
    current_time += timedelta(minutes=30)
#end
timeaxis = np.array(times)

#--- Loop through each site and identify the nearest gridpoint, then pass that data into
#    a Numpy array.
ntime = T_cube.shape[0]
ntile = 10
nvegtile = 5

T_site = np.zeros((nsite,ntime))
RH_site = np.zeros((nsite,ntime))
Ld_site = np.zeros((nsite,ntime))
Id_site = np.zeros((nsite,ntime))
Kd_site = np.zeros((nsite,ntime))
P_site = np.zeros((nsite,ntime))
WS_site = np.zeros((nsite,ntime))
T_surf_tiles_site = np.zeros((ntile,nsite,ntime))
H_site = np.zeros((nsite,))
W_site = np.zeros((nsite,))
LAI_site = np.zeros((nvegtile,nsite))
LCF_tiles_site = np.zeros((ntile,nsite))

for isite in np.arange(nsite):
    print(("Extracting data from site: "+sites_obs[isite]))
    lon_site = lon_sites[isite]
    lat_site = lat_sites[isite]
    delta = 0.0008999999999996788/2. # grid length (in deg) divide 2

   # constrain on lat_site and lon_site
    T_cube_site = spatial_constraint(T_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    RH_cube_site = spatial_constraint(RH_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    Ld_cube_site = spatial_constraint(Ld_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    Id_cube_site = spatial_constraint(Id_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    Kd_cube_site = spatial_constraint(Kd_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    P_cube_site = spatial_constraint(P_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    WS_cube_site = spatial_constraint(WS_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    T_surf_tiles_cube_site = spatial_constraint(T_surf_tiles_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    H_cube_site = spatial_constraint(H_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    W_cube_site = spatial_constraint(W_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    LAI_cube_site = spatial_constraint(LAI_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    tf_cube_site = spatial_constraint(tf_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)
    LCF_tiles_cube_site = spatial_constraint(LCF_tiles_cube,lon_site+delta,lon_site-delta,lat_site+delta,lat_site-delta)

    T_site[isite,:] = T_cube_site.data
    RH_site[isite,:] = RH_cube_site.data
    Ld_site[isite,:] = Ld_cube_site.data
    Id_site[isite,:] = Id_cube_site.data
    Kd_site[isite,:] = Kd_cube_site.data
    P_site[isite,:] = P_cube_site.data
    WS_site[isite,:] = WS_cube_site.data
    T_surf_tiles_site[:,isite,:] = T_surf_tiles_cube_site.data
    H_site[isite] = H_cube_site.data
    W_site[isite] = W_cube_site.data
    LAI_site[:,isite] = LAI_cube_site.data
    LCF_tiles_site[:,isite] = LCF_tiles_cube_site.data
   
#end 

#--- Save model and observed data to a file.
fileroot = "/data/users/joshonk/data/U23/"
filename = "Paris_data_30min_at_sites.npz"

np.savez((fileroot+filename), sites_obs=sites_obs, timeaxis=timeaxis,
         T_site=T_site, RH_site=RH_site, Ld_site=Ld_site, 
         Id_site=Id_site, Kd_site=Kd_site, P_site=P_site, WS_site=WS_site,
         T_surf_tiles_site=T_surf_tiles_site, H_site=H_site, W_site=W_site,
         LAI_site=LAI_site, LCF_tiles_site=LCF_tiles_site,
         BGT_obs=BGT_obs[:,:-1], wind_obs=wind_obs[:,:-1], temp_obs=temp_obs[:,:-1], 
         RH_obs=RH_obs[:,:-1], X_obs=X_obs, canyon_orient_deg_obs=canyon_orient_deg_obs,
         tile_number_obs=tile_number_obs, lat_obs=lat_obs, lon_obs=lon_obs,
         H_obs=H_obs, W_obs=W_obs)

