#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import iris
import pvlib
import sys
sys.path.append("/home/h04/lblunn/Documents/Projects/UCanWBGT")  # Add the directory containing UCanWBGT.py to the Python path
import UCanWBGT
import importlib
importlib.reload(UCanWBGT)
from parameters import (tzinfo, WBGT_model_choice, nref, gamma_choice, WBGT_equation_choice,
                        Twb_method, gamma, elevation, Z, 
                        alb_grnd_tiles, alb_wall, emiss_grnd_tiles, emiss_wall,
                        emiss_g, a_g, d, a_SW, a_LW)

# notes on updates that I have realised need to be made to the main branch through doing this work:
# - add "if isinstance(array_or_float, int): array_or_float = float(array_or_float)" in adjust_array_shape
# - add "print(f"!!! {100 * num_H_less_than_Z_and_gn_0_true / num_H_less_than_Z_true}% of Z < H points had H > 0 !!!")"
# - add "for arr in enumerate(array_or_float_list): array_or_float_list[i][mask] = np.nan"
# - recommend 8 not 9 for urban

#%%

############################################## Data prep.

### Read in data ###

cubel = iris.load('/data/users/lblunn/Heat_Stress/Vinod_RNS_Data/paris_20230906T1200Z_48hrfcst.nc')
print("cubel original:\n",cubel)

### Constrain and interpolate data ###

def spatial_constraint(cube,max_longitude,min_longitude,max_latitude,min_latitude):

    longitude_constraint = iris.Constraint(longitude=lambda cell: min_longitude <= cell <= max_longitude)
    latitude_constraint = iris.Constraint(latitude=lambda cell: min_latitude <= cell <= max_latitude)
    constrained_cube = cube.extract(longitude_constraint & latitude_constraint)

    return constrained_cube

clon = 2.34
clat = 48.87
delta = 0.3
max_longitude = clon + delta
min_longitude = clon - delta
max_latitude = clat + delta
min_latitude = clat - delta

cubel = spatial_constraint(cubel,max_longitude,min_longitude,max_latitude,min_latitude)
print("cubel spatial constraint:\n",cubel)

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

start_time_dt = datetime.strptime('20230907T0000Z','%Y%m%dT%H%MZ')
end_time_dt = datetime.strptime('20230908T0000Z','%Y%m%dT%H%MZ')
start_time = iris.time.PartialDateTime(year=start_time_dt.year, month=start_time_dt.month, \
    day=start_time_dt.day, hour=start_time_dt.hour, minute=start_time_dt.minute)
end_time = iris.time.PartialDateTime(year=end_time_dt.year, month=end_time_dt.month, \
    day=end_time_dt.day, hour=end_time_dt.hour, minute=end_time_dt.minute)
cubel = interpolate_times(cubel,start_time,end_time,dt=0.5)
print("cubel 24 hours at half hourly interval:\n",cubel)

### extract cubes from cube list ###

T_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s03i236'))[0]
print("T_cube:\n",T_cube)
RH_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s03i245'))[0]
print("RH_cube:\n",RH_cube)
Ld_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s02i207'))[0]
print("Ld_cube:\n",Ld_cube)
Id_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s01i215'))[0]
print("Id_cube:\n",Id_cube)
Kd_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s01i216'))[0]
print("Kd_cube:\n",Kd_cube)
P_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i409'))[0]
print("P_cube:\n",P_cube)
WS_X_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i002'))[0] # data is 1 m not 1.5 m
print("WS_X_cube:\n",WS_X_cube)
WS_Y_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i003'))[0] # data is 1 m not 1.5 m
print("WS_Y_cube:\n",WS_Y_cube)
T_surf_tiles_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s03i316'))[0]
print("T_surf_tiles_cube:\n",T_surf_tiles_cube)
H_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i494'))[0]
print("H_cube:\n",H_cube)
HWR_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i495'))[0]
print("HWR_cube:\n",HWR_cube)
LAI_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i217'))[0]
print("LAI_cube:\n",LAI_cube)
tf_cube = cubel.extract(iris.AttributeConstraint(STASH='m01s00i216'))[0]
print("tf_cube:\n",tf_cube)

### Data Calculations ###

# wind speed
WS_cube = WS_X_cube.copy()
WS_cube.data = (WS_X_cube.data**2 + WS_Y_cube.data**2)**0.5

# canyon width: HWR = H / W 
W_cube = HWR_cube.copy()
W_cube.data = H_cube.data / HWR_cube.data

# lat/lon
longitude = T_cube.coord('longitude').points
latitude = T_cube.coord('latitude').points
lon, lat = np.meshgrid(longitude, latitude)
print("lon:",np.shape(lon))
print("lat:",np.shape(lat))

#%%

############################################## UCanWBGT maps

### Settings for UCanWBGT map calculations ###

# time UTC to map
time = "20230907T0600Z"
time_dt = datetime.strptime(time,'%Y%m%dT%H%MZ')
dateandtime = datetime(year=time_dt.year, month=time_dt.month, day=time_dt.day, hour=time_dt.hour, minute=time_dt.minute, second=time_dt.second)

# which tile to use (note: if tiles 8 or 9 are selected the calculation will be for the urban area using tiles 8 and 9). 0: Broad Leaf Tree, 1: Needle Leaf Tree, 2: C3 Grass, 3: C4 Grass, 4: Shrub, 5: Inland Water, 6: Bare Soil, 7: Ice, 8: Urban Canyon, 9: Urban Roof
tile_number = 8 

# black globe position where 0.0 is the centre and positive/negative values are to right/left of centre (m)
X = 0.0

# the horizontal angle measured clockwise from north to a line running parallel to the alignment of the street canyon
canyon_orient_deg = 0.0 

# ground albedo and emissvity
alb_grnd = alb_grnd_tiles[tile_number]
emiss_grnd = emiss_grnd_tiles[tile_number]

### Format Data for UCanWBGT map calculations ###

tolerance_minutes = 1
lower_bound_time = time_dt - timedelta(minutes=tolerance_minutes)
upper_bound_time = time_dt + timedelta(minutes=tolerance_minutes)
time_constraint = iris.Constraint(time=lambda cell: lower_bound_time <= cell.point <= upper_bound_time)

T = T_cube.extract(time_constraint).data - 273.15
print("T:",np.shape(T))
RH = RH_cube.extract(time_constraint).data
print("RH:",np.shape(RH))
Ld = Ld_cube.extract(time_constraint).data
print("Ld:",np.shape(Ld))
Id = Id_cube.extract(time_constraint).data
print("Id:",np.shape(Id))
Kd = Kd_cube.extract(time_constraint).data
print("Kd:",np.shape(Kd))
P = P_cube.extract(time_constraint).data
print("P:",np.shape(P))
WS = WS_cube.extract(time_constraint).data
print("WS:",np.shape(WS))
T_wall = T_surf_tiles_cube[tile_number,:,:].extract(time_constraint).data - 273.15
print("T_wall:",np.shape(T_wall))
T_grnd = np.copy(T_wall)
print("T_grnd:",np.shape(T_grnd))
H = H_cube.data
print("H:",np.shape(H))
W = W_cube.data
print("W:",np.shape(W))
if tile_number > 4:
    LAI = np.zeros(np.shape(T))
else:
    LAI = LAI_cube[tile_number,:,:].data
print("LAI:",np.shape(LAI))
tf = tf_cube[tile_number,:,:].data
print("tf:",np.shape(tf))

# make quick histograms of an array
plt.hist(H, bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.grid(True)
plt.savefig(f"./figs/H_histogram.png", dpi=300, bbox_inches='tight')

### UCanWBGT map calculations ###

print("\nCalculating UCanWBGT map ...")
Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
    fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s, Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT \
        = UCanWBGT.main(
        T=T, T_grnd=T_grnd, T_wall=T_wall, RH=RH, q=None, WS=WS, P=P, Ld=Ld, Kd=Kd, Id=Id,\
        gamma=gamma, LAI=LAI,\
        H=H, W=W, Z=Z, X=X, tile_number=tile_number, tf=tf, canyon_orient_deg=canyon_orient_deg,\
        a_SW=a_SW, a_LW=a_LW,\
        alb_grnd=alb_grnd, alb_wall=alb_wall, emiss_grnd=emiss_grnd, emiss_wall=emiss_wall,\
        emiss_g=emiss_g, a_g=a_g, d=d,\
        lat=lat, lon=lon, elevation=elevation, dateandtime=dateandtime, tzinfo=tzinfo,\
        WBGT_model_choice=WBGT_model_choice, nref=nref,\
        gamma_choice=gamma_choice, WBGT_equation_choice=WBGT_equation_choice, Twb_method=Twb_method 
        )
print("Calculation of UCanWBGT map finished")

### plot: T, RH, WS, Twb, MRT, Tg, solar_zen_deg, canyon_azi_deg, WBGT ###

fig = plt.figure(figsize=(6.0,3.0))
matplotlib.rcParams.update({'font.size': 8})

# add axes
left1 = 0.05
left2 = 0.37
left3 = 0.69
width = 0.21
bottom1 = 0.06
bottom2 = 0.38
bottom3 = 0.70
height = 0.26 
ax0 = fig.add_axes([left1, bottom3, width, height], projection=ccrs.PlateCarree()) #[left, bottom, width, height]
ax1 = fig.add_axes([left2, bottom3, width, height], projection=ccrs.PlateCarree()) #[left, bottom, width, height]
ax2 = fig.add_axes([left3, bottom3, width, height], projection=ccrs.PlateCarree()) #[left, bottom, width, height]
ax3 = fig.add_axes([left1, bottom2, width, height], projection=ccrs.PlateCarree()) #[left, bottom, width, height]
ax4 = fig.add_axes([left2, bottom2, width, height], projection=ccrs.PlateCarree()) #[left, bottom, width, height]
ax5 = fig.add_axes([left3, bottom2, width, height], projection=ccrs.PlateCarree()) #[left, bottom, width, height]
ax6 = fig.add_axes([left1, bottom1, width, height], projection=ccrs.PlateCarree()) #[left, bottom, width, height]
ax7 = fig.add_axes([left2, bottom1, width, height], projection=ccrs.PlateCarree()) #[left, bottom, width, height]
ax8 = fig.add_axes([left3, bottom1, width, height], projection=ccrs.PlateCarree()) #[left, bottom, width, height]

# axes lat/lon extent
extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
ax0.set_extent(extent, crs=ccrs.PlateCarree())
ax1.set_extent(extent, crs=ccrs.PlateCarree())
ax2.set_extent(extent, crs=ccrs.PlateCarree())
ax3.set_extent(extent, crs=ccrs.PlateCarree())
ax4.set_extent(extent, crs=ccrs.PlateCarree())
ax5.set_extent(extent, crs=ccrs.PlateCarree())
ax6.set_extent(extent, crs=ccrs.PlateCarree())
ax7.set_extent(extent, crs=ccrs.PlateCarree())
ax8.set_extent(extent, crs=ccrs.PlateCarree())

# contours
vmin = np.nanmin(T)
vmax = np.nanmax(T)
cmap = plt.cm.hot_r
cmap.set_bad(color='gray')
c0 = ax0.pcolormesh(lon, lat, T,
                   cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.nanmin(RH)
vmax = np.nanmax(RH)
cmap = plt.cm.Blues
cmap.set_bad(color='gray')
c1 = ax1.pcolormesh(lon, lat, RH,
                   cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.nanmin(WS)
vmax = np.nanmax(WS)
cmap = plt.cm.jet
cmap.set_bad(color='gray')
c2 = ax2.pcolormesh(lon, lat, WS,
                   cmap=cmap, transform=ccrs.PlateCarree())
vmin = np.nanmin(Twb)
vmax = np.nanmax(Twb)
cmap = plt.cm.hot_r
cmap.set_bad(color='gray')
c3 = ax3.pcolormesh(lon, lat, Twb,
                   cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.nanmin(MRT)
vmax = np.nanmax(MRT)
cmap = plt.cm.hot_r
cmap.set_bad(color='gray')
c4 = ax4.pcolormesh(lon, lat, MRT,
                   cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.nanmin(Tg)
vmax = np.nanmax(Tg)
cmap = plt.cm.hot_r
cmap.set_bad(color='gray')
c5 = ax5.pcolormesh(lon, lat, Tg,
                   cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
cmap = plt.cm.seismic_r
ax6.set_facecolor(color='gray')
c6 = ax6.contourf(lon, lat, solar_zen_deg,
                 cmap=cmap, levels=np.linspace(0., 180., 20), transform=ccrs.PlateCarree())
cmap = plt.cm.hsv
ax7.set_facecolor(color='gray')
c7 = ax7.contourf(lon, lat, canyon_azi_deg,
                 cmap=cmap, levels=np.linspace(0., 360., 20), transform=ccrs.PlateCarree())
vmin = np.nanmin(WBGT)
vmax = np.nanmax(WBGT)
cmap = plt.cm.hot_r
cmap.set_bad(color='gray')
c8 = ax8.pcolormesh(lon, lat, WBGT,
                   cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

# gridlines
gl0 = ax0.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl0.top_labels = False
gl0.bottom_labels = False
gl0.right_labels = False
gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl1.top_labels = False
gl1.bottom_labels = False
gl1.left_labels = False
gl1.right_labels = False
gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl2.top_labels = False
gl2.bottom_labels = False
gl2.left_labels = False
gl2.right_labels = False
gl3 = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl3.top_labels = False
gl3.bottom_labels = False
gl3.right_labels = False
gl4 = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl4.top_labels = False
gl4.bottom_labels = False
gl4.left_labels = False
gl4.right_labels = False
gl5 = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl5.top_labels = False
gl5.bottom_labels = False
gl5.left_labels = False
gl5.right_labels = False
gl6 = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl6.top_labels = False
gl6.right_labels = False
gl7 = ax7.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl7.top_labels = False
gl7.left_labels = False
gl7.right_labels = False
gl8 = ax8.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl8.top_labels = False
gl8.left_labels = False
gl8.right_labels = False

# colour bars
cwidth = 0.025
cdelta = 0.005
#0
cax0 = fig.add_axes([left1+width+cdelta, bottom3, cwidth+cdelta, height])
cbar0 = fig.colorbar(c0, cax=cax0, orientation="vertical")
cbar0.set_label(r"$T$ ($^\circ$C)", rotation=90, loc="center")
#1
cax1 = fig.add_axes([left2+width+cdelta, bottom3, cwidth+cdelta, height])
cbar1 = fig.colorbar(c1, cax=cax1, orientation="vertical")
cbar1.set_label(r"$RH$ ($\%$)", rotation=90, loc="center")
#2
cax2 = fig.add_axes([left3+width+cdelta, bottom3, cwidth+cdelta, height])
cbar2 = fig.colorbar(c2, cax=cax2, orientation="vertical")
cbar2.set_label(r"$WS$ (m s$^{-1}$)", rotation=90, loc="center")
#3
cax3 = fig.add_axes([left1+width+cdelta, bottom2, cwidth+cdelta, height])
cbar3 = fig.colorbar(c3, cax=cax3, orientation="vertical")
cbar3.set_label(r"$T_{wb}$ ($^\circ$C)", rotation=90, loc="center")
#4
cax4 = fig.add_axes([left2+width+cdelta, bottom2, cwidth+cdelta, height])
cbar4 = fig.colorbar(c4, cax=cax4, orientation="vertical")
cbar4.set_label(r"$T_r$ ($^\circ$C)", rotation=90, loc="center")
#5
cax5 = fig.add_axes([left3+width+cdelta, bottom2, cwidth+cdelta, height])
cbar5 = fig.colorbar(c5, cax=cax5, orientation="vertical")
cbar5.set_label(r"$T_g$ ($^\circ$C)", rotation=90, loc="center")
#6
cax6 = fig.add_axes([left1+width+cdelta, bottom1, cwidth+cdelta, height])
cbar6 = fig.colorbar(c6, cax=cax6, orientation="vertical")
cbar6.set_label(r"$\theta_0$ ($^\circ$)", rotation=90, loc="center")
cbar6.set_ticks([0,45,90,135,180])
#7
cax7 = fig.add_axes([left2+width+cdelta, bottom1, cwidth+cdelta, height])
cbar7 = fig.colorbar(c7, cax=cax7, orientation="vertical")
cbar7.set_label(r"$\phi_0$ ($^\circ$)", rotation=90, loc="center")
cbar7.set_ticks([0,90,180,270,360])
#8
cax8 = fig.add_axes([left3+width+cdelta, bottom1, cwidth+cdelta, height])
cbar8 = fig.colorbar(c8, cax=cax8, orientation="vertical")
cbar8.set_label(r"$WBGT$ ($^\circ$C)", rotation=90, loc="center")

plt.savefig(f"./figs/map_tile_number_{tile_number}_{time}.png", dpi=300, bbox_inches='tight')
plt.show()

#%%

############################################## UCanWBGT timeseries at sites

### Settings for UCanWBGT timeseries calculations ###

# Create a dictionary with site names
data = {
    'site1': {},
    'site2': {}
}
# Define the keys for the inner dictionaries
keys = ['Twb', 'solar_zen_deg', 'solar_azi_deg', 'canyon_azi_deg', 'Fs', 'Fr', 'Fw', 'Fsr', 'Frs', 'Fww', 'Fwr', 'Fws', 'Frw', 'Fsw',
        'fr', 'fw', 'Fpr', 'Fpw', 'Fprw1', 'Fprw2', 'Fprs', 'Fpw1r', 'Fpw1w2', 'Fpw1s', 'Sr', 'Sw', 'K', 'Ks', 'Kr', 'Kw', 'I', 'L',
        'MRT', 'Tg', 'WBGT']
# Initialize the inner dictionaries with empty lists for each key
for site_data in data.values():
    for key in keys:
        site_data[key] = []

sites = ['site1','site2']
X_sites = [0,0]
canyon_orient_deg_sites = [0,0]
tile_number_sites = [8,2]
lat_sites = [48.875,48.831] 
lon_sites = [2.345,2.442]
delta = 0.0008999999999996788/2. # grid length (in deg) divide 2

current_time = start_time_dt
times = []
while current_time < end_time_dt:
    times.append(current_time.strftime('%Y%m%dT%H%MZ'))
    current_time += timedelta(minutes=180)

### UCanWBGT timeseries calculations ###

print("\nCalculating UCanWBGT timeseries ...")
for i, site in enumerate(sites):
    print("\nsite:",site)
    
    # site specific parameters
    tile_number = tile_number_sites[i] 
    X = X_sites[i]
    canyon_orient_deg = canyon_orient_deg_sites[i]
    alb_grnd = alb_grnd_tiles[tile_number]
    emiss_grnd = emiss_grnd_tiles[tile_number]
    lat_site = lat_sites[i]
    lon_site = lon_sites[i]

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

    # lat/lon
    longitude = T_cube_site.coord('longitude').points
    latitude = T_cube_site.coord('latitude').points
    lon, lat = np.meshgrid(longitude, latitude)
    print("lon:",lon)
    print("lat:",lat)

    for j, time in enumerate(times):
        print("time:",time)
        
        # convert time to different types
        time_dt = datetime.strptime(time,'%Y%m%dT%H%MZ')
        dateandtime = datetime(year=time_dt.year, month=time_dt.month, day=time_dt.day, hour=time_dt.hour, minute=time_dt.minute, second=time_dt.second)

        # constrain on time and convert to data
        tolerance_minutes = 1
        lower_bound_time = time_dt - timedelta(minutes=tolerance_minutes)
        upper_bound_time = time_dt + timedelta(minutes=tolerance_minutes)
        time_constraint = iris.Constraint(time=lambda cell: lower_bound_time <= cell.point <= upper_bound_time)
        T = T_cube_site.extract(time_constraint).data - 273.15
        RH = RH_cube_site.extract(time_constraint).data
        Ld = Ld_cube_site.extract(time_constraint).data
        Id = Id_cube_site.extract(time_constraint).data
        Kd = Kd_cube_site.extract(time_constraint).data
        P = P_cube_site.extract(time_constraint).data
        WS = WS_cube_site.extract(time_constraint).data
        T_wall = T_surf_tiles_cube_site[tile_number].extract(time_constraint).data - 273.15
        T_grnd = np.copy(T_wall)
        H = H_cube_site.data
        W = W_cube_site.data
        if tile_number > 4:
            LAI = np.zeros(np.shape(T))
        else:
            LAI = LAI_cube_site[tile_number].data
        tf = tf_cube_site[tile_number].data
        print("tf:",tf)

        # calculate UCanWBGT
        variables = UCanWBGT.main(
                T=T, T_grnd=T_grnd, T_wall=T_wall, RH=RH, q=None, WS=WS, P=P, Ld=Ld, Kd=Kd, Id=Id,\
                gamma=gamma, LAI=LAI,\
                H=H, W=W, Z=Z, X=X, tile_number=tile_number, tf=tf, canyon_orient_deg=canyon_orient_deg,\
                a_SW=a_SW, a_LW=a_LW,\
                alb_grnd=alb_grnd, alb_wall=alb_wall, emiss_grnd=emiss_grnd, emiss_wall=emiss_wall,\
                emiss_g=emiss_g, a_g=a_g, d=d,\
                lat=lat, lon=lon, elevation=elevation, dateandtime=dateandtime, tzinfo=tzinfo,\
                WBGT_model_choice=WBGT_model_choice, nref=nref,\
                gamma_choice=gamma_choice, WBGT_equation_choice=WBGT_equation_choice, Twb_method=Twb_method 
                )
        keys = data[site].keys()
        for k, key in enumerate(keys):
            data[site][key].append(float(variables[k][0][0]))
print("Calculation of UCanWBGT timeseries finished")

#%%

### plot the WBGT at the sites ###

time_values = [datetime.strptime(time_str, '%Y%m%dT%H%MZ') for time_str in times]
WBGT_site1 = data['site1']['WBGT']
WBGT_site2 = data['site2']['WBGT']

plt.figure(figsize=(10, 6))
plt.plot(time_values, WBGT_site1, marker='o', color='b', label='site1')
plt.plot(time_values, WBGT_site2, marker='o', color='r', label='site2')
plt.xlabel('Time')
plt.ylabel(r'WBGT ($^\circ$C)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"./figs/timeseries.png", dpi=300, bbox_inches='tight')

#%%

#%%