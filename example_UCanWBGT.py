import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import iris
import mo_pack
import pvlib
import sys
sys.path.append("/home/h04/lblunn/Documents/Projects/UCanWBGT")  # Add the directory containing UCanWBGT.py to the Python path
import UCanWBGT
import importlib
importlib.reload(UCanWBGT)
from constants import sboltz
from parameters import (tzinfo, WBGT_model_choice, geometry_choice, nref, gamma_choice, WBGT_equation_choice,
                        Twb_method, gamma, LAI, elevation, canyon_orient_deg, Z, X,
                        tile_number, alb_grnd_tiles, alb_wall, emiss_grnd_tiles, emiss_wall,
                        emiss_g, a_g, d, a_SW, a_LW)

### load data ###

# UPD
UPD = iris.load('/scratch/lblunn/heat_stress_wkeat/bb189a.pd20600613.pp')
print("UPD:\n",UPD)
T = UPD.extract(iris.AttributeConstraint(STASH='m01s03i236'))[0]
print("T:\n",T)
q = UPD.extract(iris.AttributeConstraint(STASH='m01s03i237'))[0]
print("q:\n",q)
Ld = UPD.extract(iris.AttributeConstraint(STASH='m01s02i207'))[0]
print("Ld:\n",Ld)
KId = UPD.extract(iris.AttributeConstraint(STASH='m01s01i235'))[0]
print("KId:\n",KId)
P = UPD.extract(iris.AttributeConstraint(STASH='m01s00i409'))[0]
print("P:\n",P)

# UPF
UPF = iris.load('/scratch/lblunn/heat_stress_wkeat/bb189a.pf20600613.pp')
print("UPF:\n",UPF)
WS_X = UPF.extract(iris.AttributeConstraint(STASH='m01s00i002'))[0]
print("WS_X:\n",WS_X)
print("WS_X height:\n",WS_X.coord('level_height'))
WS_Y = UPF.extract(iris.AttributeConstraint(STASH='m01s00i003'))[0]
print("WS_Y:\n",WS_Y)

# UPU
UPU = iris.load('/scratch/lblunn/heat_stress_wkeat/bb189a.pu20600613.pp')
Lu = UPU.extract(iris.AttributeConstraint(STASH='m01s02i217'))[0]
print("Lu:\n",Lu)
print("Lu height:\n",Lu.coord('level_height'))

# UPE
UPE = iris.load('/scratch/lblunn/heat_stress_wkeat/bb189a.pe20600613.pp')
print("UPE:\n",UPE)

### Will specific data processing ###

# constrain on time
time = '20600613T1500Z' # the time we are calculating WBGT at 
t = datetime.strptime(time,'%Y%m%dT%H%MZ')
dateandtime = datetime(year=t.year, month=t.month, day=t.day, hour=t.hour, minute=t.minute)
target_time = iris.time.PartialDateTime(year=t.year, month=t.month, day=t.day, hour=t.hour, minute=t.minute)
target_time_m1p5 = iris.time.PartialDateTime(year=t.year, month=t.month, day=t.day, hour=t.hour-2, minute=t.minute+30)
target_time_p1p5 = iris.time.PartialDateTime(year=t.year, month=t.month, day=t.day, hour=t.hour+1, minute=t.minute+30)

constraint = iris.Constraint(time=lambda cell: cell.point == target_time)
T = T.extract(constraint) # these variables are hourly instantaneous
q = q.extract(constraint) # these variables are hourly instantaneous

def interpolate_to_hour(cube,target_time_m1p5,target_time_p1p5):

    m_constraint = iris.Constraint(time=lambda cell: cell.point == target_time_m1p5)
    p_constraint = iris.Constraint(time=lambda cell: cell.point == target_time_p1p5)
    m_time_pnt = cube.extract(m_constraint).coord('time').points
    p_time_pnt = cube.extract(p_constraint).coord('time').points
    cube = cube.interpolate([('time', (m_time_pnt + p_time_pnt)/2)], iris.analysis.Linear())[0,:,:]

    return cube

# these variables are 3 hour average (so approximate on the hour)
Ld = interpolate_to_hour(Ld,target_time_m1p5,target_time_p1p5)
KId = interpolate_to_hour(KId,target_time_m1p5,target_time_p1p5)
WS_X = interpolate_to_hour(WS_X,target_time_m1p5,target_time_p1p5)
WS_Y = interpolate_to_hour(WS_Y,target_time_m1p5,target_time_p1p5)
P = interpolate_to_hour(P,target_time_m1p5,target_time_p1p5)
Lu = interpolate_to_hour(Lu,target_time_m1p5,target_time_p1p5)

# calculate solar angles at one lat/lon in the centre of the UK (Birmingham of course) 
# since zenith angle is required to approximate diffuse/direct partitioning
# (note solar angle calculations are done on the lat/lon grid within UCanWBGT.main)
zen, cazi, solar_zen_deg, solar_azi_deg, canyon_azi_deg = \
    UCanWBGT.zen_and_cazi(52.5,-1.9,elevation,dateandtime,canyon_orient_deg,tzinfo)

# approximate diffuse and direct partitioning (choose clear sky)
# https://link.springer.com/book/10.1007/978-1-4612-1626-1, Sect. 11.2, Eq. 11.13
# note: if cloud cover is available then where cloud=True the shortwave could be set to 100% diffuse 
Kd = KId.copy()
Id = KId.copy()
tau = 0.65
m = 1/np.cos(zen)
Kd.data = 0.3*(1-tau**m)*np.cos(zen)*KId.data
Id.data = KId.data - Kd.data

# WS_X and WS_Y have different grids, interpolate onto T grid
WS_X_2p5m = WS_X[0,:,:].regrid(T, iris.analysis.Linear())
WS_Y_2p5m = WS_Y[0,:,:].regrid(T, iris.analysis.Linear())
# approximate 1.5 m wind speed by linear interpolation
WS_2p5m = WS_X_2p5m.copy()
WS_2p5m.data = (WS_X_2p5m.data**2 + WS_Y_2p5m.data**2)**0.5
WS = WS_2p5m.copy()
WS.data = WS_2p5m.data * (1.5 - 0.) / (2.5 - 0.)

# dummy surface description ancillaries
# *** at locations where urban fraction is zero the geometry_choice should be flat
# or functionality to handle Z > H should be added -- this needs doing in UCanWBGT.py ***
lc_fracs = [0.05,0.,0.3,0.,0.,0.,0.,0.,0.4,0.25]
H = 10. # *** in the current code when geometry_choice = canyon then Z < H and H > 0 must be satisfied ***
W = 20.
alb_grnd = alb_grnd_tiles[tile_number]
emiss_grnd = emiss_grnd_tiles[tile_number]

# Surface temperature of grid box (calculate using grid box average longwave up)
# Use to approximate the surface temperature in the viewing fraction of the black globe
emiss_canyon = 0.95 # approx. bulk emissivity of canyon tile based on Fig. 4 Porson et al. (2010), due to geometry it is not the same as the canyon ground emissivity 
emiss_tiles = np.copy(emiss_grnd_tiles)
emiss_tiles[-2] = emiss_canyon
emiss_gb = np.sum(emiss_tiles*lc_fracs)/np.sum(lc_fracs) # land cover fraction weighted average emissivity
print("emiss_gb:",emiss_gb)
T_grnd = Lu[0,:,:].copy() # Lu at bottom model level
T_grnd.data = (Lu[0,:,:].data/(emiss_gb*sboltz))**0.25
T_wall = T_grnd.copy()

### Convert to arrays and do any unit conversions ###

grid_longitude = T.coord('grid_longitude').points
grid_latitude = T.coord('grid_latitude').points
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_longitude, grid_latitude)
pole_lon = T.coord('grid_longitude').coord_system.grid_north_pole_longitude
pole_lat = T.coord('grid_latitude').coord_system.grid_north_pole_latitude
lon, lat = iris.analysis.cartography.unrotate_pole(grid_lon_mesh, grid_lat_mesh, pole_lon, pole_lat)

T = T.data - 273.15 # degC
q = q.data
Ld = Ld.data
Id = Id.data
Kd = Kd.data
P = P.data
WS = WS.data
T_grnd = T_grnd.data - 273.15 # degC
T_wall = T_wall.data - 273.15 # degC

### calculate WBGT ###

variables = {
    "T": T,
    "T_grnd": T_grnd,
    "T_wall": T_wall,
    "q": q,
    "WS": WS,
    "P": P,
    "Ld": Ld,
    "Kd": Kd,
    "Id": Id,
    "gamma": gamma,
    "LAI": LAI,
    "H": H,
    "W": W,
    "Z": Z,
    "X": X,
    "canyon_orient_deg": canyon_orient_deg,
    "a_SW": a_SW,
    "a_LW": a_LW,
    "alb_grnd": alb_grnd,
    "alb_wall": alb_wall,
    "emiss_grnd": emiss_grnd,
    "emiss_wall": emiss_wall,
    "emiss_g": emiss_g,
    "a_g": a_g,
    "d": d,
    "lat": lat,
    "lon": lon,
    "elevation": elevation
}
for name, var in variables.items():
    print(f"{name}: {np.shape(var)}")

Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
    fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s, Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT \
        = UCanWBGT.main(
        T=T, T_grnd=T_grnd, T_wall=T_wall, RH=None, q=q, WS=WS, P=P, Ld=Ld, Kd=Kd, Id=Id,\
        gamma=gamma, LAI=LAI,\
        H=H, W=W, Z=Z, X=X, canyon_orient_deg=canyon_orient_deg,\
        a_SW=a_SW, a_LW=a_LW,\
        alb_grnd=alb_grnd, alb_wall=alb_wall, emiss_grnd=emiss_grnd, emiss_wall=emiss_wall,\
        emiss_g=emiss_g, a_g=a_g, d=d,\
        lat=lat, lon=lon, elevation=elevation, dateandtime=dateandtime, tzinfo=tzinfo,\
        WBGT_model_choice=WBGT_model_choice, geometry_choice=geometry_choice, nref=nref,\
        gamma_choice=gamma_choice, WBGT_equation_choice=WBGT_equation_choice, Twb_method=Twb_method 
        )

# plot: T, SH, WS, Twb, MRT, Tg, solar_zen_deg, canyon_azi_deg, WBGT

fig = plt.figure(figsize=(8.0,3.2))
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
#extent = [np.min(lon), np.max(lon), np.min(lat), np.max(lat)] # UKCP extent
extent = [-1.5, 1.5, 50.5, 52.] # SE extent
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
vmin = np.min(T)
vmax = np.max(T)
c0 = ax0.pcolormesh(lon, lat, T,
                   cmap='hot_r', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.min(q)
vmax = np.max(q)
c1 = ax1.pcolormesh(lon, lat, q,
                   cmap='gray_r', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.min(WS)
vmax = np.max(WS)
c2 = ax2.pcolormesh(lon, lat, WS,
                   cmap='gray', transform=ccrs.PlateCarree())
vmin = np.min(Twb)
vmax = np.max(Twb)
c3 = ax3.pcolormesh(lon, lat, Twb,
                   cmap='hot_r', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.min(MRT)
vmax = np.max(MRT)
c4 = ax4.pcolormesh(lon, lat, MRT,
                   cmap='hot_r', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.min(Tg)
vmax = np.max(Tg)
c5 = ax5.pcolormesh(lon, lat, Tg,
                   cmap='hot_r', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
c6 = ax6.contourf(lon, lat, solar_zen_deg,
                   cmap='gray', levels=np.linspace(0., 180., 20),
                   transform=ccrs.PlateCarree())
c7 = ax7.contourf(lon, lat, canyon_azi_deg,
                   cmap='hsv', levels=np.linspace(0., 360., 20),
                   transform=ccrs.PlateCarree())
vmin = np.min(WBGT)
vmax = np.max(WBGT)
c8 = ax8.pcolormesh(lon, lat, WBGT,
                   cmap='hot_r', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

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
cbar1.set_label(r"$q$ (kg kg$^{-1}$)", rotation=90, loc="center")
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

plt.savefig(f"./figs/example_{time}.png", dpi=300, bbox_inches='tight')
plt.show()
