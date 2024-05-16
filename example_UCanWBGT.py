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
from parameters import (time, tzinfo, WBGT_model_choice, nref, gamma_choice, WBGT_equation_choice,
                        Twb_method, gamma, LAI, lat, lon, elevation, canyon_orient_deg, Z, H, W, X, tf,
                        tile_number, alb_grnd_tiles, alb_wall, emiss_grnd_tiles, emiss_wall,
                        emiss_g, a_g, d, a_SW, a_LW)

### Set Remaining Parameters ###

t = datetime.strptime(time,'%Y%m%dT%H%MZ')
dateandtime = datetime(year=t.year, month=t.month, day=t.day, hour=t.hour, minute=t.minute, second=t.second)
alb_grnd = alb_grnd_tiles[tile_number]
emiss_grnd = emiss_grnd_tiles[tile_number]

### UCanWBGT 0D example ###

# Set forcing variables
T = 35.
T_grnd = 35.
T_wall = 35.
RH = 75.
WS = 2.
Ld = 450.
Kd = 200.
Id = 600.
P = 101325.0

# calculate WBGT
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
print("\n0D:")
print(", ".join([f"Twb: {Twb}", f"solar_zen_deg: {solar_zen_deg}", f"solar_azi_deg: {solar_azi_deg}", f"canyon_azi_deg: {canyon_azi_deg}", f"Fs: {Fs}", f"Fr: {Fr}", f"Fw: {Fw}", f"Fsr: {Fsr}", f"Frs: {Frs}", f"Fww: {Fww}", f"Fwr: {Fwr}", f"Fws: {Fws}", f"Frw: {Frw}", f"Fsw: {Fsw}", f"fr: {fr}", f"fw: {fw}", f"Fpr: {Fpr}", f"Fpw: {Fpw}", f"Fprw1: {Fprw1}", f"Fprw2: {Fprw2}", f"Fpw1r: {Fpw1r}", f"Fpw1w2: {Fpw1w2}", f"Sr: {Sr}", f"Sw: {Sw}", f"K: {K}", f"Ks: {Ks}", f"Kr: {Kr}", f"Kw: {Kw}", f"I: {I}", f"L: {L}", f"MRT: {MRT}", f"Tg: {Tg}", f"WBGT: {WBGT}"]))

### UCanWBGT 1D example ###

lon = np.array([lon,lon])
lat = np.array([lat,lat])

# calculate WBGT
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
print("\n1D:")
print(", ".join([f"Twb: {Twb}", f"solar_zen_deg: {solar_zen_deg}", f"solar_azi_deg: {solar_azi_deg}", f"canyon_azi_deg: {canyon_azi_deg}", f"Fs: {Fs}", f"Fr: {Fr}", f"Fw: {Fw}", f"Fsr: {Fsr}", f"Frs: {Frs}", f"Fww: {Fww}", f"Fwr: {Fwr}", f"Fws: {Fws}", f"Frw: {Frw}", f"Fsw: {Fsw}", f"fr: {fr}", f"fw: {fw}", f"Fpr: {Fpr}", f"Fpw: {Fpw}", f"Fprw1: {Fprw1}", f"Fprw2: {Fprw2}", f"Fpw1r: {Fpw1r}", f"Fpw1w2: {Fpw1w2}", f"Sr: {Sr}", f"Sw: {Sw}", f"K: {K}", f"Ks: {Ks}", f"Kr: {Kr}", f"Kw: {Kw}", f"I: {I}", f"L: {L}", f"MRT: {MRT}", f"Tg: {Tg}", f"WBGT: {WBGT}"]))


### UCanWBGT 2D example ###

print("\n2D plot:")

# Overwrite lat, lon, elevation
num_points = 20
dlat = 180./(num_points+1)
min_lat = -90 + dlat/2.
max_lat = 90 - dlat/2.
latitude = np.linspace(min_lat, max_lat, num_points)
dlon = 360./(num_points+1)
min_lon = -180 + dlon/2.
max_lon = 180 - dlon/2.
longitude = np.linspace(min_lon, max_lon, num_points)
lon, lat = np.meshgrid(longitude, latitude)
elevation = UCanWBGT.adjust_array_shape([elevation], lon)[0]

# Calculate zenith angle to make radiation forcing variables semi-plausible
zen, cazi, solar_zen_deg, solar_azi_deg, canyon_azi_deg = \
    UCanWBGT.zen_and_cazi(lat,lon,elevation,dateandtime,canyon_orient_deg,tzinfo)

# Set forcing variables
rows = np.arange(num_points)
cols = np.arange(num_points)
row_values = np.linspace(25, 45, num_points)
col_values = np.linspace(25, 45, num_points)
T = np.zeros((num_points, num_points))
for i, row in enumerate(rows):
    for j, col in enumerate(cols):
        T[i, j] = (row_values[i] + col_values[j]) / 2
T_grnd = np.copy(T)
T_wall = np.copy(T)

RH = np.zeros((num_points, num_points))
RH[:, :] = 20.
RH[:, ::2] = 90.

WS = np.zeros((num_points, num_points))
WS[:, :] = 0.5
WS[::2, :] = 5.

Ld = 450.
zen[zen>np.pi/2] = np.pi/2
cos_zen = np.cos(zen)
Kd = 200.*cos_zen
Id = 800.*cos_zen

P = 101325.0

tf = np.zeros((num_points, num_points)) + tf
tf[::3,::3] = 0.

H = np.zeros((num_points, num_points)) + H
H[::2,::2] = 1.

# calculate WBGT
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

# plot: T, RH, WS, Twb, MRT, Tg, solar_zen_deg, canyon_azi_deg, WBGT

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
extent = [np.min(lon)-dlon/2, np.max(lon)+dlon/2, np.min(lat)-dlat/2, np.max(lat)+dlat/2]
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
ax0.set_facecolor(color='gray')
c0 = ax0.pcolormesh(lon, lat, T,
                   cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.nanmin(RH)
vmax = np.nanmax(RH)
cmap = plt.cm.Blues
ax1.set_facecolor(color='gray')
c1 = ax1.pcolormesh(lon, lat, RH,
                   cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.nanmin(WS)
vmax = np.nanmax(WS)
cmap = plt.cm.jet
ax2.set_facecolor(color='gray')
c2 = ax2.pcolormesh(lon, lat, WS,
                   cmap=cmap, transform=ccrs.PlateCarree())
vmin = np.nanmin(Twb)
vmax = np.nanmax(Twb)
cmap = plt.cm.hot_r
ax3.set_facecolor(color='gray')
c3 = ax3.pcolormesh(lon, lat, Twb,
                   cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.nanmin(MRT)
vmax = np.nanmax(MRT)
cmap = plt.cm.hot_r
ax4.set_facecolor(color='gray')
c4 = ax4.pcolormesh(lon, lat, MRT,
                   cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
vmin = np.nanmin(Tg)
vmax = np.nanmax(Tg)
cmap = plt.cm.hot_r
ax5.set_facecolor(color='gray')
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
ax8.set_facecolor(color='gray')
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

plt.savefig(f"./figs/example_{time}.png", dpi=300, bbox_inches='tight')
plt.show()

