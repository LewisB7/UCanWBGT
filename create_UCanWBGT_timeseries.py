"""
  CREATE_UCANWBGT_TIMESERIES: compiles the model and observations data into WBGT values
    and generates time series.


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
sys.path.append("/home/h05/joshonk/github/UCanWBGT/UCanWBGT")
#sys.path.append("/home/h04/joshonklblunn/Documents/Projects/UCanWBGT")  
    # Add the directory containing UCanWBGT.py to the Python path
import UCanWBGT
import importlib
importlib.reload(UCanWBGT)

#--- Import parameters from the "parameters.py" file.
from parameters import (tzinfo, WBGT_model_choice, nref, gamma_choice, WBGT_equation_choice,
                        Twb_method, gamma, elevation, Z, 
                        alb_grnd_tiles, alb_wall, emiss_grnd_tiles, emiss_wall,
                        emiss_g, a_g, d, a_SW, a_LW)

#--- Load in the processed model and observations data -- these are at the six urban sites
#    in Paris and half-hourly over the course of 07 Sep 2023.
fileroot = "/data/users/joshonk/data/U23/"
filename = "Paris_data_30min_at_sites.npz"
data = np.load((fileroot+filename))

#--- Unpack relevant model variables from the Numpy file object.
sites_obs = data["sites_obs"]
print("sites_obs:",np.shape(sites_obs))
timeaxis = data["timeaxis"]
print("timeaxis:",np.shape(timeaxis))

X_obs = data["X_obs"]
print("X_obs:",np.shape(X_obs))
canyon_orient_deg_obs = data["canyon_orient_deg_obs"]
print("phi:",np.shape(canyon_orient_deg_obs))
tile_number_obs = data["tile_number_obs"]
print("tile#:",np.shape(tile_number_obs))
lat_obs = data["lat_obs"]
print("lat_obs:",np.shape(lat_obs))
lon_obs = data["lon_obs"]
print("lon_obs:",np.shape(lon_obs))
H_obs = data["H_obs"]
print("H_obs:",np.shape(H_obs))
W_obs = data["W_obs"]
print("W_obs:",np.shape(W_obs))

T_obs = data["temp_obs"] 
print("T_obs:",np.shape(T_obs))
RH_obs = data["RH_obs"]
print("RH_obs:",np.shape(RH_obs))
Tg_obs = data["BGT_obs"]
print("Tg_obs:",np.shape(Tg_obs))
WS_obs = data["wind_obs"]
print("WS_obs:",np.shape(WS_obs))

T_site = data["T_site"] - 273.15
print("T:",np.shape(T_site))
RH_site = data["RH_site"]
print("RH:",np.shape(RH_site))
Ld_site = data["Ld_site"]
print("Ld:",np.shape(Ld_site))
Id_site = data["Id_site"]
print("Id:",np.shape(Id_site))
Kd_site = data["Kd_site"]
print("Kd:",np.shape(Kd_site))
P_site = data["P_site"]
print("P:",np.shape(P_site))
WS_site = data["WS_site"]
print("WS:",np.shape(WS_site))
T_surf_tiles_site = data["T_surf_tiles_site"] - 273.15
print("T_surf:",np.shape(T_surf_tiles_site))
H_site = data["H_site"]
print("H:",np.shape(H_site))
W_site = data["W_site"]
print("W:",np.shape(W_site))
LCF_tiles_site = data["LCF_tiles_site"]
print("LCF:",np.shape(LCF_tiles_site))
LAI_tiles_site = data["LAI_site"]
print("LAI:",np.shape(LAI_tiles_site))

nsite = T_site.shape[0]
ntime = T_site.shape[1]

#--- Update X_obs to follow measured values; also define Z_obs.
#                          RDR   BSG   RJC  AMGI   PV    QAF
phi_obs   =    np.array((115.0,110.0, 90.0,  4.0, 33.0,115.0))   # original values
X_site    =    np.array((  0.0,  0.0,  0.0,  0.0,  0.0,  0.0))
X_obs     =    np.array(( -8.0, -7.0, -2.5, -6.0, -5.5, 73.0)) 
Z_site    =    np.array((  1.5,  1.5,  1.5,  1.5,  1.5,  1.5))
Z_obs     =    np.array((  5.0,  5.0,  3.0,  4.0,  3.0,  4.0))   # original values
W_obs     =    np.array(( 20.0, 30.0, 10.0,260.0,115.0,150.0))   # original values
#H_obs     =    np.array(( 32.0, 28.0, 30.0, 24.0, 16.0, 25.0))   # original values
alb_grnd_obs = np.array((  0.08, 0.08, 0.08, 0.08, 0.08, 0.08))  # original values

phi_obs   =    np.array(( 295.0, 70.0, 90.0,  4.0, 33.0,115.0))   # experimental version
#Z_obs     =    np.array((  5.0,  5.0,  3.0,  4.0,  3.0,  4.0))   # experimental version
#W_obs     =    np.array(( 20.0, 30.0, 10.0,260.0,115.0, 25.0))   # experimental version
X_obs     =    np.array(( -8.0,  5.0, -2.5, -6.0, -5.5, 73.0))   # experimental version
H_obs     =    np.array(( 20.0, 30.0, 30.0, 24.0, 16.0,  8.0))   # experimental version
#alb_grnd_obs = np.array((0.08, 0.08, 0.08, 0.176, 0.00001, 0.08)) # experimental version

canyon_orient_deg_obs = phi_obs

#--- Extract wall and ground temperatures from the surface temperatures on tiles
T_wall_site = np.zeros((nsite,ntime))
T_grnd_site = np.zeros((nsite,ntime))
alb_grnd_site = np.zeros((nsite,))
emiss_grnd_site = np.zeros((nsite,))
for isite in np.arange(nsite):
    T_wall_site[isite,:] = T_surf_tiles_site[tile_number_obs[isite],isite,:]
    T_grnd_site[isite,:] = T_surf_tiles_site[tile_number_obs[isite],isite,:]
    alb_grnd_site[isite] = alb_grnd_tiles[tile_number_obs[isite]]
    emiss_grnd_site[isite] = emiss_grnd_tiles[tile_number_obs[isite]]
#end
print("T_wall:",np.shape(T_wall_site))
print("T_grnd:",np.shape(T_grnd_site))
print("alb_grnd:",np.shape(alb_grnd_site))
print("emiss_grnd:",np.shape(emiss_grnd_site))

#--- Deal with leaf area index.
LAI_site = np.zeros((nsite,))
for isite in np.arange(nsite):
    if tile_number_obs[isite] > 4:
        LAI_site[isite] = 0.0
    else:
        LAI_site[isite] = LAI_tiles_site[tile_number_obs[isite],isite]
    #end
#end
print("LAI:",np.shape(LAI_site))

#--- Calculate WBGT for the model data -- it looks like we will have to do it
#    separately on each site. Create blank arrays, then cycle over sites and times.
WBGT_site = np.zeros((nsite,ntime))
Tg_site = np.zeros((nsite,ntime))
Twb_site = np.zeros((nsite,ntime))
WBGT_site_obsmorph = np.zeros((nsite,ntime))
Tg_site_obsmorph = np.zeros((nsite,ntime))
Twb_site_obsmorph = np.zeros((nsite,ntime))
WBGT_site_obsmorph_loc = np.zeros((nsite,ntime))
Tg_site_obsmorph_loc = np.zeros((nsite,ntime))
Twb_site_obsmorph_loc = np.zeros((nsite,ntime))
WBGT_site_obsmorph_loc_obstemp = np.zeros((nsite,ntime))
Tg_site_obsmorph_loc_obstemp = np.zeros((nsite,ntime))
Twb_site_obsmorph_loc_obstemp = np.zeros((nsite,ntime))

for isite in np.arange(nsite): 
    print(("Working on site: "+sites_obs[isite]+"..."))
    for itime in np.arange(ntime):
        print("time:",timeaxis[itime])
        
        # convert time to different types
        time_dt = datetime.strptime(timeaxis[itime],'%Y%m%dT%H%MZ')
        dateandtime = datetime(year=time_dt.year, month=time_dt.month, 
                               day=time_dt.day, hour=time_dt.hour, 
                               minute=time_dt.minute, second=time_dt.second)

        # constrain on time and convert to data

        # Calculate WBGT using all model values.
        Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
            fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s,  \
            Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT = UCanWBGT.main(
                T=T_site[isite,itime], T_grnd=T_grnd_site[isite,itime], T_wall=T_wall_site[isite,itime], 
                RH=RH_site[isite,itime], q=None, WS=WS_site[isite,itime], P=P_site[isite,itime],
                Ld=Ld_site[isite,itime], Kd=Kd_site[isite,itime], Id=Id_site[isite,itime],
                gamma=gamma, LAI=LAI_site[isite], H=H_site[isite], W=W_site[isite], Z=Z_site[isite], 
                X=X_site[isite], tile_number=tile_number_obs[isite], tf=1.0, #LCF_tiles_site[:,isite], 
                canyon_orient_deg=canyon_orient_deg_obs[isite], a_SW=a_SW, a_LW=a_LW, 
                alb_grnd=alb_grnd_obs[isite], alb_wall=alb_wall, 
                emiss_grnd=emiss_grnd_tiles[tile_number_obs[isite]], emiss_wall=emiss_wall,
                emiss_g=emiss_g, a_g=a_g, d=d, lat=lat_obs[isite], lon=lon_obs[isite], 
                elevation=elevation, dateandtime=dateandtime, tzinfo=tzinfo,
                WBGT_model_choice=WBGT_model_choice, nref=nref, gamma_choice=gamma_choice, 
                WBGT_equation_choice=WBGT_equation_choice, Twb_method=Twb_method)

        WBGT_site[isite,itime] = WBGT
        Tg_site[isite,itime] = Tg
        Twb_site[isite,itime] = Twb

        # Calculate WBGT using all model values but observed morphology.
        Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
            fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s,  \
            Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT = UCanWBGT.main(
                T=T_site[isite,itime], T_grnd=T_grnd_site[isite,itime], T_wall=T_wall_site[isite,itime], 
                RH=RH_site[isite,itime], q=None, WS=WS_site[isite,itime], P=P_site[isite,itime],
                Ld=Ld_site[isite,itime], Kd=Kd_site[isite,itime], Id=Id_site[isite,itime],
                gamma=gamma, LAI=LAI_site[isite], H=H_obs[isite], W=W_obs[isite], Z=Z_site[isite], 
                X=X_site[isite], tile_number=tile_number_obs[isite], tf=1.0, #LCF_tiles_site[:,isite], 
                canyon_orient_deg=canyon_orient_deg_obs[isite], a_SW=a_SW, a_LW=a_LW, 
                alb_grnd=alb_grnd_obs[isite], alb_wall=alb_wall, 
                emiss_grnd=emiss_grnd_tiles[tile_number_obs[isite]], emiss_wall=emiss_wall,
                emiss_g=emiss_g, a_g=a_g, d=d, lat=lat_obs[isite], lon=lon_obs[isite], 
                elevation=elevation, dateandtime=dateandtime, tzinfo=tzinfo,
                WBGT_model_choice=WBGT_model_choice, nref=nref, gamma_choice=gamma_choice, 
                WBGT_equation_choice=WBGT_equation_choice, Twb_method=Twb_method)

        WBGT_site_obsmorph[isite,itime] = WBGT
        Tg_site_obsmorph[isite,itime] = Tg
        Twb_site_obsmorph[isite,itime] = Twb
 
        Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
            fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s,  \
            Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT = UCanWBGT.main(
                T=T_site[isite,itime], T_grnd=T_grnd_site[isite,itime], T_wall=T_wall_site[isite,itime], 
                RH=RH_site[isite,itime], q=None, WS=WS_site[isite,itime], P=P_site[isite,itime],
                Ld=Ld_site[isite,itime], Kd=Kd_site[isite,itime], Id=Id_site[isite,itime],
                gamma=gamma, LAI=LAI_site[isite], H=H_obs[isite], W=W_obs[isite], Z=Z_obs[isite], 
                X=X_obs[isite], tile_number=tile_number_obs[isite], tf=1.0, #LCF_tiles_site[:,isite], 
                canyon_orient_deg=canyon_orient_deg_obs[isite], a_SW=a_SW, a_LW=a_LW, 
                alb_grnd=alb_grnd_obs[isite], alb_wall=alb_wall, 
                emiss_grnd=emiss_grnd_tiles[tile_number_obs[isite]], emiss_wall=emiss_wall,
                emiss_g=emiss_g, a_g=a_g, d=d, lat=lat_obs[isite], lon=lon_obs[isite], 
                elevation=elevation, dateandtime=dateandtime, tzinfo=tzinfo,
                WBGT_model_choice=WBGT_model_choice, nref=nref, gamma_choice=gamma_choice, 
                WBGT_equation_choice=WBGT_equation_choice, Twb_method=Twb_method)

        WBGT_site_obsmorph_loc[isite,itime] = WBGT
        Tg_site_obsmorph_loc[isite,itime] = Tg
        Twb_site_obsmorph_loc[isite,itime] = Twb

        Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
            fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s,  \
            Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT = UCanWBGT.main(
                T=T_obs[isite,itime], T_grnd=T_grnd_site[isite,itime], T_wall=T_wall_site[isite,itime], 
                RH=RH_obs[isite,itime], q=None, WS=WS_obs[isite,itime], P=P_site[isite,itime],
                Ld=Ld_site[isite,itime], Kd=Kd_site[isite,itime], Id=Id_site[isite,itime],
                gamma=gamma, LAI=LAI_site[isite], H=H_obs[isite], W=W_obs[isite], Z=Z_obs[isite], 
                X=X_obs[isite], tile_number=tile_number_obs[isite], tf=1.0, #LCF_tiles_site[:,isite], 
                canyon_orient_deg=canyon_orient_deg_obs[isite], a_SW=a_SW, a_LW=a_LW, 
                alb_grnd=alb_grnd_obs[isite], alb_wall=alb_wall, 
                emiss_grnd=emiss_grnd_tiles[tile_number_obs[isite]], emiss_wall=emiss_wall,
                emiss_g=emiss_g, a_g=a_g, d=d, lat=lat_obs[isite], lon=lon_obs[isite], 
                elevation=elevation, dateandtime=dateandtime, tzinfo=tzinfo,
                WBGT_model_choice=WBGT_model_choice, nref=nref, gamma_choice=gamma_choice, 
                WBGT_equation_choice=WBGT_equation_choice, Twb_method=Twb_method)

        WBGT_site_obsmorph_loc_obstemp[isite,itime] = WBGT
        Tg_site_obsmorph_loc_obstemp[isite,itime] = Tg
        Twb_site_obsmorph_loc_obstemp[isite,itime] = Twb
        
    #end
#end

#--- Calculate the WBGT for the observations -- much simpler and only takes three lines!
q_obs = UCanWBGT.q_from_RH(T_obs,RH_obs,P_site)

Twb_obs = UCanWBGT.Twb_func(T_obs,RH_obs,q_obs,P_site,Twb_method)

WBGT_obs = UCanWBGT.WBGT_func(T_obs,Twb_obs,Tg_obs,RH_obs,
                              Ld_site,Kd_site,Id_site,
                              a_g,a_LW,a_SW,WBGT_model_choice,WBGT_equation_choice)



#--- Plot some time series.
timeaxis_hours = np.arange(0,24,0.5) # create a time axis that contains numbers

plt.figure(figsize=(14, 6))

plt.subplot(2,3,1)
plt.plot(timeaxis_hours, WBGT_site[0,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WBGT_obs[0,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, WBGT_site_obsmorph[0,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc[0,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc_obstemp[0,:], marker='o', color='c', label='...+obs temp/RH')
plt.ylabel(r'WBGT / $^\circ$C')
plt.xticks(np.arange(0,25,3))
plt.title("Rue de Rivoli")
plt.grid(True)

plt.subplot(2,3,2)
plt.plot(timeaxis_hours, WBGT_site[1,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WBGT_obs[1,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, WBGT_site_obsmorph[1,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc[1,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc_obstemp[1,:], marker='o', color='c', label='...+obs temp/RH')
plt.xticks(np.arange(0,25,3))
plt.title("Boulevard St Germain")
plt.grid(True)

plt.subplot(2,3,3)
plt.plot(timeaxis_hours, WBGT_site[2,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WBGT_obs[2,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, WBGT_site_obsmorph[2,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc[2,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc_obstemp[2,:], marker='o', color='c', label='...+obs temp/RH')
plt.xticks(np.arange(0,25,3))
plt.title("Rue Jacques Callot")
plt.grid(True)

plt.subplot(2,3,4)
plt.plot(timeaxis_hours, WBGT_site[3,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WBGT_obs[3,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, WBGT_site_obsmorph[3,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc[3,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc_obstemp[3,:], marker='o', color='c', label='...+obs temp/RH')
plt.ylabel(r'WBGT / $^\circ$C')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Avenue Marechal Galieni Invalides")
plt.grid(True)

plt.subplot(2,3,5)
plt.plot(timeaxis_hours, WBGT_site[4,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WBGT_obs[4,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, WBGT_site_obsmorph[4,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc[4,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc_obstemp[4,:], marker='o', color='c', label='...+obs temp/RH')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Place Vendome")
plt.grid(True)

plt.subplot(2,3,6)
plt.plot(timeaxis_hours, WBGT_site[5,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WBGT_obs[5,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, WBGT_site_obsmorph[5,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc[5,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc_obstemp[5,:], marker='o', color='c', label='...+obs temp/RH')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Quai Anatole France")
plt.grid(True)
plt.legend()

plt.suptitle("WET-BULB GLOBE TEMPERATURE from model and observations at six urban locations in Paris")
plt.tight_layout()


plt.figure(figsize=(14,6))

plt.subplot(2,3,1)
plt.plot(timeaxis_hours, Tg_site[0,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Tg_obs[0,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, Tg_site_obsmorph[0,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc[0,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc_obstemp[0,:], marker='o', color='c', label='...+obs temp/RH')
plt.ylabel(r'BGT / $^\circ$C')
plt.xticks(np.arange(0,25,3))
plt.title("Rue de Rivoli")
plt.grid(True)

plt.subplot(2,3,2)
plt.plot(timeaxis_hours, Tg_site[1,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Tg_obs[1,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, Tg_site_obsmorph[1,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc[1,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc_obstemp[1,:], marker='o', color='c', label='...+obs temp/RH')
plt.xticks(np.arange(0,25,3))
plt.title("Boulevard St Germain")
plt.grid(True)

plt.subplot(2,3,3)
plt.plot(timeaxis_hours, Tg_site[2,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Tg_obs[2,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, Tg_site_obsmorph[2,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc[2,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc_obstemp[2,:], marker='o', color='c', label='...+obs temp/RH')
plt.xticks(np.arange(0,25,3))
plt.title("Rue Jacques Callot")
plt.grid(True)

plt.subplot(2,3,4)
plt.plot(timeaxis_hours, Tg_site[3,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Tg_obs[3,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, Tg_site_obsmorph[3,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc[3,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc_obstemp[3,:], marker='o', color='c', label='...+obs temp/RH')
plt.ylabel(r'BGT / $^\circ$C')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Avenue Marechal Galieni Invalides")
plt.grid(True)

plt.subplot(2,3,5)
plt.plot(timeaxis_hours, Tg_site[4,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Tg_obs[4,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, Tg_site_obsmorph[4,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc[4,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc_obstemp[4,:], marker='o', color='c', label='...+obs temp/RH')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Place Vendome")
plt.grid(True)

plt.subplot(2,3,6)
plt.plot(timeaxis_hours, Tg_site[5,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Tg_obs[5,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, Tg_site_obsmorph[5,:], marker='o', color='g', label='model+obs morph')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc[5,:], marker='o', color='y', label='...+location')
plt.plot(timeaxis_hours, Tg_site_obsmorph_loc_obstemp[5,:], marker='o', color='c', label='...+obs temp/RH')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Quai Anatole France")
plt.grid(True)
plt.legend()

plt.suptitle("BLACK-GLOBE TEMPERATURE from model and observations at six urban locations in Paris")
plt.tight_layout()


plt.figure(figsize=(14,6))

plt.subplot(2,3,1)
plt.plot(timeaxis_hours, T_site[0,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, T_obs[0,:], marker='o', color='r', label='observations')
plt.ylabel(r'temperature / $^\circ$C')
plt.xticks(np.arange(0,25,3))
plt.title("Rue de Rivoli")
plt.grid(True)

plt.subplot(2,3,2)
plt.plot(timeaxis_hours, T_site[1,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, T_obs[1,:], marker='o', color='r', label='observations')
plt.xticks(np.arange(0,25,3))
plt.title("Boulevard St Germain")
plt.grid(True)

plt.subplot(2,3,3)
plt.plot(timeaxis_hours, T_site[2,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, T_obs[2,:], marker='o', color='r', label='observations')
plt.xticks(np.arange(0,25,3))
plt.title("Rue Jacques Callot")
plt.grid(True)

plt.subplot(2,3,4)
plt.plot(timeaxis_hours, T_site[3,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, T_obs[3,:], marker='o', color='r', label='observations')
plt.ylabel(r'temperature / $^\circ$C')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Avenue Marechal Galieni Invalides")
plt.grid(True)

plt.subplot(2,3,5)
plt.plot(timeaxis_hours, T_site[4,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, T_obs[4,:], marker='o', color='r', label='observations')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Place Vendome")
plt.grid(True)

plt.subplot(2,3,6)
plt.plot(timeaxis_hours, T_site[5,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, T_obs[5,:], marker='o', color='r', label='observations')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Quai Anatole France")
plt.grid(True)
plt.legend()

plt.suptitle("TEMPERATURE from model and observations at six urban locations in Paris")
plt.tight_layout()


plt.figure(figsize=(14,6))

plt.subplot(2,3,1)
plt.plot(timeaxis_hours, Twb_site[0,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Twb_obs[0,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, Twb_site_obsmorph[0,:], marker='o', color='g', label='observations')
plt.ylabel(r'WBT / $^\circ$C')
plt.xticks(np.arange(0,25,3))
plt.title("Rue de Rivoli")
plt.grid(True)

plt.subplot(2,3,2)
plt.plot(timeaxis_hours, Twb_site[1,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Twb_obs[1,:], marker='o', color='r', label='observations')
plt.xticks(np.arange(0,25,3))
plt.title("Boulevard St Germain")
plt.grid(True)

plt.subplot(2,3,3)
plt.plot(timeaxis_hours, Twb_site[2,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Twb_obs[2,:], marker='o', color='r', label='observations')
plt.xticks(np.arange(0,25,3))
plt.title("Rue Jacques Callot")
plt.grid(True)

plt.subplot(2,3,4)
plt.plot(timeaxis_hours, Twb_site[3,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Twb_obs[3,:], marker='o', color='r', label='observations')
plt.ylabel(r'WBT / $^\circ$C')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Avenue Marechal Galieni Invalides")
plt.grid(True)

plt.subplot(2,3,5)
plt.plot(timeaxis_hours, Twb_site[4,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Twb_obs[4,:], marker='o', color='r', label='observations')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Place Vendome")
plt.grid(True)

plt.subplot(2,3,6)
plt.plot(timeaxis_hours, Twb_site[5,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, Twb_obs[5,:], marker='o', color='r', label='observations')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Quai Anatole France")
plt.grid(True)
plt.legend()

plt.suptitle("WET-BULB TEMPERATURE from model and observations at six urban locations in Paris")
plt.tight_layout()


plt.figure(figsize=(14,6))

plt.subplot(2,3,1)
plt.plot(timeaxis_hours, RH_site[0,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, RH_obs[0,:], marker='o', color='r', label='observations')
plt.ylabel(r'RH (%)')
plt.xticks(np.arange(0,25,3))
plt.title("Rue de Rivoli")
plt.grid(True)

plt.subplot(2,3,2)
plt.plot(timeaxis_hours, RH_site[1,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, RH_obs[1,:], marker='o', color='r', label='observations')
plt.xticks(np.arange(0,25,3))
plt.title("Boulevard St Germain")
plt.grid(True)

plt.subplot(2,3,3)
plt.plot(timeaxis_hours, RH_site[2,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, RH_obs[2,:], marker='o', color='r', label='observations')
plt.xticks(np.arange(0,25,3))
plt.title("Rue Jacques Callot")
plt.grid(True)

plt.subplot(2,3,4)
plt.plot(timeaxis_hours, RH_site[3,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, RH_obs[3,:], marker='o', color='r', label='observations')
plt.ylabel(r'RH (%)')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Avenue Marechal Galieni Invalides")
plt.grid(True)

plt.subplot(2,3,5)
plt.plot(timeaxis_hours, RH_site[4,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, RH_obs[4,:], marker='o', color='r', label='observations')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Place Vendome")
plt.grid(True)

plt.subplot(2,3,6)
plt.plot(timeaxis_hours, RH_site[5,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, RH_obs[5,:], marker='o', color='r', label='observations')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Quai Anatole France")
plt.grid(True)
plt.legend()

plt.suptitle("RELATIVE HUMIDITY from model and observations at six urban locations in Paris")
plt.tight_layout()


plt.figure(figsize=(14,6))

plt.subplot(2,3,1)
plt.plot(timeaxis_hours, WS_site[0,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WS_obs[0,:], marker='o', color='r', label='observations')
plt.ylabel(r'wind speed / m s$^{-1}$')
plt.xticks(np.arange(0,25,3))
plt.title("Rue de Rivoli")
plt.grid(True)

plt.subplot(2,3,2)
plt.plot(timeaxis_hours, WS_site[1,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WS_obs[1,:], marker='o', color='r', label='observations')
plt.xticks(np.arange(0,25,3))
plt.title("Boulevard St Germain")
plt.grid(True)

plt.subplot(2,3,3)
plt.plot(timeaxis_hours, WS_site[2,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WS_obs[2,:], marker='o', color='r', label='observations')
plt.xticks(np.arange(0,25,3))
plt.title("Rue Jacques Callot")
plt.grid(True)

plt.subplot(2,3,4)
plt.plot(timeaxis_hours, WS_site[3,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WS_obs[3,:], marker='o', color='r', label='observations')
plt.ylabel(r'wind speed / m s$^{-1}$')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Avenue Marechal Galieni Invalides")
plt.grid(True)

plt.subplot(2,3,5)
plt.plot(timeaxis_hours, WS_site[4,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WS_obs[4,:], marker='o', color='r', label='observations')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Place Vendome")
plt.grid(True)

plt.subplot(2,3,6)
plt.plot(timeaxis_hours, WS_site[5,:], marker='o', color='b', label='model')
plt.plot(timeaxis_hours, WS_obs[5,:], marker='o', color='r', label='observations')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Quai Anatole France")
plt.grid(True)
plt.legend()

plt.suptitle("WIND SPEED from model and observations at six urban locations in Paris")
plt.tight_layout()
#plt.savefig(f"./figs/obs_timeseries.png", dpi=300, bbox_inches='tight')

plt.show()
#%%


#---- SAMPLE FIGURE FOR LEWIS'S PRESENTATION

plt.figure(figsize=(14, 4))

plt.subplot(1,3,3)
plt.plot(timeaxis_hours, WBGT_obs[0,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc[0,:], marker='o', color='b', label='UCanWBGT')
plt.ylabel(r'WBGT / $^\circ$C')
plt.xticks(np.arange(0,25,3))
plt.title("Rue de Rivoli")
plt.legend()
plt.grid(True)

plt.subplot(1,3,1)
plt.plot(timeaxis_hours, WBGT_obs[2,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc[2,:], marker='o', color='b', label='UCanWBGT')
plt.ylabel(r'WBGT / $^\circ$C')
plt.xticks(np.arange(0,25,3))
plt.title("Rue Jacques Callot")
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(timeaxis_hours, WBGT_obs[3,:], marker='o', color='r', label='observations')
plt.plot(timeaxis_hours, WBGT_site_obsmorph_loc[3,:], marker='o', color='b', label='UCanWBGT')
plt.xlabel('Time after 00:00 on 07 Sep 2023 / h')
plt.xticks(np.arange(0,25,3))
plt.title("Avenue Marechal Galieni Invalides")
plt.grid(True)


plt.suptitle("WET-BULB GLOBE TEMPERATURE from model and observations at locations in Paris")
plt.tight_layout()

