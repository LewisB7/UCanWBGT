"""
SENSITIVITY_TESTS_UCANWBGT: runs a set of sensitivity tests using the UCanWBGT 
  code to check that it behaves correctly.

"""


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
from parameters import (time, tzinfo, WBGT_model_choice, geometry_choice, nref, gamma_choice, WBGT_equation_choice,
                        Twb_method, gamma, LAI, lat, lon, elevation, canyon_orient_deg, Z, H, W, X,
                        tile_number, alb_grnd_tiles, alb_wall, emiss_grnd_tiles, emiss_wall,
                        emiss_g, a_g, d, a_SW, a_LW)

### Set Remaining Parameters ###

time = "20240621T1500Z"

t = datetime.strptime(time,'%Y%m%dT%H%MZ')
dateandtime = datetime(year=t.year, month=t.month, day=t.day, hour=t.hour, minute=t.minute, second=t.second)
alb_grnd = alb_grnd_tiles[tile_number]
emiss_grnd = emiss_grnd_tiles[tile_number]

# Now modify a field to vary -- change the height of the black globe
# to be 1.0 m, then vary canyon height from 1.0 m up to 10 m.
Z = 0.999
H = np.arange(1.0, 50.0, 0.2)


# Set forcing variables
T = np.zeros(len(H)) + 35.
T_grnd = np.zeros(len(H)) + 35.
T_wall = np.zeros(len(H)) + 35.
RH = np.zeros(len(H)) + 75.
WS = np.zeros(len(H)) + 2.
Ld = np.zeros(len(H)) + 450.
Kd = np.zeros(len(H)) + 200.
Id = np.zeros(len(H)) + 600.
P = np.zeros(len(H)) + 101325.0

lat = np.zeros(len(H)) + lat
lon = np.zeros(len(H)) + lon

# calculate WBGT
Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
    fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fpw1r, Fpw1w2, Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT \
        = UCanWBGT.main(
        T=T, T_grnd=T_grnd, T_wall=T_wall, RH=RH, q=None, WS=WS, P=P, Ld=Ld, Kd=Kd, Id=Id,\
        gamma=gamma, LAI=LAI,\
        H=H, W=W, Z=Z, X=X, canyon_orient_deg=canyon_orient_deg,\
        a_SW=a_SW, a_LW=a_LW,\
        alb_grnd=alb_grnd, alb_wall=alb_wall, emiss_grnd=emiss_grnd, emiss_wall=emiss_wall,\
        emiss_g=emiss_g, a_g=a_g, d=d,\
        lat=lat, lon=lon, elevation=elevation, dateandtime=dateandtime, tzinfo=tzinfo,\
        WBGT_model_choice=WBGT_model_choice, geometry_choice=geometry_choice, nref=nref,\
        gamma_choice=gamma_choice, WBGT_equation_choice=WBGT_equation_choice, Twb_method=Twb_method 
        )
#print(", ".join([f"Twb: {Twb}", f"solar_zen_deg: {solar_zen_deg}", f"solar_azi_deg: {solar_azi_deg}", f"canyon_azi_deg: {canyon_azi_deg}", f"Fs: {Fs}", f"Fr: {Fr}", f"Fw: {Fw}", f"Fsr: {Fsr}", f"Frs: {Frs}", f"Fww: {Fww}", f"Fwr: {Fwr}", f"Fws: {Fws}", f"Frw: {Frw}", f"Fsw: {Fsw}", f"fr: {fr}", f"fw: {fw}", f"Fpr: {Fpr}", f"Fpw: {Fpw}", f"Fprw1: {Fprw1}", f"Fprw2: {Fprw2}", f"Fpw1r: {Fpw1r}", f"Fpw1w2: {Fpw1w2}", f"Sr: {Sr}", f"Sw: {Sw}", f"K: {K}", f"Ks: {Ks}", f"Kr: {Kr}", f"Kw: {Kw}", f"I: {I}", f"L: {L}", f"MRT: {MRT}", f"Tg: {Tg}", f"WBGT: {WBGT}"]))


plt.figure(figsize=(15,6))

plt.subplot(2,4,1)
plt.plot(H,Fs,"c-",label="Fs")
plt.plot(H,Fw,"r-",label="Fw")
plt.plot(H,Fr,"g-",label="Fr")
plt.plot(H,(Fs+Fw+Fr),"k--",label="Fs+Fw+Fr")
plt.legend()
plt.grid(True)
plt.ylabel("fractional quantity (see legends)")

plt.subplot(2,4,2)
plt.plot(H,Fsw,"r-",label="Fsw")
plt.plot(H,Fsr,"g-",label="Fsr")
plt.plot(H,(Fsw+Fsw+Fsr),"k--",label="2Fsw+Fsr")
plt.legend()
plt.grid(True)

plt.subplot(2,4,3)
plt.plot(H,Fws,"c-",label="Fws")
plt.plot(H,Fww,"r-",label="Fww")
plt.plot(H,Fwr,"g-",label="Fwr")
plt.plot(H,(Fws+Fww+Fwr),"k--",label="Fws+Fww+Fwr")
plt.legend()
plt.grid(True)

plt.subplot(2,4,4)
plt.plot(H,Frs,"c-",label="Frs")
plt.plot(H,Frw,"r-",label="Frw")
plt.plot(H,(Frs+Frw+Frw),"k--",label="Frs+2Frw")
plt.legend()
plt.grid(True)

plt.subplot(2,4,5)
plt.plot(H,fw,"r-",label="fw")
plt.plot(H,fr,"g-",label="fr")
plt.legend()
plt.grid(True)
plt.ylabel("fractional quantity (see legends)")
plt.xlabel("canyon height H / m")

plt.subplot(2,4,6)
plt.plot(H,Fpw,"r-",label="Fpw")
plt.plot(H,Fpr,"g-",label="Fpr")
plt.legend()
plt.grid(True)
plt.xlabel("canyon height H / m")

plt.subplot(2,4,7)
plt.plot(H,Fpw1w2,"m-",label="Fpw1w2")
plt.plot(H,Fpw1r,"g-",label="Fpw1r")
plt.legend()
plt.grid(True)
plt.xlabel("canyon height H / m")

plt.subplot(2,4,8)
plt.plot(H,Fprw1,"r-",label="Fprw1")
plt.plot(H,Fprw2,"m-",label="Fprw2")
plt.legend()
plt.grid(True)
plt.xlabel("canyon height H / m")


plt.suptitle(("Variations of fractional quantities in UCanWBGT "  \
              +"with varying canyon height. Z = 1.0 m; W = 10.0 m; " \
                +"X = 0.0 m; SZA = 45.5; SAA = 244.4."))


plt.show()