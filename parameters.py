"""
UCanWBGT heat index calculation parameters.
"""

time = "20240621T1200Z" # time UTC
tzinfo = "UTC" # timezones https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
WBGT_model_choice = "UCanWBGT_outdoor" # whether to use `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` model
geometry_choice = "canyon" # whether to use canyon or flat geometry
nref = 2 # defines the number of shortwave diffuse reflections as nref and the number of shortwave direct reflections as nref+1
gamma_choice = "prescribe" # whether to use a gamma direct beam attenuation factor that is `prescribed` or modelled using `LAI`
WBGT_equation_choice = "full" # whether to use the `full` or approximate `ISO` WBGT equation
Twb_method = "thermo_isobaric" # whether to use the `Stull` (2011), `thermo_isobaric`, or `thermo_adiabatic` Twb method
gamma = 1.0 # direct beam attenuation factor (1.0 is no attenuation and 0.0 is full attenuation)
LAI = 1.0 # leaf area index (m2 m-2)
lat = 55.0 # latitude (deg)
lon = 0.0 # longitude (deg)
elevation = 0.0 # height with respect to the horizon (m)
Z = 1.5 # black globe height (m)
H = 10.0 # canyon height (m)
W = 10.0 # canyon width (m)
X = 0.0 # black globe position where 0.0 is the centre and positive/negative values are to right/left of centre (m)
tf = 1.0 # tile fraction (should be one by default -- if provided as an array with the same dimensions as lat/lon then zero values will be ignored in the calculation)
canyon_orient_deg = 0.0 # the horizontal angle measured clockwise from north to a line running parallel to the alignment of the street canyon
tile_number = 8 # which tile to use (note: if tiles 8 or 9 are selected the calculation will be for the urban area using tiles 8 and 9). 0: Broad Leaf Tree, 1: Needle Leaf Tree, 2: C3 Grass, 3: C4 Grass, 4: Shrub, 5: Inland Water, 6: Bare Soil, 7: Ice, 8: Urban Canyon, 9: Urban Roof
alb_grnd_tiles = [0.143,0.088,0.176,0.16,0.193,0.06,0.1,0.75,0.08,0.18] # MORUSES 10 tile albedo values
alb_wall = 0.375 # MORUSES wall albedo value
emiss_grnd_tiles = [0.98,0.99,0.98,0.98,0.98,0.985,0.9,0.99,0.95,0.97]  # MORUSES 10 tile emissivity values
emiss_wall = 0.875 # MORUSES wall emissivity value
emiss_g = 0.95 # black globe emissivity
a_g = 0.95 # black globe absorptivity
d = 0.15 # black globe diameter (m)
a_SW = 0.7 # shortwave absorptivity of a human
a_LW = 0.97 # longwave absorptivity of a human

