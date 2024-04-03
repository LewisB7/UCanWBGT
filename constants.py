"""
UCanWBGT heat index calculation constants.
"""

g = 9.81 # Acceleration due to gravity (m s-2)
Rd = 287.0 # Specific gas constant for dry air (J kg-1 K-1)
Rv = 461.5 # Specific gas constant for water vapour (J kg-1K -1)
eps = Rd / Rv # Ratio of gas constants for dry air and water vapour
cpd = 1005.0 # Isobaric specific heat of dry air (J kg-1 K-1)
cpv = 2040.0 # # Isobaric specific heat of water vapour (J kg-1 K-1), optimised value from Ambaum (2020)
cpl = 4220.0 # Isobaric specific heat of liquid water (J kg-1 K-1), triple-point value from Wagner and Pruß (2002)
cpi = 2097.0 # Isobaric specific heat of ice (J kg-1 K-1), triple-point value from Feistel and Wagner (2006)
p_ref = 1.0e5 # Reference pressure (Pa)
T0 = 273.16 # Triple point temperature (K)
es0 = 611.657 # Saturation vapour pressure at the triple point (Pa), Guildner et al. (1976)
Lv0 = 2.501e6 # Latent heat of vaporisation at the triple point (J kg-1), Wagner and Pruß (2002)
Lf0 = 0.333e6 # Latent heat of freezing at the triple point (J kg-1), Feistel and Wagner (2006)
Ls0 = Lv0 + Lf0 # Latent heat of sublimation at the triple point (J kg-1)
T_liq = 273.15 # Temperature above which all condensate is assumed to be liquid (K)
T_ice = 253.15 # Temperature below which all condensate is assumed to be ice (K)
sboltz = 0.0000000567 # Stefan-Boltzmann constant (W m-2 K-4)

