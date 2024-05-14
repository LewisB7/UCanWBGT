import numpy as np
import pvlib
import pytz
import sys
from scipy.special import lambertw
from constants import (Rd, Rv, eps, cpd, cpv, cpl, cpi, p_ref,
                            T0, es0, Lv0, Lf0, Ls0, T_liq, T_ice, sboltz)

"""
UCanWBGT heat index calculation functions.
"""

def adjust_array_shape(array_or_float_list, reference):
    """
    Adjust the shape of the input array or float to match the shape of the reference array.
    If the input is a float, it will be repeated to create an array with the same shape as the reference.

    Arguments:
    - array_or_float (The array or float; numpy.ndarray or float)  
    - reference (The reference array or float; numpy.ndarray or float)

    Returns:
    - array_or_float (The adjusted array or float; numpy.ndarray or float)     
    """

    out_list = []
    for i in range(len(array_or_float_list)):
        array_or_float = array_or_float_list[i]
        if array_or_float is None:
            pass
        else:
            if isinstance(array_or_float, float):
                array_or_float = np.array([array_or_float])
            if (array_or_float.shape != reference.shape):
                array_or_float = np.full_like(reference, array_or_float)
        out_list.append(array_or_float)

    return out_list

def mask_and_1d(arr,mask):

    if arr is None:
        arr = None
    else:
        arr_without_zeros = arr[~mask]
        arr = arr_without_zeros.flatten()

    return arr

def put_array_shape_back(array_or_float_list, mask):

    out_list = []
    for i in range(len(array_or_float_list)):
        result_1d = array_or_float_list[i]
        result = np.zeros(np.shape(mask))
        result[~mask] = result_1d
        out_list.append(result)

    return out_list

def q_from_RH(T,RH,P):
    """
    Specific humidity calculation. 

    Arguments:
    - T (air temperature; degC)
    - RH (relative humidity; %)
    - P (surface pressure; Pa)

    Returns:
    - q (specific humidity; kg kg-1)
    """

    es = 6.108*np.exp((17.27*T/(T+237.3)))*100. # saturated water vapour pressure (Pa) (Liu et al., 2018)
    ea = es*RH/100 # water vapour pressure (Pa) (Liu et al., 2018)
    q = ea*0.6219/(P-ea*(1-0.6219)) # specific humidity (kg kg-1) (rearranging thermo.py vapour_pressure)

    return q

def RH_from_q(T,q,P):
    """
    Relative humidity calculation. 

    Arguments:
    - T (air temperature; degC)
    - q (specifc humidity; kg kg-1)
    - P (surface pressure; Pa)

    Returns:
    - RH (relative humidity; %)
    """

    es = 6.108*np.exp((17.27*T/(T+237.3)))*100. # saturated water vapour pressure (Pa) (Liu et al., 2018)
    ea = P*q/(0.6219*(1-q) + q) # water vapour pressure (Pa) (thermo.py vapour_pressure)
    RH = 100*ea/es # relative humidity (%) (Liu et al., 2018)    

    return RH

def Twb_Stull_func(T,RH):
    """
    Wet bulb temperature calculation. 
    Stull (2011).

    Arguments:
    - T (air temperature; degC)
    - RH (relative humidity; %)

    Returns:
    - Twb (wet bulb temperature; degC)
    """

    Twb = T*np.arctan(0.151977*(RH + 8.313659)**0.5)\
        + np.arctan(T + RH) - np.arctan(RH - 1.676331)\
            + 0.00391838*RH**(1.5)*np.arctan(0.023101*RH)\
                -4.686035

    return Twb

def lifting_condensation_level(p,T,q,RH):
    """
    Function taken from https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py
    (2024/04/02).

    Computes pressure and temperature at the lifted condensation level (LCL)
    using equations from Romps (2017).

    Arguments:
    - p (pressure; Pa)
    - T (temperature; K)
    - q (specific humidity; kg kg-1)
    - RH (relative humidity; fraction)

    Returns:
    - p_lcl (pressure at the LCL; Pa)
    - T_lcl (temperature at the LCL; K)
    """
   
    # Compute effective gas constant and specific heat
    Rm = (1 - q) * Rd + q * Rv
    cpm = (1 - q) * cpd + q * cpv

    # Set constants (Romps 2017, Eq. 22d-f)
    a = cpm / Rm + (cpl - cpv) / Rv
    b = -(Lv0 + (cpl - cpv) * T0) / (Rv * T)
    c = b / a

    # Compute temperature at the LCL (Romps 2017, Eq. 22a)
    fn = np.power(RH, (1 / a)) * c * np.exp(c)
    W = lambertw(fn, k=-1).real
    T_lcl = c * (1 / W) * T
    
    # Compute pressure at the LCL (Romps 2017, Eq. 22b)
    p_lcl = p * np.power((T_lcl / T), (cpm / Rm))
    
    # Ensure that LCL temperature and pressure do not exceed initial values
    T_lcl = np.minimum(T_lcl, T)
    p_lcl = np.minimum(p_lcl, p)
    
    return p_lcl, T_lcl

def pseudoadiabat_wbpt(p,T):
    """
    Function taken from https://github.com/robwarrenwx/atmos/blob/main/atmos/pseudoadiabat.py
    (2024/04/02).

    Computes the wet-bulb potential temperature (WBPT) thw of the
    pseudoadiabat that passes through pressure p and temperature T.

    Uses polynomial approximations from Moisseeva and Stull (2017)
    with revised coefficients.

    Moisseeva, N. and Stull, R., 2017. A noniterative approach to
        modelling moist thermodynamics. Atmospheric Chemistry and
        Physics, 17, 15037-15043.

    Arguments:
    - p (pressure; Pa)
    - T (temperature; K)

    Returns:
    - thw (wet-bulb potential temperature; K)
    """

    # Convert p to hPa and T to degC
    p_ = p / 100.
    T_ = T - 273.15

    # Check that values fall in the permitted range
    condition = (np.any(T_ < -100.) or np.any(T_ > 50.) or np.any(p_ > 1100.) or np.any(p_ < 50.))
    if condition:
        return np.nan

    # Compute theta-w using Eq. 4-6 from Moisseeva & Stull 2017
    Tref = 5.480079558912395754e+01 + p_*(-5.702912372295836363e-01 + p_*(6.214635352625029474e-03 + p_*(-6.634002758531432769e-05 + p_*(4.775890354003946154e-07 + p_*(-2.284857526370373519e-09 + p_*(6.641586128297075239e-12 + p_*(-7.712521760947926640e-15 + p_*(-2.044872914238500011e-17 + p_*(1.120406966307735934e-19 + p_*(-2.393726252696363534e-22 + p_*(2.666858658235125377e-25 + p_*(-8.684116177147550627e-29 + p_*(-1.672712626829787198e-31 + p_*(2.183518619078883688e-34 + p_*(-1.547439304626812778e-38 + p_*(-1.937257327731052786e-40 + p_*(2.161580416088237645e-43 + p_*(-1.160157488827817865e-46 + p_*(3.277570207101446812e-50 + p_*(-3.923759467350554795e-54))))))))))))))))))))  # noqa: E501
    thw = 4.232924249278688222e+01 + T_*(5.718008668788681081e-01 + T_*(6.676735845163824824e-03 + T_*(4.022733506471462767e-05 + T_*(-9.509825840570524462e-07 + T_*(-7.879928387880090530e-09 + T_*(1.767656648682178749e-10 + T_*(2.337360533109320417e-12 + T_*(-3.564315751634256907e-14 + T_*(-6.615642909573364126e-16 + T_*(5.465028726086066100e-18 + T_*(1.710624575834384904e-19 + T_*(-1.714074684770886933e-22 + T_*(-3.374318285502554966e-23 + T_*(-1.906956286638301820e-25 + T_*(3.748889164551010026e-27 + T_*(4.895600997189238897e-29 + T_*(-3.555428293757981745e-32 + T_*(-3.897220799151443075e-33 + T_*(-2.551411607182840165e-35 + T_*(-5.417589606240612172e-38)))))))))))))))))))) + Tref*(3.623240553023549526e-01 + T_*(1.023375828905026411e-03 + T_*(7.652197539188903771e-05 + T_*(-3.335127874231546565e-07 + T_*(-1.156314790854086800e-08 + T_*(5.997724820856132307e-11 + T_*(3.205744785514147323e-12 + T_*(-3.725255127402225228e-15 + T_*(-4.985713597883638890e-16 + T_*(-1.788582915460586641e-17 + T_*(8.672108280207142462e-20 + T_*(1.195118892414961522e-20 + T_*(-3.811303263205360248e-24 + T_*(-3.991925996145037697e-24 + T_*(-1.495975110164947026e-26 + T_*(6.968870338282429091e-28 + T_*(6.209536671787076346e-30 + T_*(-3.566388507077176018e-32 + T_*(-7.450360218688953006e-34 + T_*(-3.816398138670827989e-36 + T_*(-6.712873080670899696e-39)))))))))))))))))))) + Tref*(2.901968635714725214e-03 + T_*(6.487857234220085253e-05 + T_*(7.743602693621120145e-07 + T_*(-9.204198773051246169e-09 + T_*(-2.122032402927605809e-10 + T_*(4.125625669666862445e-12 + T_*(3.532509087599244118e-14 + T_*(-5.242786376794922436e-16 + T_*(-7.260942673196442122e-18 + T_*(-2.453561164912172901e-19 + T_*(5.678598204071161723e-21 + T_*(1.229928951189520298e-22 + T_*(-2.566095852346387466e-24 + T_*(-2.594462117958716581e-26 + T_*(6.198016439919091221e-28 + T_*(4.079274536668629507e-30 + T_*(-8.523972978819286856e-32 + T_*(-6.333007168641729819e-34 + T_*(4.884637765078024090e-36 + T_*(5.970619077275256973e-38 + T_*(1.583552627627211185e-40)))))))))))))))))))) + Tref*(6.709096824546971884e-05 + T_*(1.164430354507516326e-06 + T_*(1.492519185739810694e-08 + T_*(-1.004992535578806199e-10 + T_*(-5.194638284127568879e-12 + T_*(-8.852593174458557238e-14 + T_*(7.743098463983663378e-15 + T_*(-1.349430816115996717e-17 + T_*(-5.111520085301861189e-18 + T_*(7.246506699900768993e-20 + T_*(1.688571447195582101e-21 + T_*(-4.458601255074950933e-23 + T_*(-3.906571883790175152e-25 + T_*(1.401012855147709264e-26 + T_*(1.049655774799796372e-28 + T_*(-2.326684433917799696e-30 + T_*(-2.546715297351952591e-32 + T_*(1.076294752796537571e-34 + T_*(2.697444862568323708e-36 + T_*(1.404413350272535558e-38 + T_*(2.444355701979738372e-41)))))))))))))))))))) + Tref*(5.609138211497678676e-07 + T_*(2.619248932484586790e-08 + T_*(1.268762007103502264e-10 + T_*(-1.010471605550804941e-12 + T_*(-1.709919997846759906e-13 + T_*(-9.956527977396212394e-16 + T_*(2.437200472916002895e-16 + T_*(-2.053684743415526158e-18 + T_*(-1.347065496266485984e-19 + T_*(2.110916506052379395e-21 + T_*(3.809682178285610452e-23 + T_*(-7.906486212697025412e-25 + T_*(-6.716781189658700860e-27 + T_*(1.545503563949260188e-28 + T_*(8.638277498899926506e-31 + T_*(-1.671970193463004779e-32 + T_*(-7.800495852299937013e-35 + T_*(9.773608301957742171e-37 + T_*(3.864485572169008074e-39 + T_*(-3.451329681590587444e-41 + T_*(-1.629936596105260491e-43)))))))))))))))))))) + Tref*(6.211050901462248372e-09 + T_*(3.874480667788168084e-10 + T_*(4.106851040300304659e-12 + T_*(-3.105169520741854561e-13 + T_*(6.522151398403065119e-16 + T_*(3.096283820459271858e-16 + T_*(-6.565926448866785715e-18 + T_*(-7.159450456455795076e-20 + T_*(4.894802418930987747e-21 + T_*(-6.001966927419548388e-23 + T_*(-1.513821153252045092e-24 + T_*(5.010070342130555091e-26 + T_*(3.162510217391503999e-28 + T_*(-1.770123180458273006e-29 + T_*(-1.053932297446173471e-31 + T_*(3.190827677655366054e-33 + T_*(3.180222107117949408e-35 + T_*(-1.648380290063498673e-37 + T_*(-3.706141650512771631e-39 + T_*(-1.917722264772635693e-41 + T_*(-3.362382426227650908e-44)))))))))))))))))))) + Tref*(1.263124849842055366e-10 + T_*(6.187412516523896449e-12 + T_*(4.856354852070540130e-14 + T_*(-9.099759444145425678e-15 + T_*(1.560190728375411453e-16 + T_*(7.844891934613453307e-18 + T_*(-3.045057457877816053e-19 + T_*(-2.301735187073350059e-22 + T_*(1.799768784739026120e-22 + T_*(-2.234656647103105002e-24 + T_*(-5.049062278648236422e-26 + T_*(1.234519814239864004e-27 + T_*(8.987122286706115913e-30 + T_*(-3.505893794831842691e-31 + T_*(-1.936267714478340583e-33 + T_*(5.615005967424700712e-35 + T_*(4.748626410980998077e-37 + T_*(-3.117187102765338827e-39 + T_*(-5.342048836982600040e-41 + T_*(-2.315823054010398044e-43 + T_*(-3.233422625384509680e-46)))))))))))))))))))) + Tref*(3.041933860088385168e-12 + T_*(7.093111999258697384e-14 + T_*(-1.844005764063040847e-15 + T_*(-3.012726184142518609e-18 + T_*(2.286804695745121757e-18 + T_*(-3.778946134964007883e-20 + T_*(-5.425688203547919150e-22 + T_*(1.720852091689116264e-23 + T_*(-2.132216029860537651e-25 + T_*(1.204097053514651607e-26 + T_*(9.674019166203633406e-29 + T_*(-1.310039385511127070e-29 + T_*(-3.441825993234164085e-32 + T_*(5.218559560287548561e-33 + T_*(2.792178449174765620e-35 + T_*(-9.909642936552052681e-37 + T_*(-1.027812531335395456e-38 + T_*(4.888180488713776030e-41 + T_*(1.228573493776009552e-42 + T_*(6.780669148124561191e-45 + T_*(1.273888703944060394e-47)))))))))))))))))))) + Tref*(4.269738908941117406e-14 + T_*(3.515691990558187008e-16 + T_*(-5.752406629775988591e-17 + T_*(2.935526497578685499e-18 + T_*(-1.396603096459919851e-20 + T_*(-3.076116403302916373e-21 + T_*(9.652802795654607107e-23 + T_*(9.479133640033158372e-26 + T_*(-6.298473009854447451e-26 + T_*(1.067398063166126933e-27 + T_*(1.747803610841354067e-29 + T_*(-6.553290148516108139e-31 + T_*(-3.304812034827328800e-33 + T_*(2.060835499621757621e-34 + T_*(1.070192724105747808e-36 + T_*(-3.514326016203337634e-38 + T_*(-3.229557610146469936e-40 + T_*(1.843938327239782823e-42 + T_*(3.752745971817626646e-44 + T_*(1.855294075718714751e-46 + T_*(3.111545317890359977e-49)))))))))))))))))))) + Tref*(3.187227602061177940e-16 + T_*(-1.653226486329436251e-18 + T_*(-5.503279650803687541e-19 + T_*(3.910889420859907195e-20 + T_*(-4.933223004188857164e-22 + T_*(-3.385127545168397706e-23 + T_*(1.323951437519895638e-24 + T_*(-2.175849457167253713e-27 + T_*(-7.767216266915100570e-28 + T_*(1.278597695041199424e-29 + T_*(2.085697980946302978e-31 + T_*(-6.977592433892922353e-33 + T_*(-3.774593366845591632e-35 + T_*(2.047178660783037856e-36 + T_*(1.050350400037973286e-38 + T_*(-3.355921373609339394e-40 + T_*(-2.957951585520604123e-42 + T_*(1.782466345297969279e-44 + T_*(3.384503662089840092e-46 + T_*(1.597754239120040422e-48 + T_*(2.533589690591361309e-51)))))))))))))))))))) + Tref*(9.638094015677524082e-19 + T_*(-1.825497478946932691e-20 + T_*(-1.687807799941253959e-21 + T_*(1.540362530250114278e-22 + T_*(-2.627978436697326027e-24 + T_*(-1.158894467996326174e-25 + T_*(5.277085797941572453e-27 + T_*(-1.725975868117368293e-29 + T_*(-2.913361578529600527e-30 + T_*(4.827457622910831529e-32 + T_*(7.681289388780240843e-34 + T_*(-2.437302887117087984e-35 + T_*(-1.370426782411350938e-37 + T_*(6.806940090614902811e-39 + T_*(3.495435520132081072e-41 + T_*(-1.081253869553308734e-42 + T_*(-9.298263251666167061e-45 + T_*(5.750522446000603233e-47 + T_*(1.045544411570244632e-48 + T_*(4.767127037305669495e-51 + T_*(7.211593011699459480e-54))))))))))))))))))))))))))))))  # noqa: E501

    # Return theta-w converted to K
    return thw + 273.15

def pseudoadiabat_temp(p,thw):
    """
    Function taken from https://github.com/robwarrenwx/atmos/blob/main/atmos/pseudoadiabat.py
    (2024/04/02).

    Computes the temperature T at pressure p on a pseudoadiabat with
    wet-bulb potential temperature thw.

    Uses polynomial approximations from Moisseeva and Stull (2017) with
    revised coefficients.

    Moisseeva, N. and Stull, R., 2017. A noniterative approach to
        modelling moist thermodynamics. Atmospheric Chemistry and
        Physics, 17, 15037-15043.

    Arguments:
    - p (pressure; Pa)
    - thw (wet-bulb potential temperature; K)

    Returns:
    - T (temperature; K)
    """

    # Convert p to hPa and theta-w to degC
    p_ = p / 100.
    thw_ = thw - 273.15

    # Check that values fall in the permitted range
    condition = (np.any(thw_ < -70.) or np.any(thw_ > 50.) or np.any(p_ > 1100.) or np.any(p_ < 50.))
    if condition:
        return np.nan

    # Compute T using Eq. 1-3 from Moisseeva & Stull 2017
    thref = -1.958881611671661176e+02 + p_*(2.134884082821395079e+00 + p_*(-2.651475098307509368e-02 + p_*(2.861864119262733791e-04 + p_*(-2.298718394514268143e-06 + p_*(1.360057184923542422e-08 + p_*(-5.958196441636455271e-11 + p_*(1.938675375399162892e-13 + p_*(-4.665355621127693766e-16 + p_*(8.139597343471045903e-19 + p_*(-9.718027816571788133e-22 + p_*(6.514043622263483823e-25 + p_*(4.795894401108516600e-29 + p_*(-5.561331861642867047e-31 + p_*(4.256943236052971359e-34 + p_*(1.115187417957733097e-37 + p_*(-4.675607928134105329e-40 + p_*(4.189061674074321886e-43 + p_*(-1.989920659873727387e-46 + p_*(5.148437033814795851e-50 + p_*(-5.751272231517078191e-54))))))))))))))))))))  # noqa: E501
    T = -2.899089457107268331e+01 + thw_*(1.337227498242554491e+00 + thw_*(9.989220649709655911e-03 + thw_*(-5.289649585393284086e-05 + thw_*(-8.125516739581656903e-06 + thw_*(-1.669385809901756079e-07 + thw_*(3.902176729685648592e-09 + thw_*(2.785299448741561866e-10 + thw_*(1.199597501486574654e-12 + thw_*(-2.356495994204141054e-13 + thw_*(-3.754462622941184458e-15 + thw_*(1.109955443870932428e-16 + thw_*(2.958323602057082693e-18 + thw_*(-2.247001341245910925e-20 + thw_*(-1.185942142170470679e-21 + thw_*(-2.645697164120065566e-24 + thw_*(2.354624142321289223e-25 + thw_*(2.070711502559931296e-27 + thw_*(-1.458747561161565743e-29 + thw_*(-2.729648310305078289e-31 + thw_*(-1.030941535866486469e-33)))))))))))))))))))) + thref*(1.429869503550506904e+00 + thw_*(-7.879837833208863662e-03 + thw_*(-6.838366952421416926e-04 + thw_*(-1.598425851503747948e-05 + thw_*(2.249449259819928238e-07 + thw_*(3.397632056104877195e-08 + thw_*(7.042819999431954275e-10 + thw_*(-3.396305216284396052e-11 + thw_*(-1.427634441882554734e-12 + thw_*(1.718717725756761431e-14 + thw_*(1.351301204465966675e-15 + thw_*(-6.192018154861673091e-19 + thw_*(-7.436786948388566283e-19 + thw_*(-4.586031806307956854e-21 + thw_*(2.361621265751940082e-22 + thw_*(2.687010240026768440e-24 + thw_*(-3.700784758878172927e-26 + thw_*(-6.641106252235517576e-28 + thw_*(9.656001298499274765e-31 + thw_*(6.328645165577936637e-32 + thw_*(2.937789149798092732e-34)))))))))))))))))))) + thref*(5.040685977297330346e-03 + thw_*(-5.183478788794109284e-04 + thw_*(-8.614096880135471002e-06 + thw_*(6.838202302696602762e-07 + thw_*(4.744824589422048218e-08 + thw_*(4.117928705641196483e-10 + thw_*(-8.162969260401781373e-11 + thw_*(-2.506138383042399551e-12 + thw_*(6.408005845436154616e-14 + thw_*(3.422786758033482118e-15 + thw_*(-1.423870299706560812e-17 + thw_*(-2.347455893945982803e-18 + thw_*(-1.318085039469029337e-20 + thw_*(8.499042116124600158e-22 + thw_*(1.076807733584116111e-23 + thw_*(-1.338371226030613206e-25 + thw_*(-3.037207845698305137e-27 + thw_*(-2.280138198346503550e-30 + thw_*(2.951988306149798322e-31 + thw_*(2.237589772936888619e-33 + thw_*(4.618048611010569571e-36)))))))))))))))))))) + thref*(-1.693011573882441043e-04 + thw_*(-4.055505851887614839e-06 + thw_*(7.152523677763036439e-07 + thw_*(3.843257858703898518e-08 + thw_*(-5.844911310861310370e-10 + thw_*(-1.223121149866247755e-10 + thw_*(-1.922842158181238273e-12 + thw_*(1.860182907731545371e-13 + thw_*(5.585246327323852151e-15 + thw_*(-1.464431913286146844e-16 + thw_*(-6.559251795406730752e-18 + thw_*(4.899420092423094807e-20 + thw_*(4.199638168089739444e-21 + thw_*(7.896679384714861248e-24 + thw_*(-1.500870337486252050e-24 + thw_*(-1.219254495390460205e-26 + thw_*(2.649010861349269856e-28 + thw_*(3.763581851733364165e-30 + thw_*(-1.106962190807747265e-32 + thw_*(-4.004411643912248489e-34 + thw_*(-1.755651685348803267e-36)))))))))))))))))))) + thref*(-3.362873566158531171e-06 + thw_*(2.400942971282232631e-07 + thw_*(2.216103471465453210e-08 + thw_*(-2.358530382561856690e-10 + thw_*(-9.126436748430261388e-11 + thw_*(-1.931576827275146865e-12 + thw_*(1.740569037403095845e-13 + thw_*(6.367717531052867911e-15 + thw_*(-1.593228821759510933e-16 + thw_*(-8.645773823963254084e-18 + thw_*(5.183005959622386943e-20 + thw_*(6.216660172191152019e-21 + thw_*(2.280562673183416492e-23 + thw_*(-2.407055696000981067e-24 + thw_*(-2.497159694469028016e-26 + thw_*(4.310619140953843283e-28 + thw_*(7.767424624269287546e-30 + thw_*(-8.253668170995669652e-33 + thw_*(-8.265671014355856887e-34 + thw_*(-5.055332555603654578e-36 + thw_*(-6.866406703219498400e-39)))))))))))))))))))) + thref*(3.313006618121957636e-08 + thw_*(6.298294495254139702e-09 + thw_*(-1.508115170726802090e-10 + thw_*(-3.666058876697772524e-11 + thw_*(-6.929713447725405319e-13 + thw_*(1.001411867284538917e-13 + thw_*(3.999009200007800670e-15 + thw_*(-1.355140171228429127e-16 + thw_*(-7.891853868577618701e-18 + thw_*(8.201739075969596332e-20 + thw_*(8.157502196248929472e-21 + thw_*(1.872664694326145830e-24 + thw_*(-4.816139246993863831e-24 + thw_*(-3.251690295448622985e-26 + thw_*(1.596069969864303561e-27 + thw_*(1.911924811476329357e-29 + thw_*(-2.539650134731972660e-31 + thw_*(-4.766708118214056693e-33 + thw_*(5.905961735438100206e-36 + thw_*(4.542824189463173739e-37 + thw_*(2.146594442973875009e-39)))))))))))))))))))) + thref*(1.827646537676912968e-09 + thw_*(2.700624495348298076e-11 + thw_*(-1.693998063665866103e-11 + thw_*(-6.246467298925717754e-13 + thw_*(5.645404365776630832e-14 + thw_*(3.192962277411950210e-15 + thw_*(-8.101287993813953261e-17 + thw_*(-7.067390583742533053e-18 + thw_*(2.929925634625895208e-20 + thw_*(8.199180601829400231e-21 + thw_*(5.440328877403649776e-23 + thw_*(-5.211512128625378953e-24 + thw_*(-7.469192185405930370e-26 + thw_*(1.680194632500845717e-27 + thw_*(3.931613027193546259e-29 + thw_*(-1.570489589795977390e-31 + thw_*(-9.529584442335513237e-33 + thw_*(-4.564784845461340239e-35 + thw_*(7.845559059350718099e-37 + thw_*(9.429015911623818610e-39 + thw_*(2.969487264396173626e-41)))))))))))))))))))) + thref*(2.692489627401177037e-11 + thw_*(-8.866612549898191130e-13 + thw_*(-3.282133330510417204e-13 + thw_*(-2.176991467782678476e-15 + thw_*(1.525548282484946781e-15 + thw_*(3.255140898734812961e-17 + thw_*(-3.448238972506286259e-18 + thw_*(-9.819269470584681398e-20 + thw_*(4.103973882000636586e-21 + thw_*(1.421360768154288777e-22 + thw_*(-2.535930235573404443e-24 + thw_*(-1.138555920321440711e-25 + thw_*(6.357268121704545670e-28 + thw_*(5.168500299704030730e-29 + thw_*(8.966189300449154540e-32 + thw_*(-1.256207628001013337e-32 + thw_*(-8.343615181202799912e-35 + thw_*(1.324806169302695400e-36 + thw_*(1.409632581667273272e-38 + thw_*(-1.806982775263108798e-41 + thw_*(-4.102485528397047362e-43)))))))))))))))))))) + thref*(2.012525693579932692e-13 + thw_*(-1.409558681656955683e-14 + thw_*(-3.000478150434391517e-15 + thw_*(3.523413088054992200e-17 + thw_*(1.627592243174956061e-17 + thw_*(1.005263036983865846e-19 + thw_*(-4.256579318654112554e-20 + thw_*(-5.804877230475824782e-22 + thw_*(5.995951096343076474e-23 + thw_*(1.084604700494159426e-24 + thw_*(-4.758795393781846959e-26 + thw_*(-1.049404375984382912e-27 + thw_*(2.091294597612180069e-29 + thw_*(5.744517949902291411e-31 + thw_*(-4.386198422439627512e-33 + thw_*(-1.770452206683057335e-34 + thw_*(9.429173370089740969e-38 + thw_*(2.809453846841474016e-38 + thw_*(1.073364281318929982e-40 + thw_*(-1.723453344911116331e-42 + thw_*(-1.112769846261706298e-44)))))))))))))))))))) + thref*(7.785302726588919706e-16 + thw_*(-8.115422322369204709e-17 + thw_*(-1.359423616814330821e-17 + thw_*(3.494943093543307934e-19 + thw_*(8.118345806601954328e-20 + thw_*(-3.050691842782283411e-22 + thw_*(-2.304670197233203703e-22 + thw_*(-1.264239996599368699e-24 + thw_*(3.525780381700034179e-25 + thw_*(3.805080939256278211e-27 + thw_*(-3.081460621592729418e-28 + thw_*(-4.551988505380969105e-30 + thw_*(1.546975182182166271e-31 + thw_*(2.902341615631649949e-33 + thw_*(-4.149081096195920384e-35 + thw_*(-1.029080159047627957e-36 + thw_*(4.190976007612783375e-39 + thw_*(1.906088654394316995e-40 + thw_*(3.603634349189407222e-43 + thw_*(-1.430304392526457524e-44 + thw_*(-8.040338359871879916e-47)))))))))))))))))))) + thref*(1.239375080712697757e-18 + thw_*(-1.697253721082437881e-19 + thw_*(-2.457949980434437093e-20 + thw_*(9.128133448732742170e-22 + thw_*(1.568961363107645293e-22 + thw_*(-1.757664150666109165e-24 + thw_*(-4.703796383698215715e-25 + thw_*(2.947044230184508861e-29 + thw_*(7.581119354391254214e-28 + thw_*(4.806360277319273094e-30 + thw_*(-7.004664901833034528e-31 + thw_*(-7.627496323163632015e-33 + thw_*(3.757741880027517288e-34 + thw_*(5.601780195611557017e-36 + thw_*(-1.108150827912297810e-37 + thw_*(-2.199929486827426309e-39 + thw_*(1.405470896802169264e-41 + thw_*(4.459709728573167858e-43 + thw_*(3.991616449307566712e-46 + thw_*(-3.666103046079535075e-47 + thw_*(-1.943440793139680257e-49))))))))))))))))))))))))))))))  # noqa: E501

    # Return T converted to K
    return T + 273.15

def follow_moist_adiabat(pi,pf,Ti):
    """
    Cut down version of function from https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py
    (2024/04/02).

    Computes parcel temperature following a saturated adiabat or pseudoadiabat.
    For descending parcels, a pseudoadiabat is always used. By default,
    pseudoadiabatic calculations use polynomial fits for fast calculations, but
    can optionally use slower iterative method. Saturated adiabatic ascent must
    be performed iteratively (for now). At present, polynomial fits are only
    available for liquid-only pseudoadiabats.

    Arguments:
    - pi (initial pressure; Pa)
    - pf (final pressure; Pa)
    - Ti (initial temperature; K)

    Returns:
    - Tf (final temperature; K)
    """

    # Compute the wet-bulb potential temperature of the pseudoadiabat
    # that passes through (pi, Ti)
    thw = pseudoadiabat_wbpt(pi, Ti)

    # Compute the temperature on this pseudoadiabat at pf
    Tf = pseudoadiabat_temp(pf, thw)

    return Tf

def dewpoint_temperature_from_relative_humidity(T, RH):
    """
    Function taken from https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py
    (2024/04/02).

    Computes dewpoint temperature from temperature and relative humidity over
    liquid water using equations from Romps (2021).

    Arguments:
    - T (temperature; K)
    - RH (relative humidity; fraction)

    Returns:
    - Td (dewpoint temperature; K)
    """

    # Set constant (Romps 2021, Eq. 6)
    c = (Lv0 - (cpv - cpl) * T0) / ((cpv - cpl) * T)

    # Compute dewpoint temperature (Romps 2021, Eq. 5)
    fn = np.power(RH, (Rv / (cpl - cpv))) * c * np.exp(c)
    W = lambertw(fn, k=-1).real
    Td = c * (1 / W) * T

    return Td

def saturation_specific_humidity(p, T):
    """
    Taken from https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py
    (2024/04/02).

    Computes saturation specific humidity from pressure and temperature.

    Arguments:
    - p (pressure; Pa)
    - T (temperature; K)

    Returns:
    - qs (float or ndarray): saturation specific humidity (kg/kg)
    """

    # Compute latent heat of vaporisation
    Lv = Lv0 + (cpv - cpl) * (T - T0) # # taken from thermo.py/latent_heat_of_vaporisation  

    # Compute SVP over liquid water (Ambaum 2020, Eq. 13)
    es = es0 * np.power((T0 / T), ((cpl - cpv) / Rv)) * \
        np.exp((Lv0 / (Rv * T0)) - (Lv / (Rv * T)))
    
    # compute saturation specific humidity
    qs = eps * es / (p - (1 - eps) * es)

    return qs

def Twb_thermo_func(T,RH,q,P,Twb_method):
    """
    thermo wet bulb temperature calculation. 
    Uses adiabatic_wet_bulb_temperature ("NEWT") or isobaric_wet_bulb_temperature from 
    https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py revision:94bc441
    All calculations assume liquid phase.

    Adiabatic (or pseudo) wet-bulb temperature is the temperature of a parcel
    of air lifted adiabatically to saturation and then brought
    pseudoadiabatically at saturation back to its original pressure. It is
    always less than the isobaric wet-bulb temperature.

    Isobaric wet-bulb temperature is the temperature of a parcel of air cooled
    isobarically to saturation via the evaporation of water into it, with all
    latent heat supplied by the parcel. It is always greater than the adiabatic
    wet-bulb temperature. Isobaric wet-bulb temperature is similar (but not
    identical) to the quantity measured by a wet-bulb thermometer. 

    See https://glossary.ametsoc.org/wiki/Wet-bulb_temperature.
    
    Notes from a communication with Rob Warren:
    - In the WBGT equation isobaric Twb (rather than adiabatic Twb) makes sense to use physically since humans/wet bulbs are cooling through evaporation at 
      constant pressure
    - However, isobaric Twb is only a small fraction of a degree different from adiabatic Twb, particularly for larger RH values 
      (thermo adiabatic Twb is smaller than thermo isobaric Twb, but only by <0.5 degC for RH>30%) 
    - In the literature, Twb in the WBGT equation is commonly referred to as the natural (unventilated) Twb. This is also the thermodynamic (or isobaric) Twb.
      Another type of Twb which should not be used is the psychometric (or aspirated) Twb.
      (https://uploads-ssl.webflow.com/5dcd0bc3137dcb207a26c19a/5dce09a27dad12e3934ef849_v02-calculation-of-the-natural-unventilated-wet-bulb-temperature.pdf)
    - Stull (2011) tends to overestimate Twb at high T and low/moderate RH by ~1 degC
    - thermo_isobaric is recommended but may take longer as unliken thermo_adiabatic and Stull it is iterative

    Arguments:
    - T (air temperature; degC)
    - RH (relative humidity; %)
    - q (specifc humidity; kg kg-1)
    - P (surface pressure; Pa)
    - Twb_method (whether to use `thermo_adiabatic` or `thermo_isobaric`)

    Returns:
    - Twb (wet bulb temperature; degC)
    """

    # Precision for iterative temperature calculations (K)
    precision = 0.001

    # Maximum number of iterations for iterative calculations
    max_n_iter = 20

    # convert to Kelvin and fraction
    T = T + 273.15
    RH = RH / 100.

    if Twb_method == 'thermo_adiabatic': # taken from adiabatic_wet_bulb_temperature

        # Get pressure and temperature at the LCL
        p_lcl, T_lcl = lifting_condensation_level(P, T, q, RH) 

        # Follow a pseudoadiabat from the LCL to the original pressure
        Twb = follow_moist_adiabat(p_lcl, P, T_lcl)

    elif Twb_method == 'thermo_isobaric': # taken from thermo.py/isobaric_wet_bulb_temperature

        # Compute dewpoint temperature
        Td = dewpoint_temperature_from_relative_humidity(T, RH)

        # Initialise Twb as mean of T and Td
        Twb = (T + Td) / 2

        # Compute the latent heat at temperature T
        Lv_T = Lv0 + (cpv - cpl) * (T - T0) # taken from thermo.py/latent_heat_of_vaporisation    

        # Iterate to convergence
        converged = False
        count = 0
        while not converged:

            # Update the previous Twb value
            Twb_prev = Twb

            # Compute the ice fraction at Twb
            omega_Twb = 0.0

            # Compute saturation specific humidity at Twb
            qs_Twb = saturation_specific_humidity(P, Twb)

            # Compute the effective specific heat at qs(Twb)
            cpm_qs_Twb = cpm = (1 - qs_Twb) * cpd + qs_Twb * cpv # taken from thermo.py/effective_specific_heat

            # Compute the latent heat of vaporisation at Twb
            Lv_Twb = Lv0 + (cpv - cpl) * (Twb - T0) # taken from thermo.py/latent_heat_of_vaporisation 

            # Compute the derivative of qs with respect to Twb
            dqs_dTwb = qs_Twb * (1 + qs_Twb / eps - qs_Twb) * Lv_Twb / (Rv * Twb**2)

            # Compute f and f'
            f = cpm_qs_Twb * (T - Twb) - Lv_T * (qs_Twb - q)
            fprime = ((cpv - cpd) * (T - Twb) - Lv_T) * dqs_dTwb - cpm_qs_Twb
         
            # Update Twb using Newton's method
            Twb = Twb - f / fprime

            # Check for convergence
            if np.max(np.abs(Twb - Twb_prev)) < precision:
                converged = True
            else:
                count += 1
                if count == max_n_iter:
                    print("Twb not converged after 20 iterations")
                    break
    
    else:
        sys.exit("!!! Twb_thermo_func -- Twb_method should be thermo_adibatic or thermo_isobaric !!!")

    # Return Twb converted to degC
    return Twb - 273.15

def Twb_func(T,RH,q,P,Twb_method):
    """
    Calculate wet bulb temperature using Stull (2011) or 
    thermo https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py.

    Arguments:
    - T (air temperature; degC)
    - RH (relative humidity; %)
    - q (specifc humidity; kg kg-1)
    - P (surface pressure; Pa)
    - Twb_method (whether to use `thermo_adiabatic`, `thermo_isobaric`, or `Stull` (2011) method)

    Returns:
    - Twb (wet bulb temperature; degC)
    """

    if Twb_method == 'thermo_adiabatic' or Twb_method == 'thermo_isobaric':
        Twb = Twb_thermo_func(T,RH,q,P,Twb_method)
    elif Twb_method == 'Stull':
        Twb = Twb_Stull_func(T,RH)
    else:
        sys.exit("!!! Twb_func -- Twb_method should be thermo_adibatic, thermo_isobaric, or Stull !!!")

    return Twb

def zen_and_cazi(lat,lon,elevation,dateandtime,canyon_orient_deg,tzinfo):
    """
    Calculates solar zenith angle and solar azimuth angle (degrees clockwise from North).
    Calculates the canyon azimuth angle -- the difference between the solar azimuth angle 
    and canyon orientation angle.

    Arguments:
    - lat (latitudes; deg in WGS84)
    - lon (longitudes; deg in WGS84)
    - elevation (elevations; m in WGS84)
    - dateandtime (contains the time in datetime format)
    - canyon_orient_deg (canyon orientation angle defined as the horizontal angle measured clockwise from north to a
      line running prarallel to the alignment of the street canyon; deg)
    - tzinfo (the timezone; https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)

    Returns:
    - zen (solar zenith angle; rad)
    - cazi (canyon azimuth angle; rad)
    - solar_zen_deg (solar zenith angle; deg)
    - solar_azi_deg (solar azimuth angle; deg)
    - canyon_azi_deg (canyon azimuth angle; deg)
    """
    
    dateandtime = dateandtime.replace(tzinfo=pytz.timezone(tzinfo))

    if np.isscalar(lat) or isinstance(lat, np.ndarray) and lat.ndim == 1:
        lat_flat = np.atleast_1d(lat)
        lon_flat = np.atleast_1d(lon)
        elevation_flat = np.atleast_1d(elevation)
    else:
        lat_flat = lat.flatten()
        lon_flat = lon.flatten()
        elevation_flat = elevation.flatten()

    location = pvlib.location.Location(latitude=lat_flat,
                                        longitude=lon_flat,
                                        altitude=elevation_flat)

    solar_position = location.get_solarposition([dateandtime] * len(lat_flat))

    if isinstance(lat, np.ndarray) and lat.ndim == 2:
        solar_zen_deg = solar_position['zenith'].values.reshape(lat.shape)
        solar_azi_deg = solar_position['azimuth'].values.reshape(lat.shape)
    else:
        solar_zen_deg = solar_position['zenith'].values
        solar_azi_deg = solar_position['azimuth'].values

    zen = solar_zen_deg*2.*np.pi/360. # to radians
    canyon_azi_deg = solar_azi_deg-canyon_orient_deg
    canyon_azi_deg[canyon_azi_deg<0] = canyon_azi_deg[canyon_azi_deg<0] + 360.
    cazi = canyon_azi_deg*2.*np.pi/360. # to radians

    return zen, cazi, solar_zen_deg, solar_azi_deg, canyon_azi_deg

def prevent_X_in_wall(X,W):
    """
    Prevent X from being in the wall.

    Arguments:
    - W (canyon width; m)
    - X (black globe horizontal distance right of the centre line; m)

    Returns:
    - X (black globe horizontal distance right of the centre line; m)
    """

    if not isinstance(X, np.ndarray):
        X = np.array([X])
    X[X == W/2.] -= 0.0001
    X[X == -W/2.] += 0.0001
    if np.any(np.logical_or(X > W/2., X < -W/2.)):
        sys.exit("!!! prevent_X_in_wall -- X should be -W/2 < X < W/2 !!!")
    else:
        pass
    
    return X

def point2facet_view_fracs(H,W,Z,X,geometry_choice):
    """
    Viewing fractions of each of the facets as viewed from the globe.

    Arguments:
    - H (canyon height; m)
    - W (canyon width; m)
    - Z (black globe height; m)
    - X (black globe horizontal distance right of the centre line; m)
    - geometry_choice (whether to use: `flat` or `canyon` geometry)

    Returns:
    - Fs (sky viewing fraction from the black globe; -)
    - Fr (road viewing fraction from the black globe; -)
    - Fw (two wall viewing fraction from the black globe; -)
    """

    if geometry_choice == "canyon":
        ap = (W+2*X)/(2*(H-Z))
        am = (W-2*X)/(2*(H-Z))
        bp = (W+2*X)/(2*Z)
        bm = (W-2*X)/(2*Z)
        Fs = 1/(2*np.pi)*(np.arctan(ap)+np.arctan(am))
        Fr = 1/(2*np.pi)*(np.arctan(bm)+np.arctan(bp))
        Fw = 1/(2*np.pi)*(np.arctan(1/am)+np.arctan(1/bm)+np.arctan(1/bp)+np.arctan(1/ap))
    elif geometry_choice == "flat":
        Fs = 0.5
        Fr = 0.5
        Fw = 0.
    else:
        sys.exit("!!! point2facet_view_fracs -- geometry_choice can only be `canyon` or `flat` !!!")

    return Fs, Fr, Fw

def facet2facet_view_fracs(H,W,geometry_choice):
    """
    Viewing fractions of each of the facets as viewed from each facet.

    Arguments:
    - H (canyon height; m)
    - W (canyon width; m)
    - geometry_choice (whether to use: `flat` or `canyon` geometry)

    Returns:
    - Fsr (fraction of sky viewed from the road; -)
    - Frs (fraction of road viewed from the sky; -)
    - Fww (fraction of wall viewed from the wall; -)
    - Fwr (fraction of wall viewed from the road; -)
    - Fws (fraction of wall viewed from the sky; -)
    - Frw (fraction of road viewed from the wall; -)
    - Fsw (fraction of sky viewed from the wall; -)
    """

    if geometry_choice == "canyon":
        Fsr = Frs = np.sqrt(1+(H/W)**2)-H/W
        Fww = np.sqrt(1+(W/H)**2)-W/H
        Fwr = Fws = 0.5*(1-Fww)
        Frw = Fsw = 0.5*(1-Fsr)
    elif geometry_choice == "flat":
        Fsr = Frs = 0.5
        Fww = Fwr = Fws = Frw = Fsw = 0.
    else:
        sys.exit("!!! facet2facet_view_fracs -- geometry_choice can only be `canyon` or `flat` !!!")

    return Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw

def direct_illuminated_fractions(H,W,cazi,zen,geometry_choice):
    """
    Fraction of wall and road illuminated by the direct beam.
    Depending on azi the direct beam might illuminate none of the wall,
    part of the wall, or all of the wall and part of the road.

    Arguments:
    - H (canyon height; m)
    - W (canyon width; m)
    - cazi (canyon azimuth angle; rad)
    - zen (solar zenith angle; rad)
    - geometry_choice (whether to use: `flat` or `canyon` geometry)

    Returns:
    - fr (fraction of the road illuminated by the direct beam; -)
    - fw (fraction of the wall illuminated by the direct beam; -)
    """

    if geometry_choice == "canyon":
        fr = 1 - (H*np.tan(zen)*np.absolute(np.cos(cazi)))/W
        fw = W/(H*np.tan(zen)*np.absolute(np.cos(cazi)))
        fr[fr < 0] = 0.
        fr[fr > 1] = 1.
        fw[fw < 0] = 0.
        fw[fw > 1] = 1.
        fr[zen > 90*np.pi/180.] = 0.
    elif geometry_choice == "flat":
        fr = np.ones(np.shape(H))
        fw = np.zeros(np.shape(H))
    else:
        sys.exit("!!! partial2facet_view_fracs -- geometry_choice can only be `canyon` or `flat` !!!")

    return fr, fw

def point2partial_view_fracs(H,W,Z,X,fr,fw,cazi,geometry_choice):
    """
    Partial illumination viewing fractions of each of the facets as viewed from the black globe.

    Arguments:
    - H (canyon height; m)
    - W (canyon width; m)
    - Z (black globe height; m)
    - X (black globe horizontal distance right of the centre line; m)
    - fr, fw (fraction of the road/wall illuminated by the direct beam; -)
    - cazi (canyon azimuth angle; rad)
    - geometry_choice (whether to use: `flat` or `canyon` geometry)

    Returns:
    - Fpr (partial viewing fraction of the road; -)
    - Fpw (partial viewing fraction of the wall; -)
    """

    if geometry_choice == "canyon":
        indices = np.absolute(cazi) > np.pi/2
        X[indices] = -X[indices]
        Fpr = 1/(2*np.pi)*(np.arctan((W+2*X)/(2*Z))-np.arctan(((1-2*fr)*W+2*X)/(2*Z)))
        Fpw = 1/(2*np.pi)*(np.arctan((2*(H-Z))/(W+2*X))-np.arctan(2*(((1-fw)*H)-Z)/(W+2*X)))
    elif geometry_choice == "flat":
        Fpr = 0.5
        Fpw = 0.
    else:
        sys.exit("!!! partial2facet_view_fracs -- geometry_choice can only be `canyon` or `flat` !!!")

    return Fpr, Fpw

def partial2facet_view_fracs(H,W,fr,fw,geometry_choice):
    """
    Viewing fractions of each of the facets as viewed from the partially illuminated facet.

    Arguments:
    - H (canyon height; m)
    - W (canyon width; m)
    - fr, fw (fraction of the road/wall illuminated by the direct beam; -)
    - geometry_choice (whether to use: `flat` or `canyon` geometry)

    Returns:
    - Fprw1 (view from partially illuminated road of the partially illuminated wall; -)
    - Fprw2 (view from partially illuminated road of the opposite wall; -)
    - Fprs (view from partially illuminated road of the sky; -)
    - Fpw1r (view from partially illuminated wall of the road; -)
    - Fpw1w2 (view from partially illuminated wall of the opposite wall; -)
    - Fpw1s (view from partially illuminated wall of the sky; -)
    """

    if (type(fr) == float) or (fr.ndim == 0):
        fr = np.array(fr,ndmin=1)
    if (type(fw) == float) or (fw.ndim == 0):
        fw = np.array(fw,ndmin=1)
    if (type(H) == float) or (H.ndim == 0):
        H = np.array(H,ndmin=1)
    if (type(W) == float) or (W.ndim == 0):
        W = np.array(W,ndmin=1)
    Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s = np.zeros(np.shape(fr)), np.zeros(np.shape(fr)), \
                                               np.zeros(np.shape(fr)), np.zeros(np.shape(fr)), \
                                               np.zeros(np.shape(fr)), np.zeros(np.shape(fr))
    if geometry_choice == "canyon":
        # wall partially illuminated 
        m = fw!=0
        H_, W_, fr_, fw_ = np.copy(H[m]), np.copy(W[m]), \
            np.copy(fr[m]), np.copy(fw[m])
        np.putmask(Fpw1r,m,0.5+(np.sqrt(W_**2+(1-fw_)**2*H_**2)-np.sqrt(H_**2+W_**2))/(2*fw_*H_)) # proportion to road
        np.putmask(Fpw1w2,m,(np.sqrt(W_**2+fw_**2*H_**2)-np.sqrt(W_**2+(1-fw_)**2*H_**2)+np.sqrt(H_**2+W_**2)-W_)/(2*fw_*H_)) # proportion to shaded wall
        np.putmask(Fpw1s,m,0.5+(W_-np.sqrt(W_**2+fw_**2*H_**2))/(2*fw_*H_)) # proportion to sky
        # road partially illuminated
        m = fr!=0
        H_, W_, fr_, fw_ = np.copy(H[m]), np.copy(W[m]), \
            np.copy(fr[m]), np.copy(fw[m])
        np.putmask(Fprw1,m,0.5+(H_-np.sqrt(H_**2+fr_**2*W_**2))/(2*fr_*W_)) # proportion to illuminated wall
        np.putmask(Fprw2,m,0.5+(np.sqrt(H_**2+(1-fr_)**2*W_**2)-np.sqrt(H_**2+W_**2))/(2*fr_*W_)) # proportion to shaded wall
        np.putmask(Fprs,m,(np.sqrt(H_**2+fr_**2*W_**2)-np.sqrt(H_**2+(1-fr_)**2*W_**2)+np.sqrt(H_**2+W_**2)-H_)/(2*fr_*W_)) # proportion to sky
    elif geometry_choice == "flat":
        pass
    else:
        sys.exit("!!! partial2facet_view_fracs -- geometry_choice can only be `canyon` or `flat` !!!")

    return Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s

def SrSw_func(Id,alb_grnd,alb_wall,zen,cazi):
    """
    Calculates diffuse fluxes from the road and wall that originate from the direct beam.

    Arguments:
    - Id (downwelling shortwave direct radiation; W m-2)
    - alb_grnd (albedo of the ground; -)
    - alb_wall (albedo of the walls; -)
    - zen (solar zenith angle; rad)
    - cazi (canyon azimuth angle; rad)

    Returns:
    - Sr (diffuse flux from the road; W m-2)
    - Sw (diffuse flux from the wall; W m-2)
    """

    Sr = Id*alb_grnd
    Sw = Id*alb_wall*np.tan(zen)*np.absolute(np.cos(cazi))

    return Sr, Sw

def K_func(Fs,Fr,Fw,Fsr,Frs,Fww,Fwr,Fws,Frw,Fsw,alb_grnd,alb_wall,Kd,Sr,Sw,Fpr,Fpw,Fprw1,Fprw2,Fpw1r,Fpw1w2,nref):
    """
    Calculates the sum of the diffuse components. 

    Arguments:
    - Fs, Fr, Fw (sky/road/wall viewing fraction from the black globe; -)
    - Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw (fraction of x viewed from y where x and y can be road r, wall w, or sky s, and Fxy; -)
    - alb_grnd (albedo of the ground; -)
    - alb_wall (albedo of the walls; -)
    - Kd (downwelling shortwave diffuse radiation; W m-2)
    - Sr, Sw (diffuse flux from the road/wall; W m-2)
    - Fpr, Fpw (partial viewing fraction of the road/wall; -)
    - Fprw1, Fprw2, Fpw1r, Fpw1w2 (view from partially illuminated x of the partially illuminated y where x and y can be road r, wall w, or sky s, and Fxy; -)
    - nref (defines the number of shortwave diffuse reflections as nref and the number of shortwave direct reflections as nref+1)

    Returns:
    - Ks (diffuse from sky; W m-2)
    - Kr (direct to diffuse from partially illuminated road; W m-2)
    - Kw (direct to diffuse from partially illuminated wall; W m-2)
    - K (the sum; W m-2)
    """

    Ks = Ks_func(Fs,Fr,Fw,Fsr,Frs,Fww,Fwr,Fws,Frw,Fsw,alb_grnd,alb_wall,Kd,nref)
    Kr, Kw =  KrKw_func(Sr,Sw,Fr,Fw,Fpw,Fpr,Fww,Fwr,Frw,Fprw1,Fprw2,Fpw1r,Fpw1w2,alb_grnd,alb_wall,nref)
    K = Ks + Kr + Kw

    return K, Ks, Kr, Kw

def Ks_func(Fs,Fr,Fw,Fsr,Frs,Fww,Fwr,Fws,Frw,Fsw,alb_grnd,alb_wall,Kd,nref):
    """
    Diffuse radiation received by the black globe that originates from 
    downward diffuse radiation from the sky. For 0, 1, and 2 reflections.

    Arguments:
    - Fs, Fr, Fw (sky/road/wall viewing fraction from the black globe; -)
    - Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw (fraction of x viewed from y where x and y can be road r, wall w, or sky s, and Fxy; -)
    - alb_grnd (albedo of the ground; -)
    - alb_wall (albedo of the walls; -)
    - Kd (downwelling shortwave diffuse radiation; W m-2)
    - nref (defines the number of shortwave diffuse reflections as nref and the number of shortwave direct reflections as nref+1)

    Returns:
    - Ks (diffuse from sky; W m-2)
    """

    if nref == 0:
        Ks = Kd*Fs
    elif nref == 1:
        Ks = Kd*(Fs+Fsr*alb_grnd*Fr+Fsw*alb_wall*Fw)
    elif nref == 2:
        Ks = Kd*(Fs+(Fsr*alb_grnd+2*Fsw*Fwr*alb_wall*alb_grnd)*Fr+\
                (Fsw*alb_wall+Fsw*Fww*alb_wall**2+Fsr*Frw*alb_grnd*alb_wall)*Fw)
    else:
        sys.exit("!!! Ks_func -- nref can only be `0`, `1`, or `2` !!!")

    return Ks

def KrKw_func(Sr,Sw,Fr,Fw,Fpw,Fpr,Fww,Fwr,Frw,Fprw1,Fprw2,Fpw1r,Fpw1w2,alb_grnd,alb_wall,nref):
    """
    Diffuse radiation received by the black globe that originates from the 
    direct beam, and whose final reflection was from the road or wall.

    Arguments:
    - Sr, Sw (diffuse flux from the road/wall; W m-2)
    - Fr, Fw (road/wall viewing fraction from the black globe; -)
    - Fpr, Fpw (partial viewing fraction of the road/wall; -)
    - Fww, Fwr, Frw (fraction of x viewed from y where x and y can be road r, wall w, or sky s, and Fxy; -)
    - Fprw1, Fprw2, Fpw1r, Fpw1w2 (view from partially illuminated x of the partially illuminated y where x and y can be road r, wall w, or sky s, and Fxy; -)
    - alb_grnd (albedo of the ground; -)
    - alb_wall (albedo of the walls; -)
    - nref (defines the number of shortwave diffuse reflections as nref and the number of shortwave direct reflections as nref+1)

    Returns:
    - Kr (direct to diffuse from partially illuminated road; W m-2)
    - Kw (direct to diffuse from partially illuminated wall; W m-2)
    """

    if nref == 0:
        Kr = Sr*Fpr
        Kw = Sw*Fpw
    elif nref == 1:
        Kr = Sr*(Fpr+(Fprw1*alb_wall+Fprw2*alb_wall)*Fw/2)  
        Kw = Sw*(Fpw+Fpw1w2*Fw/2+Fpw1r*alb_grnd*Fr) 
    elif nref == 2:
        Kr = Sr*(Fpr+((Fprw1*alb_wall+Fprw2*alb_wall+Fprw1*alb_wall**2*Fww+Fprw2*alb_wall**2*Fww)*Fw/2+\
                        (Fprw1*alb_wall*Fwr*alb_grnd+Fprw2*alb_wall*Fwr*alb_grnd)*Fr))    
        Kw = Sw*(Fpw+((Fpw1w2+2*Fpw1r*alb_grnd*Frw*alb_wall+Fpw1w2*alb_wall**2*Fww)*Fw/2+\
                (Fpw1r*alb_grnd+Fpw1w2*alb_wall*Fwr*alb_grnd)*Fr))
    else:
        sys.exit("!!! KrKw_func -- nref can only be `0`, `1`, or `2` !!!")

    return Kr, Kw

def gamma_func(gamma,LAI,gamma_choice):
    """
    Direct beam gamma attenuation factor calculation.

    Arguments:
    - gamma (direct beam attenuation factor; -)
    - LAI (direct beam attenuation factor; -)
    - gamma_choice (whether to use: a gamma that is `prescribed` or modelled using `LAI`)

    Returns:
    - gamma (direct beam attenuation factor; -)
    """

    if gamma_choice == 'prescribe':
        pass
    elif gamma_choice == 'LAI':
        gamma = np.exp(-0.5*LAI)
    else:
        sys.exit("!!! gamma_func -- gamma_choice can only be `prescribe` or `LAI` !!!")

    return gamma

def I_func(Id,H,W,Z,X,zen,cazi,gamma,geometry_choice):
    """
    Shortwave direct beam radiation at black globe calculation.

    Arguments:
    - Id (downwelling shortwave direct radiation; W m-2)
    - H (canyon height; m)
    - W (canyon width; m)
    - Z (black globe height; m)
    - X (black globe horizontal distance right of the centre line; m)
    - zen (solar zenith angle; rad)
    - cazi (canyon azimuth angle; rad)
    - gamma (direct beam attenuation factor; -)
    - gamma_choice (whether to use: a gamma that is `prescribed` or modelled using `LAI`)

    Returns:
    - I (shortwave direct radiation received by the black globe; W m-2)
    """

    if geometry_choice == "flat":
        cos_zen = np.cos(zen)
        cos_zen[cos_zen<0.1] = 0.1
        I = gamma*Id/(4*cos_zen) # max stops blow up at small zenith angles
    elif geometry_choice == "canyon":
        indices = np.absolute(cazi) > np.pi
        X[indices] = -X[indices]
        beta = np.arctan((W-2*X)/(2*(H-Z)*np.absolute(np.cos(cazi-np.pi/2))))
        cos_zen = np.cos(zen)
        cos_zen[cos_zen<0.1] = 0.1
        I = gamma*Id/(4*cos_zen) # max stops blow up at small zenith angles
        if type(I) == np.float64:
            I = np.array([I])
        I[zen>=beta] = 0. # if Sun below buildings then I=0
    else:
        sys.exit("!!! I_func -- geometry_choice can only be `flat` or `canyon` !!!")

    return I

def L_func(T_grnd,T_wall,Ld,emiss_grnd,emiss_wall,Fr,Fw,Fs,emiss_g,T,WBGT_model_choice):
    """
    Longwave radiative flux incident on the black globe.

    Arguments:
    - T_grnd (surface temperature of the gound; degC)
    - T_wall (surface temperature of the walls; degC)
    - Ld (downwelling longwave radiation; W m-2)
    - emiss_grnd (emissivity of the ground; -)
    - emiss_wall (emissivity of the walls; -)
    - Fs, Fr, Fw (sky/road/wall viewing fraction from the black globe; -)
    - emiss_g (emissivity of the black globe; -)
    - T (air temperature; degC)
    - WBGT_model_choice (which model to use: `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` models)

    Returns:
    - L (longwave radiation incident on the black globe; W m-2)
    """

    if WBGT_model_choice == 'UCanWBGT_outdoor':
        T_grnd = np.copy(T_grnd+273.15)
        T_wall = np.copy(T_wall+273.15)
        L = emiss_grnd*sboltz*T_grnd**4*Fr + emiss_wall*sboltz*T_wall**4*Fw + Ld*Fs
    elif WBGT_model_choice == 'UCanWBGT_indoor':
        T = np.copy(T+273.15)
        L = sboltz*T**4
    else:   
        sys.exit("!!! WBGT_model_choice can only be `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` !!!") 

    return L

def MRT_func(T,L,K,I,a_g,emiss_g,WBGT_model_choice):
    """
    Mean radiant temperature calculation.

    Arguments:
    - T (air temperature; degC)
    - L (longwave radiation incident on the black globe; W m-2)
    - K (shortwave diffuse radiation incident on the black globe; W m-2)
    - I (shortwave direct radiation incident on the black globe; W m-2)
    - a_g (absorptivity of the black globe; -)
    - emiss_g (emissivity of the black globe; -)
    - WBGT_model_choice (which model to use: `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` models)

    Returns:
    - MRT (mean radiant temperature; equivalent to T_{MRT} in Shonk et al. (XXX); degC)
    """

    if WBGT_model_choice == "UCanWBGT_indoor": # assume walls, air, and black globe are equal (T=MRT=Tg)
        MRT = np.copy(T)
    elif WBGT_model_choice == "UCanWBGT_outdoor":
        Qenv = a_g*(L+K+I) 
        MRT = (Qenv/(emiss_g*sboltz))**0.25 - 273.15
    else:
        sys.exit("!!! MRT_func -- should only be called if WBGT_model_choice is `UCanWBGT_indoor` or `UCanWBGT_outdoor` !!!")

    return MRT

def Tg_func(T,MRT,WS,d,emiss_g,WBGT_model_choice):
    """
    Black globe temperature calculation. Leroyer (2018).

    Arguments:
    - T (air temperature; degC)
    - MRT (mean radiant temperature of the black globe; degC)
    - WS (wind speed; m s-1)
    - d (diameter of the black globe; m)
    - emiss_g (emissivity of the black globe; -)
    - WBGT_model_choice (which model to use: `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` models)

    Returns:
    - Tg (black globe temperature; degC)
    """

    if WBGT_model_choice == "UCanWBGT_indoor": # assume walls, air, and black globe are equal (T=MRT=Tg)
        Tg = np.copy(T)
    elif WBGT_model_choice == "UCanWBGT_outdoor": # outdoor assume forced convection. Solution from Leroyer et al. (2018) Appendix A.
        hcg = 6.3*WS**0.6/d**0.4 
        a = hcg/(emiss_g*sboltz)
        b = (MRT+273.15)**4 + a*(T+273.15)
        M = 9.*a**2
        N = 27.*a**4
        P = 256.*b**3
        E =(M + 1.73205*np.sqrt(N+P))**(1/3)
        Q = 3.4943*b
        K = 0.381571*E - Q/E
        I = 0.5*((2*a/np.sqrt(K))-K)**0.5
        J = np.sqrt(K)/2.
        Tg = I-J-273.15
    else:
        sys.exit("!!! Tg_func -- should only be called if WBGT_model_choice is `UCanWBGT_indoor` or `UCanWBGT_outdoor` !!!")

    return Tg

def WBGT_func(T,Twb,Tg,RH,L,K,I,a_g,a_LW,a_SW,WBGT_model_choice,WBGT_equation_choice):
    """
    Calculation of WBGT for one of:
    UCanWBGT_outdoor, ISO_outdoor, simple_outdoor, UCanWBGT_indoor, or ISO_indoor.

    Arguments:
    - T (air temperature; degC)
    - Twb (wet bulb temperature; degC)
    - Tg (black globe temperature; degC)
    - RH (relative humidity; %)
    - L (longwave radiation incident on the black globe; W m-2)
    - K (shortwave diffuse radiation incident on the black globe; W m-2)
    - I (shortwave direct radiation incident on the black globe; W m-2)
    - a_g (absorptivity of the black globe; -)
    - a_SW (shortwave absorptivity of a human; -)
    - a_LW (longwave absorptivity of a human; -)
    - WBGT_model_choice (which model to use: `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` models)
    - WBGT_equation_choice (whether to use: the `full` or approximate `ISO` WBGT equation)

    Returns:
    - WBGT (wet bulb globe temperature; degC)
    """

    if WBGT_model_choice == "UCanWBGT_outdoor" or WBGT_model_choice == "UCanWBGT_indoor": 
        if WBGT_equation_choice == "full":
            # Parsons (2006) Eq. 3 which reduces to the ISO standard equations in SW and LW limits
            r =  (K+I)/(L+K+I)
            a_h = (1-r)*a_LW + r*a_SW
            WBGT =  0.7*Twb + 0.3*(a_h*a_g*(Tg-T)+T)
        elif WBGT_equation_choice == "ISO":
            if WBGT_model_choice == "UCanWBGT_outdoor": 
                # ISO standard 100% shortwave
                WBGT = 0.7*Twb + 0.2*Tg + 0.1*T
            elif WBGT_model_choice == "UCanWBGT_indoor":  
                # ISO standard 100% longwave
                WBGT = 0.7*Twb + 0.3*Tg
            else:
                sys.exit("!!! WBGT_func -- WBGT_model_choice can only be `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` !!!")
        else:
            sys.exit("!!! WBGT_func -- WBGT_equation_choice can only be `full` or `ISO` !!!")
    elif WBGT_model_choice == "simpleWBGT": 
        # simplified WBGT Liu et al. (2018)
        es = 6.108*np.exp((17.27*T/(T+237.3))) # saturated water vapour pressure
        ea = es*RH/100 # water vapour pressure
        WBGT = 0.567*T + 0.393*ea + 3.94
    else:
        sys.exit("!!! WBGT_func -- WBGT_model_choice can only be `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` !!!")

    return WBGT

def main(
    T=None, T_grnd=None, T_wall=None, RH=None, q=None, WS=None, P=None, Ld=None, Kd=None, Id=None,\
    gamma=None, LAI=None,\
    H=None, W=None, Z=None, X=None, tile_number=None, tf=None, canyon_orient_deg=None,\
    a_SW=None, a_LW=None,\
    alb_grnd=None, alb_wall=None, emiss_grnd=None, emiss_wall=None,\
    emiss_g=None, a_g=None, d=None,\
    lat=None, lon=None, elevation=None, dateandtime=None, tzinfo=None,\
    WBGT_model_choice=None, geometry_choice=None, nref=None,\
    gamma_choice=None, WBGT_equation_choice=None, Twb_method=None 
    ):
    """
    main of UCanWBGT module.

    Arguments:
    - T (air temperature; -; float/1D/2D; degC)
    - T_grnd (surface temperature of the gound; equivalent to Tr in Shonk et al. (XXX) when the ground is road; float/1D/2D; degC)
    - T_wall (surface temperature of the walls; equivalent to Tw in Shonk et al. (XXX); float/1D/2D; degC)
    - RH (relative humidity; -; float/1D/2D; %)
    - q (specific humidity; -; float/1D/2D; kg kg-1)
    - WS (wind speed; -; float/1D/2D; m s-1)
    - P (surface pressure; -; float/1D/2D; Pa)
    - Ld (downwelling longwave radiation; equivalent to L_\{downarrow} in Shonk et al. (XXX); float/1D/2D; W m-2)
    - Kd (downwelling shortwave diffuse radiation; equivalent to K_\{downarrow} in Shonk et al. (XXX); float/1D/2D; W m-2)
    - Id (downwelling shortwave direct radiation; equivalent to I_\{downarrow} in Shonk et al. (XXX); float/1D/2D; W m-2)
    - gamma (direct beam attenuation factor; -; float/1D/2D; -)
    - LAI (direct beam attenuation factor; -; float/1D/2D; -)
    - H (canyon height; -; float/1D/2D; m)
    - W (canyon width; -; float/1D/2D; m)
    - Z (black globe height; -; float/1D/2D; m)
    - X (black globe horizontal distance right of the centre line; -; float/1D/2D; m)
    - tile_number (tile number; -; int; 0-9)
    - tf (tile fraction; -; float/1D/2D; -)
    - canyon_orient_deg (canyon orientation angle defined as the horizontal angle measured clockwise from north to a
      line running parallel to the alignment of the street canyon; -; float/1D/2D; deg)
    - a_SW (shortwave absorptivity of a human; -; float; -)
    - a_LW (longwave absorptivity of a human; -; float; -)
    - alb_grnd (albedo of the ground; equivalent to \alpha_r in Shonk et al. (XXX) when the ground is road; float/1D/2D; -)
    - alb_wall (albedo of the walls; equivalent to \alpha_w in Shonk et al. (XXX); float/1D/2D; -)
    - emiss_grnd (emissivity of the ground; equivalent to \epsilon_r in Shonk et al. (XXX) when the ground is road; float/1D/2D; -)
    - emiss_wall (emissivity of the walls; equivalent to \epsilon_w in Shonk et al. (XXX); float/1D/2D; -)
    - emiss_g (emissivity of the black globe; equivalent to \epsilon_g in Shonk et al. (XXX); float; -)
    - a_g (absorptivity of the black globe; -; float; -)
    - d (diameter of the black globe; -; float; m)
    - lat (latitudes; -; float/1D/2D; deg in WGS84)
    - lon (longitudes; -; float/1D/2D; deg in WGS84)
    - elevation (elevations; -; float/1D/2D; m in WGS84)
    - dateandtime (contains the time in datetime format)
    - tzinfo (the timezone; https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
    - WBGT_model_choice (which model to use: `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` models)
    - geometry_choice (whether to use: `flat` or `canyon` geometry)
    - nref (defines the number of shortwave diffuse reflections as nref and the number of shortwave direct reflections as nref+1)
    - gamma_choice (whether to use: a gamma that is `prescribed` or modelled using `LAI`)
    - WBGT_equation_choice (whether to use: the `full` or approximate `ISO` WBGT equation)
    - Twb_method (whether to use: `Stull` (2011), or the `thermo_isobaric` or `thermo_adiabatic` 
      https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py Twb method)

    Returns:
    - Twb (wet bulb temperature; -; float/1D/2D; degC)
    - solar_zen_deg (solar zenith angle; equivalent to \theta_0 in Shonk et al. (XXX); float/1D/2D; deg)
    - solar_azi_deg (solar azimuth angle; -; float/1D/2D; deg)
    - canyon_azi_deg (canyon azimuth angle; equivalent to \phi_0 in Shonk et al. (XXX); float/1D/2D; deg)
    - Fs, Fr, Fw (sky/road/wall viewing fraction from the black globe; -; float/1D/2D; -)
    - Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw (fraction of x viewed from y where x and y can be road r, wall w, or sky s, and Fxy; -; float/1D/2D; -)
    - fr, fw (fraction of the road/wall illuminated by the direct beam; -; float/1D/2D; -)
    - Fpr, Fpw (partial viewing fraction of the road/wall; -; float/1D/2D; -)
    - Fprw1, Fprw2, Fpw1r, Fpw1w2 (view from partially illuminated x of the partially illuminated y where x and y can be road r, wall w, or sky s, and Fxy; -; float/1D/2D; -)
    - Sr, Sw (diffuse flux from the road/wall; -; float/1D/2D; W m-2)
    - K, Ks, Kr, Kw (shortwave diffuse total/sky/road/wall; -; float/1D/2D; W m-2)
    - I (shortwave direct radiation received by the black globe; -; float/1D/2D; W m-2)
    - L (longwave radiation incident on the black globe; -; float/1D/2D; W m-2)
    - MRT (mean radiant temperature of the black globe; equivalent to T_{MRT} in Shonk et al. (XXX); float/1D/2D; degC)
    - Tg (black globe temperature; -; float/1D/2D; degC)
    - WBGT (wet bulb globe temperature; -; float/1D/2D; degC)
    """

    # make sure lon/lat are arrays
    if isinstance(lon, float):
        lon = np.array([lon])
    if isinstance(lat, float):
        lat = np.array([lat])

    # ensure all have the same shape
    array_or_float_list = T, T_grnd, T_wall, RH, q, WS, P, Ld, Kd, Id, gamma, LAI, H, W, Z, X, tf, canyon_orient_deg, a_SW, a_LW,\
    alb_grnd, alb_wall, emiss_grnd, emiss_wall, emiss_g, a_g, d, elevation
    T, T_grnd, T_wall, RH, q, WS, P, Ld, Kd, Id, gamma, LAI, H, W, Z, X, tf, canyon_orient_deg, a_SW, a_LW,\
    alb_grnd, alb_wall, emiss_grnd, emiss_wall, emiss_g, a_g, d, elevation = adjust_array_shape(array_or_float_list,lon)

    # remove values with zero tile fraction and convert to 1D
    array_list = [T, T_grnd, T_wall, RH, q, WS, P, Ld, Kd, Id, gamma, LAI, H, W, Z, X, tf, canyon_orient_deg, a_SW, a_LW,\
    alb_grnd, alb_wall, emiss_grnd, emiss_wall, emiss_g, a_g, d, elevation, lon, lat]  
    mask = tf == 0.
    array_list_1d = []
    for i, arr in enumerate(array_list):
        processed_array = mask_and_1d(arr,mask)
        array_list_1d.append(processed_array)
    T, T_grnd, T_wall, RH, q, WS, P, Ld, Kd, Id, gamma, LAI, H, W, Z, X, tf, canyon_orient_deg, a_SW, a_LW,\
    alb_grnd, alb_wall, emiss_grnd, emiss_wall, emiss_g, a_g, d, elevation, lon, lat = array_list_1d

    # find if there is a mixture of canyon and flat
    # make a dictionary containing canyon, flat or both, containing the variables split in two
    # loop over the number of keys
        # run the routine saving the output to a dictionary
    # combine the arrays back into one 1d array if both

    # loop over the code twice if some values are canyon and flat
    # split the 1d arrays into two 1d arrays where True and False
    # - if tile != 8 or 9 then set geometry_choice = flat
    # - if tile == 8 or 9 then run twice ((i) and (ii))
        # - and (i) H = 0 then set geometry_choice = flat
        # - and (ii) H > 0 and Z < H then set geometry_choice = canyon
        # - and (iii) H > 0 but Z > H then set to flat and give a warning (globe cannot be above canyon)
    
    #if (tile_number == 8) or (tile_number == 9):
    #    geometry_choice = 'canyon'
    #else:
    #    geometry_choice = 'flat'

    # Calculate RH and q if they do not already exist
    if RH is not None and q is not None:
        pass
    elif RH is not None and q is None:
        if P is None:
            sys.exit("!!! If q is not provided then P must be !!!")            
        q = q_from_RH(T,RH,P)
    elif RH is None and q is not None:
        if P is None:
            sys.exit("!!! If RH is not provided then P must be !!!")
        RH = RH_from_q(T,q,P)
    else:
        sys.exit("!!! Neither RH or q provided !!!")
        
    # Twb calculation
    if WBGT_model_choice == "UCanWBGT_outdoor" or WBGT_model_choice == "UCanWBGT_indoor":
        Twb = Twb_func(T,RH,q,P,Twb_method)
    elif WBGT_model_choice == "simpleWBGT":
        Twb = None
    else:   
        sys.exit("!!! WBGT_model_choice can only be `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` !!!")

    # shortwave calculations
    if WBGT_model_choice == "UCanWBGT_outdoor":
        zen, cazi, solar_zen_deg, solar_azi_deg, canyon_azi_deg = zen_and_cazi(lat,lon,elevation,dateandtime,canyon_orient_deg,tzinfo)
        X = prevent_X_in_wall(X,W)
        Fs, Fr, Fw = point2facet_view_fracs(H,W,Z,X,geometry_choice)
        Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw = facet2facet_view_fracs(H,W,geometry_choice)
        fr, fw = direct_illuminated_fractions(H,W,cazi,zen,geometry_choice)
        Fpr, Fpw = point2partial_view_fracs(H,W,Z,X,fr,fw,cazi,geometry_choice)
        Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s = partial2facet_view_fracs(H,W,fr,fw,geometry_choice)
        Sr, Sw = SrSw_func(Id,alb_grnd,alb_wall,zen,cazi)
        K, Ks, Kr, Kw = K_func(Fs,Fr,Fw,Fsr,Frs,Fww,Fwr,Fws,Frw,Fsw,alb_grnd,alb_wall,Kd,Sr,Sw,Fpr,Fpw,Fprw1,Fprw2,Fpw1r,Fpw1w2,nref)
        gamma = gamma_func(gamma,LAI,gamma_choice)
        I = I_func(Id,H,W,Z,X,zen,cazi,gamma,geometry_choice)
    elif WBGT_model_choice == "UCanWBGT_indoor":
        solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, fr, fw, Fpr, Fpw, Fprw1, Fprw2, \
            Fpw1r, Fpw1w2, Sr, Sw, K, Ks, Kr, Kw, I = \
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, \
                None, None, None, 0., 0., 0., 0., 0.
    elif WBGT_model_choice == "simpleWBGT":
        solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, fr, fw, Fpr, Fpw, Fprw1, Fprw2, \
            Fpw1r, Fpw1w2, Sr, Sw, K, Ks, Kr, Kw, I = \
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, \
                None, None, None, None, None, None, None, None
    else:
        sys.exit("!!! main -- WBGT_model_choice can only be `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` !!!")

    # longwave calculation
    if WBGT_model_choice == "UCanWBGT_outdoor" or WBGT_model_choice == "UCanWBGT_indoor":
        L = L_func(T_grnd,T_wall,Ld,emiss_grnd,emiss_wall,Fr,Fw,Fs,emiss_g,T,WBGT_model_choice)
    elif WBGT_model_choice == "simpleWBGT":
        L = None
    else:   
        sys.exit("!!! main -- WBGT_model_choice can only be `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` !!!")    

    # mean radiant temperature and black globe temperature calculation
    if WBGT_model_choice == "UCanWBGT_outdoor" or WBGT_model_choice == "UCanWBGT_indoor":
        MRT = MRT_func(T,L,K,I,a_g,emiss_g,WBGT_model_choice)
        Tg = Tg_func(T,MRT,WS,d,emiss_g,WBGT_model_choice)
    elif WBGT_model_choice == "simpleWBGT":
        MRT = None
        Tg = None
    else:   
        sys.exit("!!! main -- WBGT_model_choice can only be `UCanWBGT_outdoor`, `UCanWBGT_indoor`, or `simpleWBGT` !!!")
    
    # WBGT calculation                                           
    WBGT = WBGT_func(T,Twb,Tg,RH,L,K,I,a_g,a_LW,a_SW,WBGT_model_choice,WBGT_equation_choice)

    # if originally 2D then convert from 1D back to 2D (and fill with zeros where tf = 0)
    array_or_float_list = Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
            fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s, Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT
    Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
            fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s, Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT = put_array_shape_back(array_or_float_list, mask)


    return Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
            fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fprs, Fpw1r, Fpw1w2, Fpw1s, Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT

if __name__ == "__main__":
    # This block ensures that main() is called only when the script is executed directly
    main()

