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

def adjust_array_shape(array_or_float, reference):
    """
    Adjust the shape of the input array or float to match the shape of the reference array.
    If the input is a float, it will be repeated to create an array with the same shape as the reference.

    Arguments:
    - array_or_float (The array or float; numpy.ndarray or float)  
    - reference (The reference array or float; numpy.ndarray or float)

    Returns:
    - array_or_float (The adjusted array or float; numpy.ndarray or float)     
    """

    if isinstance(array_or_float, np.ndarray) and isinstance(reference, np.ndarray):
        if array_or_float.shape != reference.shape:
            return array_or_float
    elif isinstance(array_or_float, float) and hasattr(reference, 'shape'):
        return np.full_like(reference, array_or_float)
    
    return array_or_float

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
    Taken from https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py
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
    Taken from https://github.com/robwarrenwx/atmos/blob/main/atmos/pseudoadiabat.py
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
    Tref = 5.545194269969744028e+01 + p_*(-6.160922643206258043e-01 + p_*(7.587811691476412448e-03 + p_*(-8.979212604246789573e-05 + p_*(7.343181162799252997e-07 + p_*(-4.203811338238339756e-09 + p_*(1.680461676878753079e-11 + p_*(-4.663028376804742115e-14 + p_*(8.792987053284386843e-17 + p_*(-1.047685061993484244e-19 + p_*(5.803471746171544039e-23 + p_*(2.719984341254661375e-26 + p_*(-6.275663461476053709e-29 + p_*(1.325022483616707296e-32 + p_*(4.338742237555935124e-35 + p_*(-3.090431264151296911e-38 + p_*(-2.003198659255209158e-41 + p_*(4.151977079563261066e-44 + p_*(-2.671125528966892761e-47 + p_*(8.274902032133455127e-51 + p_*(-1.046220932987381136e-54))))))))))))))))))))  # noqa: E501
    thw = 4.232940944232577607e+01 + T_*(5.718057512715307711e-01 + T_*(6.676677597885334776e-03 + T_*(4.022328480366987597e-05 + T_*(-9.508933019160445564e-07 + T_*(-7.872837881810170976e-09 + T_*(1.764981358907121684e-10 + T_*(2.337314128547671647e-12 + T_*(-3.546237055245935149e-14 + T_*(-6.653359895702736545e-16 + T_*(5.416100578454232859e-18 + T_*(1.736801137333358149e-19 + T_*(-1.614941755022829075e-22 + T_*(-3.466397463900693713e-23 + T_*(-1.954775222373624531e-25 + T_*(3.917994330304135765e-27 + T_*(5.062948051510256424e-29 + T_*(-4.425505843709759732e-32 + T_*(-4.099457794462657487e-33 + T_*(-2.659946576039995724e-35 + T_*(-5.616686551472560432e-38)))))))))))))))))))) + Tref*(3.623382401650120932e-01 + T_*(1.023543556426264028e-03 + T_*(7.651560288975466182e-05 + T_*(-3.333050342938257735e-07 + T_*(-1.155276051647147619e-08 + T_*(5.929972799181639469e-11 + T_*(3.206717543550327453e-12 + T_*(-2.806173962086747008e-15 + T_*(-5.124959937929081926e-16 + T_*(-1.855121616602876157e-17 + T_*(9.821368425617630502e-20 + T_*(1.227932266156539288e-20 + T_*(-7.683681814109954133e-24 + T_*(-4.104781036660294525e-24 + T_*(-1.459303936349590696e-26 + T_*(7.195188421056683942e-28 + T_*(6.308529814583321843e-30 + T_*(-3.728722888842601765e-32 + T_*(-7.651013597331745427e-34 + T_*(-3.900687460769850595e-36 + T_*(-6.838418124460199041e-39)))))))))))))))))))) + Tref*(2.901311619805157452e-03 + T_*(6.485698819407039969e-05 + T_*(7.728091544277208254e-07 + T_*(-9.073366523455469539e-09 + T_*(-2.141374965578981410e-10 + T_*(3.977216429327843141e-12 + T_*(4.122840262503433229e-14 + T_*(-5.256646599001689577e-16 + T_*(-1.145434112370376915e-17 + T_*(-1.781311250561409403e-19 + T_*(7.157635199010274020e-21 + T_*(8.010843921970355959e-23 + T_*(-2.967462937546381677e-24 + T_*(-1.231180428827811944e-26 + T_*(7.411974321317298560e-28 + T_*(1.871989870585702621e-30 + T_*(-1.139473166366657039e-31 + T_*(-5.550450159434532228e-34 + T_*(7.751320464793886659e-36 + T_*(7.636712088445193183e-38 + T_*(1.904910678489293722e-40)))))))))))))))))))) + Tref*(6.704619097436729624e-05 + T_*(1.163029842424645954e-06 + T_*(1.492287334118758421e-08 + T_*(-9.803556343362327257e-11 + T_*(-5.321938015151650545e-12 + T_*(-8.750285129654440476e-14 + T_*(7.883547117854788030e-15 + T_*(-1.969218043870929035e-17 + T_*(-5.127731809674121555e-18 + T_*(7.842500025808431793e-20 + T_*(1.656834067909569676e-21 + T_*(-4.771060315991388879e-23 + T_*(-3.789330073585334101e-25 + T_*(1.503827044450349823e-26 + T_*(1.063631966232871725e-28 + T_*(-2.515473868945231314e-30 + T_*(-2.685544596332960150e-32 + T_*(1.185962932727380623e-34 + T_*(2.890950955818082802e-36 + T_*(1.500601905017281505e-38 + T_*(2.612549730540674067e-41)))))))))))))))))))) + Tref*(5.622752095209163301e-07 + T_*(2.622299773664040882e-08 + T_*(1.305336393284960921e-10 + T_*(-1.276960779075757062e-12 + T_*(-1.683548375977049797e-13 + T_*(-6.271150950803480645e-16 + T_*(2.315028031157249709e-16 + T_*(-2.146347028376092958e-18 + T_*(-1.245693873642227093e-19 + T_*(2.045375912756655024e-21 + T_*(3.384747838730439315e-23 + T_*(-7.402265189368336296e-25 + T_*(-5.511288828625452761e-27 + T_*(1.398311724960150295e-28 + T_*(5.941982483198068820e-31 + T_*(-1.481882184010474742e-32 + T_*(-3.464262458502697741e-35 + T_*(9.867780531525398202e-37 + T_*(6.017124351539231275e-40 + T_*(-5.595813431706730079e-41 + T_*(-2.065635042206331018e-43)))))))))))))))))))) + Tref*(6.310399231532088346e-09 + T_*(3.901231962638336016e-10 + T_*(4.180122829870311996e-12 + T_*(-3.192353977095813438e-13 + T_*(9.004741911651702877e-16 + T_*(3.149258318289430785e-16 + T_*(-7.025210952408942387e-18 + T_*(-6.361263629427768113e-20 + T_*(5.124872290307491148e-21 + T_*(-7.054075409523944098e-23 + T_*(-1.553787992819717914e-24 + T_*(5.585474476756643726e-26 + T_*(3.240541368795211673e-28 + T_*(-1.953300445074685362e-29 + T_*(-1.131145285513222428e-31 + T_*(3.507374001441693019e-33 + T_*(3.473614062208678184e-35 + T_*(-1.806024510910001329e-37 + T_*(-4.054752894421383860e-39 + T_*(-2.101330765235148035e-41 + T_*(-3.695013142657365282e-44)))))))))))))))))))) + Tref*(1.271719593514502614e-10 + T_*(6.212652945668079643e-12 + T_*(4.718325172207772923e-14 + T_*(-9.049900408478350932e-15 + T_*(1.581335742011448829e-16 + T_*(7.664434600667353565e-18 + T_*(-3.032068138918179548e-19 + T_*(-2.954470910946507580e-23 + T_*(1.764470276812099321e-22 + T_*(-2.362502682406429885e-24 + T_*(-4.810519050498481162e-26 + T_*(1.293799446056060717e-27 + T_*(8.263862859728115334e-30 + T_*(-3.703475842686161888e-31 + T_*(-1.875235811381363166e-33 + T_*(5.999930281812112887e-35 + T_*(4.929459653637661333e-37 + T_*(-3.378361077097880353e-39 + T_*(-5.686371346389453665e-41 + T_*(-2.469609650556775141e-43 + T_*(-3.483253301441309733e-46)))))))))))))))))))) + Tref*(3.006868042816497974e-12 + T_*(6.996687305913599728e-14 + T_*(-1.902923934541224379e-15 + T_*(2.538423830307459436e-18 + T_*(2.175603590409096192e-18 + T_*(-4.327281625931597660e-20 + T_*(-2.645614098017516364e-22 + T_*(1.558894913470146477e-23 + T_*(-3.942387509153589096e-25 + T_*(1.606988956610959619e-26 + T_*(1.540754199772559106e-28 + T_*(-1.543468479057767344e-29 + T_*(-4.922832778131969665e-32 + T_*(5.943402637170408262e-33 + T_*(3.300968515126008702e-35 + T_*(-1.109485447615582067e-36 + T_*(-1.161425587366964164e-38 + T_*(5.384616121931944142e-41 + T_*(1.368792887313021082e-42 + T_*(7.550551701298664713e-45 + T_*(1.415937577614429599e-47)))))))))))))))))))) + Tref*(4.182255148516593461e-14 + T_*(3.264676233541051175e-16 + T_*(-5.820220450542899159e-17 + T_*(3.026293244201152484e-18 + T_*(-1.682107528877679333e-20 + T_*(-3.123096672164356461e-21 + T_*(1.015072165615557285e-22 + T_*(-1.617560469774974781e-27 + T_*(-6.536703646025585238e-26 + T_*(1.186271179408631216e-27 + T_*(1.783041190999098461e-29 + T_*(-7.191842428389848484e-31 + T_*(-3.361306213352488339e-33 + T_*(2.262717601618277057e-34 + T_*(1.148352949368445723e-36 + T_*(-3.863263832687730034e-38 + T_*(-3.541738215440329823e-40 + T_*(2.021964051892091788e-42 + T_*(4.128740907197908282e-44 + T_*(2.049947393504634744e-46 + T_*(3.458727452025841363e-49)))))))))))))))))))) + Tref*(3.112534666153723136e-16 + T_*(-1.873677837019567290e-18 + T_*(-5.532179823077458254e-19 + T_*(3.972525584479210441e-20 + T_*(-5.188987892734337527e-22 + T_*(-3.391862596574743670e-23 + T_*(1.360453159001269833e-24 + T_*(-3.242112215867949590e-27 + T_*(-7.896373372992186301e-28 + T_*(1.390393221833685870e-29 + T_*(2.076015827047371213e-31 + T_*(-7.560992513098718830e-33 + T_*(-3.706163991610245534e-35 + T_*(2.232620184844021166e-36 + T_*(1.100559065706659980e-38 + T_*(-3.682533656835250976e-40 + T_*(-3.223844945267788809e-42 + T_*(1.959334144661060940e-44 + T_*(3.724119862112135153e-46 + T_*(1.769242397715477826e-48 + T_*(2.834630967748369536e-51)))))))))))))))))))) + Tref*(9.411328600369717429e-19 + T_*(-1.894119309117077157e-20 + T_*(-1.690159920349779348e-21 + T_*(1.555805697490074380e-22 + T_*(-2.709557222214898465e-24 + T_*(-1.153517854329395671e-25 + T_*(5.376287143187457606e-27 + T_*(-2.108474084097347475e-29 + T_*(-2.936789306313718369e-30 + T_*(5.194452993306611497e-32 + T_*(7.564143480889889416e-34 + T_*(-2.625235896598812213e-35 + T_*(-1.323767432812173567e-37 + T_*(7.405986542712064772e-39 + T_*(3.613468467711806690e-41 + T_*(-1.187961459658149567e-42 + T_*(-1.011384549214553935e-44 + T_*(6.348852099821227991e-47 + T_*(1.153959216106457934e-48 + T_*(5.305336228574541091e-51 + T_*(8.145685183982982076e-54))))))))))))))))))))))))))))))  # noqa: E501

    # Return theta-w converted to K
    return thw + 273.15

def pseudoadiabat_temp(p,thw):
    """
    Taken from https://github.com/robwarrenwx/atmos/blob/main/atmos/pseudoadiabat.py
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
    thref = -1.937920800653422475e+02 + p_*(1.994328166933734936e+00 + p_*(-2.237013575312922151e-02 + p_*(2.161737851675146153e-04 + p_*(-1.538607021098927656e-06 + p_*(7.955260853377150502e-09 + p_*(-2.983279520973976083e-11 + p_*(8.039566675064638876e-14 + p_*(-1.515292673478941849e-16 + p_*(1.854100168418495209e-19 + p_*(-1.115026192870100507e-22 + p_*(-3.969971732515197016e-26 + p_*(1.161169755869383278e-28 + p_*(-3.444083207986219187e-32 + p_*(-7.866653966250901723e-35 + p_*(6.625041435759308529e-38 + p_*(3.366795656456184692e-41 + p_*(-8.457678956195843103e-44 + p_*(5.836181847569190006e-47 + p_*(-1.910491737787847276e-50 + p_*(2.538391257078461717e-54))))))))))))))))))))  # noqa: E501
    T = -2.894809154394712181e+01 + thw_*(1.336650338083321898e+00 + thw_*(9.920943036173456367e-03 + thw_*(-5.531214167407858472e-05 + thw_*(-8.119357346521603519e-06 + thw_*(-1.620387392038411444e-07 + thw_*(4.037532909615920346e-09 + thw_*(2.733163059497750080e-10 + thw_*(9.427391543821474143e-13 + thw_*(-2.327854753793110044e-13 + thw_*(-3.504745859399798296e-15 + thw_*(1.108525038070919562e-16 + thw_*(2.815122082998062845e-18 + thw_*(-2.331111962009056858e-20 + thw_*(-1.138455622006271100e-21 + thw_*(-2.121842082147546585e-24 + thw_*(2.276656333522116348e-25 + thw_*(1.935220598348594753e-27 + thw_*(-1.435126816259871481e-29 + thw_*(-2.595738870945626914e-31 + thw_*(-9.691377297361992261e-34)))))))))))))))))))) + thref*(1.428881737537489149e+00 + thw_*(-7.928369124218480335e-03 + thw_*(-6.836727328681309508e-04 + thw_*(-1.575778513934627693e-05 + thw_*(2.363089166681158867e-07 + thw_*(3.377370448942234055e-08 + thw_*(6.763932007194685287e-10 + thw_*(-3.404787955461038834e-11 + thw_*(-1.392897110579503108e-12 + thw_*(1.754731183379382578e-14 + thw_*(1.325210356297305846e-15 + thw_*(-9.979176873736201255e-19 + thw_*(-7.316075852388014696e-19 + thw_*(-4.367591110344463074e-21 + thw_*(2.329382669197465213e-22 + thw_*(2.612519003597594999e-24 + thw_*(-3.663141187578808357e-26 + thw_*(-6.500140433674884368e-28 + thw_*(9.818511911722120689e-31 + thw_*(6.214415118411740173e-32 + thw_*(2.876486598268410161e-34)))))))))))))))))))) + thref*(4.972737015894919572e-03 + thw_*(-5.180031163839903972e-04 + thw_*(-8.443354080975294695e-06 + thw_*(6.970604985870997408e-07 + thw_*(4.745538679375798390e-08 + thw_*(3.605934838576005089e-10 + thw_*(-8.294489100400780575e-11 + thw_*(-2.420658446084975386e-12 + thw_*(6.731764854037241284e-14 + thw_*(3.351958552392139447e-15 + thw_*(-1.800972624739673327e-17 + thw_*(-2.324301238607420061e-18 + thw_*(-1.072851574806577385e-20 + thw_*(8.560224907361107825e-22 + thw_*(9.874192461006248167e-24 + thw_*(-1.414588503720268695e-25 + thw_*(-2.876521559727777627e-27 + thw_*(6.403220467343629572e-32 + thw_*(2.884475470792268876e-31 + thw_*(1.985478640920712375e-33 + thw_*(3.502058845458745414e-36)))))))))))))))))))) + thref*(-1.699534078374525885e-04 + thw_*(-3.900689915192402402e-06 + thw_*(7.214960588486119121e-07 + thw_*(3.788345751455296015e-08 + thw_*(-6.273481073442938028e-10 + thw_*(-1.214840085243346332e-10 + thw_*(-1.807232389121394695e-12 + thw_*(1.857449223972803575e-13 + thw_*(5.417783359399633895e-15 + thw_*(-1.472967199669265098e-16 + thw_*(-6.415493945575949307e-18 + thw_*(5.039132448915254268e-20 + thw_*(4.125098114685523165e-21 + thw_*(6.886806005805598986e-24 + thw_*(-1.478684652781221744e-24 + thw_*(-1.179343694500363276e-26 + thw_*(2.618396835490149079e-28 + thw_*(3.680001891075254443e-30 + thw_*(-1.108975483763781084e-32 + thw_*(-3.931760167079157084e-34 + thw_*(-1.718553853952838073e-36)))))))))))))))))))) + thref*(-3.325925800984791198e-06 + thw_*(2.450733346832845189e-07 + thw_*(2.215366969237199514e-08 + thw_*(-2.737615636676214507e-10 + thw_*(-9.250840980982384768e-11 + thw_*(-1.809819952947534347e-12 + thw_*(1.795487723452759831e-13 + thw_*(6.173650129477465262e-15 + thw_*(-1.701344681053855650e-16 + thw_*(-8.494797520341489610e-18 + thw_*(6.357161726488955308e-20 + thw_*(6.183553852752261233e-21 + thw_*(1.538679053848080909e-23 + thw_*(-2.438981420752364444e-24 + thw_*(-2.230971768664982437e-26 + thw_*(4.571162612759753901e-28 + thw_*(7.297138929988679687e-30 + thw_*(-1.572635491856643009e-32 + thw_*(-8.087623670303043750e-34 + thw_*(-4.271209690350353269e-36 + thw_*(-3.314300903114658836e-39)))))))))))))))))))) + thref*(3.414616856301753333e-08 + thw_*(6.268580800459015886e-09 + thw_*(-1.580414880896166803e-10 + thw_*(-3.678514556327383886e-11 + thw_*(-6.675021176897811867e-13 + thw_*(1.012337614818958783e-13 + thw_*(3.961263281808238210e-15 + thw_*(-1.381645980256255610e-16 + thw_*(-7.875887580502500946e-18 + thw_*(8.513951910182300971e-20 + thw_*(8.177221172048067876e-21 + thw_*(-8.703350255131019920e-26 + thw_*(-4.844774877757557582e-24 + thw_*(-3.190640346940055164e-26 + thw_*(1.610996932999312759e-27 + thw_*(1.907049805756753079e-29 + thw_*(-2.575040071002652052e-31 + thw_*(-4.785413537935581317e-33 + thw_*(6.186014275961251495e-36 + thw_*(4.578462095249169054e-37 + thw_*(2.158200510527907784e-39)))))))))))))))))))) + thref*(1.832539084881889899e-09 + thw_*(2.232154349041515070e-11 + thw_*(-1.720691072260209130e-11 + thw_*(-5.962355584738739171e-13 + thw_*(5.843949898672926130e-14 + thw_*(3.115793612394376880e-15 + thw_*(-8.732283600324087146e-17 + thw_*(-6.963438435050066380e-18 + thw_*(4.006495961779091573e-20 + thw_*(8.144337413199188819e-21 + thw_*(4.366031611409002476e-23 + thw_*(-5.234490503853010733e-24 + thw_*(-6.829505960497079623e-26 + thw_*(1.728116462140354231e-27 + thw_*(3.713189734493524910e-29 + thw_*(-1.838398243709750054e-31 + thw_*(-9.166025002345479808e-33 + thw_*(-3.883145420373912959e-35 + thw_*(7.747074099177721883e-37 + thw_*(8.755083691453450219e-39 + thw_*(2.651398973574923431e-41)))))))))))))))))))) + thref*(2.678809287247188447e-11 + thw_*(-9.936336895749793214e-13 + thw_*(-3.331147267147120202e-13 + thw_*(-1.424906631040857522e-15 + thw_*(1.568937881338988438e-15 + thw_*(3.023102970013440824e-17 + thw_*(-3.599415306574552030e-18 + thw_*(-9.462981008148341325e-20 + thw_*(4.378770462977206976e-21 + thw_*(1.396512678567408808e-22 + thw_*(-2.823320460275946884e-24 + thw_*(-1.137831839057805968e-25 + thw_*(8.132223243232400173e-28 + thw_*(5.273046758652151134e-29 + thw_*(2.712801402117304436e-32 + thw_*(-1.324919601404670228e-32 + thw_*(-7.264519583033080077e-35 + thw_*(1.510686767239771840e-36 + thw_*(1.373988911767683650e-38 + thw_*(-3.705871967500240777e-41 + thw_*(-4.980988505967098614e-43)))))))))))))))))))) + thref*(1.989628070690088685e-13 + thw_*(-1.523344650151727583e-14 + thw_*(-3.049694495393029476e-15 + thw_*(4.371888536717950116e-17 + thw_*(1.673864731634620975e-17 + thw_*(7.307645927407795769e-20 + thw_*(-4.423707668937182769e-20 + thw_*(-5.365318477748185415e-22 + thw_*(6.307660745198924017e-23 + thw_*(1.051921483697963415e-24 + thw_*(-5.090997089250726191e-26 + thw_*(-1.045822982223108853e-27 + thw_*(2.299390931720666921e-29 + thw_*(5.856181064117101660e-31 + thw_*(-5.127747701200545643e-33 + thw_*(-1.848983658781484082e-34 + thw_*(2.237846108161224605e-37 + thw_*(3.026680754951921398e-38 + thw_*(1.028290625341324349e-40 + thw_*(-1.947842364396178267e-42 + thw_*(-1.215938016409552301e-44)))))))))))))))))))) + thref*(7.648553716659091758e-16 + thw_*(-8.710548696300586344e-17 + thw_*(-1.384960302031060676e-17 + thw_*(3.953752472780564185e-19 + thw_*(8.363535806701477117e-20 + thw_*(-4.577577613176423449e-22 + thw_*(-2.394890583051712372e-22 + thw_*(-1.013681538018464148e-24 + thw_*(3.696580248182901828e-25 + thw_*(3.612583836691359672e-27 + thw_*(-3.265586466259759211e-28 + thw_*(-4.523940336584103313e-30 + thw_*(1.663325208493514205e-31 + thw_*(2.961550524803013470e-33 + thw_*(-4.566581435837660283e-35 + thw_*(-1.072435365589353776e-36 + thw_*(4.925056055046214252e-39 + thw_*(2.027553955194815686e-40 + thw_*(3.341279110766656890e-43 + thw_*(-1.556569362026769510e-44 + thw_*(-8.619061688502930173e-47)))))))))))))))))))) + thref*(1.209668896797693583e-18 + thw_*(-1.820592893624389411e-19 + thw_*(-2.511342379897640287e-20 + thw_*(1.010039918014097075e-21 + thw_*(1.620519955904728492e-22 + thw_*(-2.087854511890771101e-24 + thw_*(-4.895653838936800837e-25 + thw_*(5.810948626229826322e-28 + thw_*(7.948112235807493652e-28 + thw_*(4.372796890258928390e-30 + thw_*(-7.403655384903163203e-31 + thw_*(-7.554231171010766963e-33 + thw_*(4.011562970961407727e-34 + thw_*(5.725988369219874812e-36 + thw_*(-1.199729053573629101e-37 + thw_*(-2.293705058593750521e-39 + thw_*(1.567359307437282233e-41 + thw_*(4.724987348794075787e-43 + thw_*(3.401927490570441596e-46 + thw_*(-3.943223448362956556e-47 + thw_*(-2.070180466386040082e-49))))))))))))))))))))))))))))))  # noqa: E501

    # Return T converted to K
    return T + 273.15

def follow_moist_adiabat(pi,pf,Ti):
    """
    Taken from https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py
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
    Taken from https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py
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
    Lv = Lv0 + (cpv - cpl) * (T - T0)  

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
    https://github.com/robwarrenwx/atmos/blob/main/atmos/thermo.py.
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

    # convert to Kelvin and fraction
    T = T + 273.15
    RH = RH / 100.

    if Twb_method == 'thermo_adiabatic':

        # Get pressure and temperature at the LCL
        p_lcl, T_lcl = lifting_condensation_level(P, T, q, RH)

        # Follow a pseudoadiabat from the LCL to the original pressure
        Twb = follow_moist_adiabat(p_lcl, P, T_lcl)

    elif Twb_method == 'thermo_isobaric':

        # Compute dewpoint temperature
        Td = dewpoint_temperature_from_relative_humidity(T, RH)

        # Initialise Twb as mean of T and Td
        Twb = (T + Td) / 2

        # Compute the latent heat at temperature T
        Lv_T = Lv0 + (cpv - cpl) * (T - T0)     

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
            cpm_qs_Twb = cpm = (1 - qs_Twb) * cpd + qs_Twb * cpv

            # Compute the latent heat of vaporisation at Twb
            Lv_Twb = Lv0 + (cpv - cpl) * (Twb - T0)  

            # Compute the derivative of qs with respect to Twb
            dqs_dTwb = qs_Twb * (1 + qs_Twb / eps - qs_Twb) * Lv_Twb / (Rv * Twb**2)

            # Compute f and f'
            f = cpm_qs_Twb * (T - Twb) - Lv_T * (qs_Twb - q)
            fprime = ((cpv - cpd) * (T - Twb) - Lv_T) * dqs_dTwb - cpm_qs_Twb
         
            # Update Twb using Newton's method
            Twb = Twb - f / fprime

            # Check for convergence
            if np.max(np.abs(Twb - Twb_prev)) < 0.0001:
                converged = True
            else:
                count += 1
                if count == 20:
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
    - Fpw1r (view from partially illuminated wall of the road; -)
    - Fpw1w2 (view from partially illuminated wall of the opposite wall; -)
    """

    if (type(fr) == float) or (fr.ndim == 0):
        fr = np.array(fr,ndmin=1)
    if (type(fw) == float) or (fw.ndim == 0):
        fw = np.array(fw,ndmin=1)
    if (type(H) == float) or (H.ndim == 0):
        H = np.array(H,ndmin=1)
    if (type(W) == float) or (W.ndim == 0):
        W = np.array(W,ndmin=1)
    Fprw1, Fprw2, Fpw1r, Fpw1w2 = np.zeros(np.shape(fr)), np.zeros(np.shape(fr)), \
                                    np.zeros(np.shape(fr)), np.zeros(np.shape(fr))
    if geometry_choice == "canyon":
        # wall partially illuminated 
        m = fw!=0
        H_, W_, fr_, fw_ = np.copy(H[m]), np.copy(W[m]), \
            np.copy(fr[m]), np.copy(fw[m])
        np.putmask(Fpw1r,m,0.5+(np.sqrt(W_**2+(1+fw_)**2*H_**2)-np.sqrt(H_**2+W_**2))/(2*fw_*H_)) # proportion from road
        np.putmask(Fpw1w2,m,(np.sqrt(W_**2+fr_**2*H_**2)-np.sqrt(W_**2+(1-fw_)**2*H_**2)+np.sqrt(H_**2+W_**2)-W_)/(2*fw_*H_)) # proportion from opposite wall
        # road partially illuminated
        m = fr!=0
        H_, W_, fr_, fw_ = np.copy(H[m]), np.copy(W[m]), \
            np.copy(fr[m]), np.copy(fw[m])
        np.putmask(Fprw1,m,0.5+(H_-np.sqrt(H_**2+fr_**2*W_**2))/(2*fr_*W_)) # proportion from partially illuminated wall
        np.putmask(Fprw2,m,0.5+(np.sqrt(H_**2+(1+fr_)**2*W_**2)-np.sqrt(H_**2+W_**2))/(2*fr_*W_)) # proportion from opposite wall
    elif geometry_choice == "flat":
        pass
    else:
        sys.exit("!!! partial2facet_view_fracs -- geometry_choice can only be `canyon` or `flat` !!!")

    return Fprw1, Fprw2, Fpw1r, Fpw1w2

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
    H=None, W=None, Z=None, X=None, canyon_orient_deg=None,\
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

    # ensure these have the same shape as the grid (and do nothing if single point)
    elevation = adjust_array_shape(elevation, lon)
    H = adjust_array_shape(H, lon)
    W = adjust_array_shape(W, lon)
    X = adjust_array_shape(X, lon)

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
        Fprw1, Fprw2, Fpw1r, Fpw1w2 = partial2facet_view_fracs(H,W,fr,fw,geometry_choice)
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

    return Twb, solar_zen_deg, solar_azi_deg, canyon_azi_deg, Fs, Fr, Fw, Fsr, Frs, Fww, Fwr, Fws, Frw, Fsw, \
            fr, fw, Fpr, Fpw, Fprw1, Fprw2, Fpw1r, Fpw1w2, Sr, Sw, K, Ks, Kr, Kw, I, L, MRT, Tg, WBGT

if __name__ == "__main__":
    # This block ensures that main() is called only when the script is executed directly
    main()

