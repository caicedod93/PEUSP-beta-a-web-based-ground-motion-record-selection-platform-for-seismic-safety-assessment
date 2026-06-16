# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:18:37 2022

@author: Daniel Caicedo
"""

import numpy as np
import numpy.matlib

T = np.arange(start=0.00, stop=4.05, step=0.05)
S_alpha_RP = 0.35 # Reference Sa corresponding to the constant acceleration range of the horizontal 5%-damped elastic RS, on site category A and return period Tref = 475 years
S_beta_RP = 0.1 # Reference Sa at Tb = 1 s of the horizontal 5%-damped elastic RS, on site category A and return period Tref = 475 years
Soil_type = 'D'

def get_RS_EC82024(T, S_alpha_RP, S_beta_RP, Soil_type):
    """
    Horizontal elastic response spectrum based on TBSC (2018).
    References:
    S_DS: Design spectral acceleration coefficient for the short period region
    D_D1: Design spectral acceleration coefficient for T=1s
    TA and TB: Corner periods
    TL: Transition period to the constant displacement zone
    """
    TA = 0.02
    FA = 2.5
    x = 4
    FT = 1
    if S_beta_RP*9.80665 > 1:
        TD = 1 + S_beta_RP*9.80665
    else:
        TD = 2
    
    if Soil_type == 'A':
        F_alpha = 1
        F_beta = 1
    elif Soil_type == 'B':
        F_alpha = 1.3*(1-0.1*S_alpha_RP)
        F_beta = 1.6*(1-0.2*S_beta_RP)
    elif Soil_type == 'C':
        F_alpha = 1.6*(1-0.2*S_alpha_RP)
        F_beta = 2.3*(1-0.3*S_beta_RP)
    elif Soil_type == 'D':
        F_alpha = 1.8*(1-0.3*S_alpha_RP)
        F_beta = 3.2*(1-S_beta_RP)
    
    S_alpha = FT*F_alpha*S_alpha_RP
    S_beta = FT*F_beta*S_beta_RP
    
    TC = S_beta/S_alpha
    
    if ((TC/x >= 0.05) and (TC/x <= 0.1)): 
        TB = TC/x
    elif TC/x < 0.05:
        TB = 0.05
    elif TC/x > 0.10:
        TB = 0.10
        
            
    Sa = np.zeros(np.size(T))
    idx1 = np.where((T < TA) * (T >= 0))
    idx2 = np.where((T >= TA) * (T < TB))
    idx3 = np.where((T >= TB) * (T < TC))
    idx4 = np.where((T >= TC) * (T < TD))
    idx5 = np.where(T >= TD)
    
    Sa[idx1] = S_alpha/FA
    Sa[idx2] = (S_alpha/(TB-TA))*((T[idx2]-TA)+(TB-T[idx2])/FA)
    Sa[idx3] = S_alpha
    Sa[idx4] = S_beta/T[idx4]
    Sa[idx5] = TD*S_beta/T[idx5]**2
    return Sa
