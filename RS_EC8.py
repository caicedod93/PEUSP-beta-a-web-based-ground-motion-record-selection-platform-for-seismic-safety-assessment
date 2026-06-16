# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:18:37 2022

@author: Daniel Caicedo
"""

import numpy as np
import numpy.matlib

def get_RS_EC8_1(agr, zeta, T, Ifactor, Type, Soil_Type):
    """
    Horizontal elastic response spectrum based on EC8-part 1.
    References:
    Eurocode 8: Design of Structures for Earthquake Resistance
    Part 1: General Rules, Seismic Actions and Rules for Buildings
    (EN 1998-1:2004)
    agr: The reference peak ground acceleration on type A ground
    zeta: Damping ratio (i.e. zeta = 5 for 5% damping, eta=1)
    T: Range of periods for SDOF
    Ifactor: Importance factor
    Type: Type of spectrum for seismic hazard accroding to
    surface wave magnitude: 'Type1' for Mw > 5.5 or 'Type2' for Mw < 5.5
    Soil_Type: Soil_Type condition ('A' or 'B' or 'C' or 'D' or 'E')
    """
    # Table 3.2-3 values of the parameters describing response spectra
    spectrum_data = {
        'Type1': {
            'A': {'S': 1.0, 'Tb': 0.15, 'Tc': 0.4, 'Td': 2},
            'B': {'S': 1.2, 'Tb': 0.15, 'Tc': 0.5, 'Td': 2},
            'C': {'S': 1.15, 'Tb': 0.2, 'Tc': 0.6, 'Td': 2},
            'D': {'S': 1.35, 'Tb': 0.2, 'Tc': 0.8, 'Td': 2},
            'E': {'S': 1.4, 'Tb': 0.15, 'Tc': 0.5, 'Td': 2}
        },
        'Type2': {
            'A': {'S': 1.0, 'Tb': 0.05, 'Tc': 0.25, 'Td': 1.2},
            'B': {'S': 1.35, 'Tb': 0.05, 'Tc': 0.25, 'Td': 1.2},
            'C': {'S': 1.5, 'Tb': 0.10, 'Tc': 0.25, 'Td': 1.2},
            'D': {'S': 1.8, 'Tb': 0.10, 'Tc': 0.30, 'Td': 1.2},
            'E': {'S': 1.6, 'Tb': 0.05, 'Tc': 0.25, 'Td': 1.2}
        }
    }
    S = spectrum_data[Type][Soil_Type]['S']
    Tb = spectrum_data[Type][Soil_Type]['Tb']
    Tc = spectrum_data[Type][Soil_Type]['Tc']
    Td = spectrum_data[Type][Soil_Type]['Td']
    eta = max((10 / (5 + zeta*100)) ** 0.5, 0.55)
    ag = agr * Ifactor
    Sa = np.zeros(np.size(T))
    idx1 = np.where((T <= Tb) * (T >= 0))
    idx2 = np.where((T >= Tb) * (T <= Tc))
    idx3 = np.where((T >= Tc) * (T <= Td))
    idx4 = np.where((T >= Td) * (T <= 4.0))
    Sa[idx1] = S * (1 + T[idx1] * (eta * 2.5 - 1) / Tb)
    Sa[idx2] = S * eta * 2.5
    Sa[idx3] = S * eta * 2.5 * (Tc / T[idx3])
    Sa[idx4] = S * eta * 2.5 * (Tc * Td / T[idx4] ** 2)
    Sa = Sa * ag
    return Sa



