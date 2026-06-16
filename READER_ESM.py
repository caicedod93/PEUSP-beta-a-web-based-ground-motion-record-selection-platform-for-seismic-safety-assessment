# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:53:44 2022

@author: Daniel Caicedo
"""
"""
READER FOR SIGMA RECORD FILES (.ACC)
"""
import numpy as np
"""
INPUT PARAMETERS
"file" AS FULL DIRECTION OF THE .ASC FILE TO BE READ
"""
def readESM(file):
    """
    READING ESM FILE (.ASC)
    """   
    try:
        with open(file, 'r') as inputfile:
            content = inputfile.readlines()
            
        SAMPLING_INTERVAL_S = content[28].split()
        recData = content[64:]
        dt = float(SAMPLING_INTERVAL_S.pop())
        ac = np.loadtxt(recData).flatten()
        dt = np.round(dt, 3)
        t = np.arange(len(ac)) * dt
        return dt, t, ac
    except Exception:
        print('Reading ESM Record Failed')
        
# path = '/Users\Daniel Caicedo\Desktop\Daniel Caicedo\Minho\Records\ESM\Portugal/by_event\PT-1996-0007/EU.GAR.00.HN2.D.PT-1996-0007.ACC.MP.ASC'

# dt, t, ac = readESM(path)