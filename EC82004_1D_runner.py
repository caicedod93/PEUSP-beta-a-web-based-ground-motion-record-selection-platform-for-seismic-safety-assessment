# AnnexD.py  (add this function and keep the rest of your code reusable)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import differential_evolution
import RS_EC8 as RS_EC8

def run_EC82024_1d_ec82004(
    metadata_path: str,
    # RECORDS AND MATCHING
    N_records: int,
    T1: float,
    T2: float,
    Soil_Type_filtering: str,
    
    agr: float,
    Ifactor: float,
    Type: str,
    Soil_Type_spectrum: str,
    
    Min_scalling: float,
    Max_scalling: float,
    Min_Target_MEAN: float,
    Max_Target_MEAN: float,
    Min_Target_ind: float,
    Max_Target_ind: float,
    # FILTERS (allow '' strings)
    Style_of_Faulting,
    Mw_lower_bound,
    Mw_upper_bound,
    Rjb_lower_bound,
    Rjb_upper_bound,
    depth_lower_bound,
    depth_upper_bound,
    seed: int = 1,
):
    np.random.seed(seed)

    # ---- Soil Vs30 bounds (same logic as your script)
    if Soil_Type_filtering == 'A':
        Vs30_lower_bound, Vs30_upper_bound = 800, 5000
    elif Soil_Type_filtering == 'B':
        Vs30_lower_bound, Vs30_upper_bound = 360, 800
    elif Soil_Type_filtering == 'C':
        Vs30_lower_bound, Vs30_upper_bound = 180, 360
    elif Soil_Type_filtering == 'D':
        Vs30_lower_bound, Vs30_upper_bound = 0, 180
    elif Soil_Type_filtering == 'ALL':
        Vs30_lower_bound, Vs30_upper_bound = 0, 5000
    else:
    
        raise ValueError("Soil_Type must be one of: A, B, C, D")

    # ---- Defaulting logic for blanks (same as your script)
    if Mw_lower_bound == '': Mw_lower_bound = 0
    if Mw_upper_bound == '': Mw_upper_bound = 10
    if Rjb_lower_bound == '': Rjb_lower_bound = 0
    if Rjb_upper_bound == '': Rjb_upper_bound = 10000
    if depth_lower_bound == '': depth_lower_bound = 0
    if depth_upper_bound == '': depth_upper_bound = 10000

    metadata = pd.read_csv(metadata_path)

    T = np.arange(start=0, stop=4.05, step=0.05)
    Design_Spectrum = RS_EC8.get_RS_EC8_1(agr, 0.05, T, Ifactor, Type, Soil_Type_spectrum) * 980.665

    bounds = [(Min_scalling, Max_scalling) for _ in range(N_records)]
    Domain = np.where((T >= T1) * (T <= T2))
    Target_PGA = Design_Spectrum[0]

    # ---- NEW: normalize mechanism input to a list (0–3 entries)
    def _normalize_mech(mech):
        if mech is None:
            return []
        if isinstance(mech, str):
            mech = mech.strip()
            return [] if mech == "" else [mech]
        # list/tuple/set/np.array/etc.
        out = []
        for m in mech:
            m = str(m).strip()
            if m != "":
                out.append(m)
        return out

    mech_list = _normalize_mech(Style_of_Faulting)

    # ---- Filtering (same constraints; upgraded mechanism logic)
    base_mask = (
        (metadata['Mw'] >= Mw_lower_bound) & (metadata['Mw'] <= Mw_upper_bound) &
        (metadata['Rjb'] >= Rjb_lower_bound) & (metadata['Rjb'] <= Rjb_upper_bound) &
        (metadata['Depth'] >= depth_lower_bound) & (metadata['Depth'] <= depth_upper_bound) &
        (metadata['Vs30'] >= Vs30_lower_bound) & (metadata['Vs30'] <= Vs30_upper_bound)
    )

    if len(mech_list) == 0:
        metadata_filtered = metadata[base_mask]
    else:
        metadata_filtered = metadata[base_mask & (metadata['Mechanism'].isin(mech_list))]

    metadata_filtered = metadata_filtered.reset_index(drop=True)
    if 'Country' in metadata_filtered.columns:
        metadata_filtered = metadata_filtered.drop(columns=['Country'])

    if len(metadata_filtered) < N_records:
        raise ValueError(f"Filtering returned only {len(metadata_filtered)} records; need at least {N_records}.")

    Filename = []
    Filename = metadata_filtered.Filename
    Filename = list(Filename)
    Filename = np.array(Filename)
    
    arr = metadata_filtered.to_numpy()
    
    PGAX = []
    PGAX = (arr[:,16]).astype(float)
    SaX = (arr[:,73:153]).astype(float)
    TX = np.zeros((np.size(T),len(Filename)))
    TX[0,:] = PGAX
    TX[1:,:] = np.transpose(SaX)
    
    PGAY = []
    PGAY = (arr[:,17]).astype(float)
    SaY = (arr[:,153:233]).astype(float)
    TY = np.zeros((np.size(T),len(Filename)))
    TY[0,:] = PGAY
    TY[1:,:] = np.transpose(SaY)
    
    SEE_X = np.zeros(len(Filename))
    SEE_Y = np.zeros(len(Filename))
    

    # --- Robust domain slicing (Domain is a tuple from np.where)
    dom_idx = Domain[0]
    TX_dom = TX[dom_idx, :]
    TY_dom = TY[dom_idx, :]
    Target_dom = Design_Spectrum[dom_idx]

    for i in range(len(Filename)):
        SEE_X[i] = np.mean((TX_dom[:, i] - Target_dom) ** 2)
        SEE_Y[i] = np.mean((TY_dom[:, i] - Target_dom) ** 2)
    metadata_filtered['SEE_X'] = SEE_X
    metadata_filtered['SEE_Y'] = SEE_Y
    
    metadata_filtered_SEE = metadata_filtered.sort_values(by=['SEE_X', 'SEE_Y'], ascending=True)
    metadata_filtered_SEE = metadata_filtered_SEE[~metadata_filtered_SEE.duplicated('Event_ID', keep='first')]
    metadata_filtered_SEE = metadata_filtered_SEE.head(N_records)
    
    Filename = []
    Filename = metadata_filtered_SEE.Filename
    Filename = list(Filename)
    Filename = np.array(Filename)
    
    arr = metadata_filtered_SEE.to_numpy()
    
    PGAX = []
    PGAX = (arr[:,16]).astype(float)
    SaX = (arr[:,73:153]).astype(float)
    TX = np.zeros((np.size(T),len(Filename)))
    TX[0,:] = PGAX
    TX[1:,:] = np.transpose(SaX)
    
    PGAY = []
    PGAY = (arr[:,17]).astype(float)
    SaY = (arr[:,153:233]).astype(float)
    TY = np.zeros((np.size(T),len(Filename)))
    TY[0,:] = PGAY
    TY[1:,:] = np.transpose(SaY)
    
    # TR=(TX**2+TY**2)**0.5 # SRSS for bi-directional analysis
    TXY = np.concatenate((TX,TY), axis=1)
    
    SEE_X = (arr[:,-2]).astype(float)
    SEE_Y = (arr[:,-1]).astype(float)
    
    comp = np.zeros(len(Filename), dtype=int)
    T_one = np.zeros((np.size(T),len(Filename)))
    for i in range(len(Filename)):
        comp[i] = 0 if (SEE_X[i] < SEE_Y[i]) else 1
        T_one[:,i] = TXY[:,i + comp[i]*len(Filename)]
    # 1-D ANALYSIS
    Sa_ind = T_one[Domain]
    PGA_ind = T_one[0,:]
    Target = 0.90*Design_Spectrum[Domain]
    
    
    def objective(x):
        MEAN = np.zeros(np.size(Target))
        PGA_MEAN = np.zeros(1)
        IND_MIS = np.zeros((np.size(Target),N_records))
        penalty_ind_max = np.zeros(N_records)
        penalty_ind_min = np.zeros(N_records)
        
        for i in range(N_records):
            MEAN += (x[i]*Sa_ind[:,i])/N_records
            PGA_MEAN += (x[i]*PGA_ind[i])/N_records
            IND_MIS[:,i] =  (x[i]*Sa_ind[:,i])/Target
            
            # INDIVIDUAL MISMATCH
            T_ind_MAX = np.where(IND_MIS[:,i] == np.max(IND_MIS[:,i]))  
            if (np.max(IND_MIS[:,i]) <= Max_Target_ind):
                penalty_ind_max[i] = 0
            elif (np.max(IND_MIS[:,i]) > Max_Target_ind):
                penalty_ind_max[i] = (x[i]*Sa_ind[:,i])[T_ind_MAX] - Max_Target_ind*Target[T_ind_MAX]
                
            T_ind_MIN = np.where(IND_MIS[:,i] == np.min(IND_MIS[:,i]))  
            if (np.min(IND_MIS[:,i]) >= Min_Target_ind):
                penalty_ind_min[i] = 0
            elif (np.min(IND_MIS[:,i]) < Min_Target_ind):
                penalty_ind_min[i] = Min_Target_ind*Target[T_ind_MIN] - (x[i]*Sa_ind[:,i])[T_ind_MIN] 
                
            # PGA CONSTRAINT
            if PGA_MEAN >= Target_PGA:
                penalty_PGA = 0
            elif PGA_MEAN < Target_PGA:
                penalty_PGA = Target_PGA - PGA_MEAN
            penalty_PGA_cost = np.float64(penalty_PGA)
            
            # MAX MEAN LIMIT CONSTRAINT
            G1 = MEAN/Target
            T_G1_MAX = np.where(G1 == np.max(G1))
            if (np.max(G1) <= Max_Target_MEAN):
                penalty_G1_max = 0
            elif (np.max(G1) > Max_Target_MEAN):
                penalty_G1_max = MEAN[T_G1_MAX] - Max_Target_MEAN*Target[T_G1_MAX]
            penalty_G1_max_cost = np.float64(penalty_G1_max)
            
            # MIN MEAN LIMIT CONSTRAINT
            T_G1_MIN = np.where(G1 == np.min(G1))
            if (np.min(G1) >= Min_Target_MEAN):
                penalty_G1_min = 0
            elif (np.max(G1) < Max_Target_MEAN):
                penalty_G1_min = Min_Target_MEAN*Target[T_G1_MIN] - MEAN[T_G1_MIN] 
            penalty_G1_min_cost = np.float64(penalty_G1_min)
                
        SEE =  np.square(MEAN-Target).mean()**0.5
        # ALL CONSTRAINS INCLUDED
        cost = SEE + penalty_PGA_cost + penalty_G1_max_cost + penalty_G1_min_cost + np.sum(penalty_ind_max) + np.sum(penalty_ind_min)
        return cost
    
    result = differential_evolution(objective,bounds, maxiter=100, popsize=100, mutation=0.5, recombination=0.4)
    
    x=result.x
    
    # 1-D ANALYSIS
    Sa_ind = T_one
    Target = Design_Spectrum
    
    MEAN = np.zeros(np.size(Target))
    IND_MIS = np.zeros((np.size(Target),N_records))
    
    for i in range(N_records):
        MEAN += (x[i]*Sa_ind[:,i])/N_records
        IND_MIS[:,i] =  (x[i]*Sa_ind[:,i])


    fig, ax = plt.subplots(layout='constrained')
    ax.plot(T, MEAN / 980.665, color='k', linestyle='-', linewidth=2)
    ax.plot(T,Target/980.665,color='k', linestyle='--', linewidth=2)

    ax.plot(T, (Min_Target_MEAN * Target) / 980.665, color='k', linestyle='--', linewidth=1)
    ax.plot(T, (Max_Target_MEAN * Target) / 980.665, color='k', linestyle='--', linewidth=1)
    ax.plot(T, (Min_Target_ind * Target) / 980.665, color='k', linestyle='-.', linewidth=1)
    ax.plot(T, (Max_Target_ind * Target) / 980.665, color='k', linestyle='-.', linewidth=1)

    ax.axvline(x=T1, color='r', linestyle=':', linewidth=1.5)
    ax.axvline(x=T2, color='r', linestyle=':', linewidth=1.5)
    ax.axvspan(T1, T2, alpha=0.25, color='red')

    for i in range(N_records):
        ax.plot(T, IND_MIS[:, i] / 980.665, color='grey', linestyle='-', linewidth=0.7)

    legend_elements = [
        Line2D([0],[0], color='k', linestyle='-',  linewidth=2, label='Mean'),
        Line2D([0],[0], color='k', linestyle='--', linewidth=2, label='Target'),
        Line2D([0],[0], color='k', linestyle='--', linewidth=1, label='Mean limits'),
        Line2D([0],[0], color='k', linestyle='-.', linewidth=1, label='Individual limits'),
        Line2D([0],[0], color='grey', linestyle='-', linewidth=0.7, label='Individual spectra'),
        Line2D([0],[0], color='r', linestyle=':', linewidth=1.5, label='Matching range'),
    ]
    ax.legend(handles=legend_elements, fontsize=12)
    ax.set_xlabel('T [s]', fontsize=12)
    ax.set_ylabel(r'$S_a$ [g]', fontsize=12)
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 4)

    # ---- Return results
    out = metadata_filtered_SEE.copy()
    out["scale_factor"] = x
    out["comp"] = comp

    return out, fig, result, comp, x
