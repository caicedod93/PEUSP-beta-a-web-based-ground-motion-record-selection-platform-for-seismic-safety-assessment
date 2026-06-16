import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def run_scenario_based(
    metadata_path: str,
    # RECORDS AND MATCHING
    N_records: int,
    # SCENARIO (Mw,R)-based
    Mw_lower_bound,
    Mw_upper_bound,
    Rjb_lower_bound,
    Rjb_upper_bound,
    # ADDITIONAL FILTERING
    Soil_Type_filtering: str,
    Style_of_Faulting,
    depth_lower_bound,
    depth_upper_bound,

):


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

    # ---- Build spectra arrays (same indices as your script)
    Filename = []
    Filename = metadata_filtered.Filename
    Filename = list(Filename)
    Filename = np.array(Filename)
    
    arr = metadata_filtered.to_numpy()

    PGAX = []
    PGAX = (arr[:,16]).astype(float)
    sax = (arr[:,73:153]).astype(float)
    SaX = np.zeros((np.size(T),len(Filename)))
    SaX[0,:] = PGAX
    SaX[1:,:] = np.transpose(sax)

    PGAY = []
    PGAY = (arr[:,17]).astype(float)
    say = (arr[:,153:233]).astype(float)
    SaY = np.zeros((np.size(T),len(Filename)))
    SaY[0,:] = PGAY
    SaY[1:,:] = np.transpose(say)

    Sa = (SaX*SaY)**0.5 # GEOMEAN for bi-directional analysis; BETTER
    Target = Sa.mean(axis=1) 

    # import scipy.stats as st
    # conf_inf , conf_sup = st.t.interval(alpha=0.95, df=len(arr)-1, loc=np.mean(Sa, axis=1), scale=st.sem(Sa, axis=1))

    SEE = np.zeros(len(Filename))
    for i in range(len(Filename)):
        SEE[i] = (np.square(Sa[:,i] - Target)).mean(axis=0)
        
    metadata_filtered['SEE'] = SEE
    metadata_filtered_SEE = metadata_filtered.sort_values(by=['SEE'], ascending=True)
    metadata_filtered_SEE = metadata_filtered_SEE[~metadata_filtered_SEE.duplicated('Event_ID', keep='first')]
    metadata_filtered_SEE = metadata_filtered_SEE.head(N_records)

    Filename = []
    Filename = metadata_filtered_SEE.Filename
    Filename = list(Filename)
    Filename = np.array(Filename)

    arr = metadata_filtered_SEE.to_numpy()

    PGAX = []
    PGAX = (arr[:,16]).astype(float)
    sax = (arr[:,73:153]).astype(float)
    SaX = np.zeros((np.size(T),len(Filename)))
    SaX[0,:] = PGAX
    SaX[1:,:] = np.transpose(sax)

    PGAY = []
    PGAY = (arr[:,17]).astype(float)
    say = (arr[:,153:233]).astype(float)
    SaY = np.zeros((np.size(T),len(Filename)))
    SaY[0,:] = PGAY
    SaY[1:,:] = np.transpose(say)

    # Sa = (SaX**2+SaY**2)**0.5
    
    Rjb_lower_bound  = float(Rjb_lower_bound)
    if Rjb_lower_bound > 12:
        Sa = (10*(SaX*SaY)**0.5)/980.665
    else:
        Sa = ((SaX*SaY)**0.5)/980.665


    conf_inf = np.percentile(Sa, 5, axis =1)
    conf_sup = np.percentile(Sa, 95, axis =1)

    ## PLOT
    fig, ax = plt.subplots(layout='constrained')
    plt.figure (1)
    ax.loglog(T,np.mean(Sa, axis=1),color='k', linestyle='-', linewidth=2)
    ax.loglog(T,np.median(Sa, axis=1),color='k', linestyle='--', linewidth=1.5)
    ax.loglog(T,conf_inf,color='k', linestyle='-.', linewidth=1)
    ax.loglog(T,conf_sup,color='k', linestyle='-.', linewidth=1)

    for i in range(N_records):
        ax.plot(T,Sa[:,i], color = 'grey', linestyle='-', linewidth=0.2)

    legend_elements = [Line2D([0], [0], color ='k', linestyle='-', linewidth=2, label='Mean'),
                    Line2D([0], [0], color ='k', linestyle='-.', linewidth=1.5, label='Median'),
                    Line2D([0], [0], color ='k', linestyle='-.', linewidth=1, label=r'95% confidence interval'),
                    Line2D([0], [0], color ='grey', linestyle='-', linewidth=0.5, label='Individual spectra')]
    ax.legend(handles=legend_elements, fontsize=14)     
    plt.xlabel('T [s]', fontsize=14)
    plt.ylabel('S$_a$ [g]', fontsize=14)
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    # plt.ylim(0.001, 2)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlim(0, 4)
    
    out = metadata_filtered_SEE.copy()
    return out, fig