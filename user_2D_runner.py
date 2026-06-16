# AnnexD.py  (add this function and keep the rest of your code reusable)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import differential_evolution

def run_user_2d(
    metadata_path: str,
    # RECORDS AND MATCHING
    N_records: int,
    T1: float,
    T2: float,
    Soil_Type_filtering: str,
    Target_user: list,
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
    Design_Spectrum = Target_user * 980.665

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

    # ---- Build spectra arrays (same indices as your script)
    arr = metadata_filtered.to_numpy()

    PGAX = (arr[:, 16]).astype(float)
    SaX  = (arr[:, 73:153]).astype(float)
    TX = np.zeros((T.size, len(metadata_filtered)))
    TX[0, :] = PGAX
    TX[1:, :] = SaX.T

    PGAY = (arr[:, 17]).astype(float)
    SaY  = (arr[:, 153:233]).astype(float)
    TY = np.zeros((T.size, len(metadata_filtered)))
    TY[0, :] = PGAY
    TY[1:, :] = SaY.T

    # geometric mean
    TR = (TX * TY) ** 0.5

    # Pre-rank by SEE in matching domain
    TRd = TR[Domain]
    SEE_R = np.square(TRd - Design_Spectrum[Domain][:, None]).mean(axis=0)
    metadata_filtered = metadata_filtered.copy()
    metadata_filtered['SEE_R'] = SEE_R

    # Keep best by SEE, avoid duplicate Event_ID
    metadata_filtered_SEE = (
        metadata_filtered.sort_values('SEE_R', ascending=True)
        .drop_duplicates('Event_ID', keep='first')
        .head(N_records)
        .reset_index(drop=True)
    )

    # Rebuild TR for the selected candidates
    arr2 = metadata_filtered_SEE.to_numpy()

    PGAX = (arr2[:, 16]).astype(float)
    SaX  = (arr2[:, 73:153]).astype(float)
    TX = np.zeros((T.size, len(metadata_filtered_SEE)))
    TX[0, :] = PGAX
    TX[1:, :] = SaX.T

    PGAY = (arr2[:, 17]).astype(float)
    SaY  = (arr2[:, 153:233]).astype(float)
    TY = np.zeros((T.size, len(metadata_filtered_SEE)))
    TY[0, :] = PGAY
    TY[1:, :] = SaY.T

    TR = (TX * TY) ** 0.5

    # ---- Optimization (domain only)
    Sa_ind = TR[Domain]
    PGA_ind = TR[0, :]
    Target = Design_Spectrum[Domain]

    def objective(x):
        MEAN = np.zeros(Target.size)
        PGA_MEAN = 0.0

        penalty_ind_max = np.zeros(N_records)
        penalty_ind_min = np.zeros(N_records)

        for i in range(N_records):
            MEAN += (x[i] * Sa_ind[:, i]) / N_records
            PGA_MEAN += (x[i] * PGA_ind[i]) / N_records

            IND_MIS = (x[i] * Sa_ind[:, i]) / Target

            # individual penalties
            imax = np.argmax(IND_MIS)
            imin = np.argmin(IND_MIS)

            if IND_MIS.max() > Max_Target_ind:
                penalty_ind_max[i] = (x[i] * Sa_ind[imax, i]) - Max_Target_ind * Target[imax]
            if IND_MIS.min() < Min_Target_ind:
                penalty_ind_min[i] = Min_Target_ind * Target[imin] - (x[i] * Sa_ind[imin, i])

        # PGA constraint
        penalty_PGA = max(0.0, Target_PGA - PGA_MEAN)

        # mean constraints
        G1 = MEAN / Target
        gmax_i = np.argmax(G1)
        gmin_i = np.argmin(G1)

        penalty_G1_max = max(0.0, MEAN[gmax_i] - Max_Target_MEAN * Target[gmax_i])
        penalty_G1_min = max(0.0, Min_Target_MEAN * Target[gmin_i] - MEAN[gmin_i])

        SEE = np.square(MEAN - Target).mean() ** 0.5
        return float(SEE + penalty_PGA + penalty_G1_max + penalty_G1_min + penalty_ind_max.sum() + penalty_ind_min.sum())

    result = differential_evolution(objective, bounds, maxiter=100, popsize=100, mutation=0.5, recombination=0.4)
    x = result.x

    # ---- 2D plotting (full T)
    Target_full = Design_Spectrum
    MEAN = np.zeros(Target_full.size)
    IND = np.zeros((Target_full.size, N_records))

    for i in range(N_records):
        IND[:, i] = x[i] * TR[:, i]
        MEAN += IND[:, i] / N_records

    fig, ax = plt.subplots(layout='constrained')
    ax.plot(T, MEAN / 980.665, color='k', linestyle='-', linewidth=2)
    ax.plot(T, Target_full / 980.665, color='k', linestyle='--', linewidth=2)

    ax.plot(T, (Min_Target_MEAN * Target_full) / 980.665, color='k', linestyle='--', linewidth=1)
    ax.plot(T, (Max_Target_MEAN * Target_full) / 980.665, color='k', linestyle='--', linewidth=1)
    ax.plot(T, (Min_Target_ind * Target_full) / 980.665, color='k', linestyle='-.', linewidth=1)
    ax.plot(T, (Max_Target_ind * Target_full) / 980.665, color='k', linestyle='-.', linewidth=1)

    ax.axvline(x=T1, color='r', linestyle=':', linewidth=1.5)
    ax.axvline(x=T2, color='r', linestyle=':', linewidth=1.5)
    ax.axvspan(T1, T2, alpha=0.25, color='red')

    for i in range(N_records):
        ax.plot(T, IND[:, i] / 980.665, color='grey', linestyle='-', linewidth=0.7)

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

    return out, fig, result
