# app.py
import os
from pathlib import Path

import streamlit as st
import AnnexD_2D_runner
import AnnexD_1D_runner
import EC82004_1D_runner
import EC82004_2D_runner
import user_1D_runner
import user_2D_runner
import Scenario_based_runner
import numpy as np
import io

import requests
from zipfile import ZipFile
import tempfile

st.set_page_config(page_title="PEUSP_beta", layout="centered")

# -----------------------------
# Helpers
# -----------------------------

def go(route_name: str):
    st.session_state.route = route_name


def parse_float_required(txt: str) -> float:
    """
    Required float field.
    Accepts '0.35' or '0,35' and converts to float using '.' internally.
    """
    s = (txt or "").strip().replace(",", ".")
    if s == "":
        raise ValueError("This field is required.")
    return float(s)


def parse_float_or_blank(txt: str):
    """
    Optional float field.
    Blank -> '' (so runner defaulting logic can kick in).
    Accepts '0.35' or '0,35'.
    """
    s = (txt or "").strip().replace(",", ".")
    if s == "":
        return ""
    return float(s)



def parse_text_or_blank(txt: str) -> str:
    return (txt or "").strip()


def mechanism_checkboxes(prefix: str):
    """Return selected mechanisms among ['Normal', 'Thrust', 'Strike-slip'].
    Empty list means ALL (no filtering).
    """
    st.markdown("Mechanism (style of faulting)")
    c1, c2, c3 = st.columns(3)
    with c1:
        normal = st.checkbox("Normal", key=f"{prefix}_mech_normal")
    with c2:
        thrust = st.checkbox("Thrust", key=f"{prefix}_mech_thrust")
    with c3:
        strike = st.checkbox("Strike-slip", key=f"{prefix}_mech_strike")

    selected = []
    if normal:
        selected.append("Normal")
    if thrust:
        selected.append("Thrust")
    if strike:
        selected.append("Strike-slip")
    return selected

def friendly_filtering_error(e: Exception) -> str:
    """
    Convert low-level numerical/index errors into a user-friendly message
    when filtering returns fewer records than requested.
    """
    msg = str(e).lower()

    keywords = [
        "out of bounds",
        "index",
        "shape",
        "axis",
        "empty",
        "cannot select",
        "size"
    ]

    if any(k in msg for k in keywords):
        return (
            "❌ The number of records remaining after filtering is smaller than the number "
            "requested.\n\n"
            "Please relax the filtering conditions (e.g., mechanism, Mw, distance, depth, "
            "soil type) or reduce the number of required records, and try again."
        )

    # Fallback: show original error
    # return f"❌ An unexpected error occurred:\n\n{e}"
    return (
        "❌ The number of records remaining after filtering is smaller than the number "
        "requested.\n\n"
        "Please relax the filtering conditions (e.g., mechanism, Mw, distance, depth, "
        "soil type) or reduce the number of required records, and try again."
    )


def compute_response_spectrum(acc_g, dt, periods=None, zeta: float = 0.05):
    """Compute 5%-damped elastic acceleration response spectrum from acceleration in g.

    The spectrum is computed from the final exported acceleration history, so the
    resulting ordinates remain fully consistent with the records written to the
    download package. If the record is PGA-normalized, then Sa(T=0) = 1.
    Returns (periods, Sa[g]).
    """
    from scipy.signal import cont2discrete

    acc_g = np.asarray(acc_g, dtype=float).flatten()
    if periods is None:
        periods = np.arange(0.0, 4.05, 0.05)
    periods = np.asarray(periods, dtype=float)

    if acc_g.size == 0:
        return periods, np.zeros_like(periods)

    sa = np.zeros_like(periods, dtype=float)
    pga = float(np.max(np.abs(acc_g))) if acc_g.size else 0.0

    for j, T in enumerate(periods):
        if T <= 1e-12:
            sa[j] = pga
            continue

        w = 2.0 * np.pi / float(T)
        A = np.array([[0.0, 1.0], [-w**2, -2.0 * zeta * w]], dtype=float)
        B = np.array([[0.0], [-1.0]], dtype=float)
        C = np.eye(2, dtype=float)
        D = np.zeros((2, 1), dtype=float)

        Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), dt, method="zoh")
        x = np.zeros(2, dtype=float)
        aabs_max = 0.0

        for ag in acc_g:
            aabs = -2.0 * zeta * w * x[1] - (w**2) * x[0]
            if abs(aabs) > aabs_max:
                aabs_max = abs(aabs)
            x = Ad @ x + Bd[:, 0] * ag

        aabs = -2.0 * zeta * w * x[1] - (w**2) * x[0]
        aabs_max = max(aabs_max, abs(aabs))
        sa[j] = aabs_max

    return periods, sa



def save_response_spectrum_plot_1d(acc_g, dt, out_path, periods=None, zeta: float = 0.05, title: str = "Scaled response spectrum (5% damping)"):
    import matplotlib.pyplot as plt

    periods, sa = compute_response_spectrum(acc_g, dt, periods=periods, zeta=zeta)
    figp = plt.figure()
    plt.plot(periods, sa)
    plt.xlabel("T [s]")
    plt.ylabel("Sa [g]")
    plt.title(title)
    plt.tight_layout()
    figp.savefig(out_path, dpi=150)
    plt.close(figp)



def save_response_spectrum_plot_2d(accE_g, accN_g, dt, out_path, periods=None, zeta: float = 0.05, title: str = "Scaled response spectra (5% damping)"):
    import matplotlib.pyplot as plt

    periods, saE = compute_response_spectrum(accE_g, dt, periods=periods, zeta=zeta)
    _, saN = compute_response_spectrum(accN_g, dt, periods=periods, zeta=zeta)

    figp = plt.figure()
    plt.plot(periods, saE, label="E")
    plt.plot(periods, saN, label="N")
    plt.xlabel("T [s]")
    plt.ylabel("Sa [g]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    figp.savefig(out_path, dpi=150)
    plt.close(figp)


def save_response_spectrum_plot_3d(accE_g, accN_g, accV_g, dt, out_path, periods=None, zeta: float = 0.05, title: str = "Scaled response spectra (5% damping)"):
    import matplotlib.pyplot as plt

    periods, saE = compute_response_spectrum(accE_g, dt, periods=periods, zeta=zeta)
    _, saN = compute_response_spectrum(accN_g, dt, periods=periods, zeta=zeta)
    _, saV = compute_response_spectrum(accV_g, dt, periods=periods, zeta=zeta)

    figp = plt.figure()
    plt.plot(periods, saE, label="E")
    plt.plot(periods, saN, label="N")
    plt.plot(periods, saV, label="V")
    plt.xlabel("T [s]")
    plt.ylabel("Sa [g]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    figp.savefig(out_path, dpi=150)
    plt.close(figp)


def compute_matching_score_from_fig(fig, T1: float, T2: float):
    """Compute a % matching score between mean and target spectra in [T1, T2].

    Score definition:
        score = max(0, (1 - RMSE(mean-target)/RMS(target)))*100

    Returns (score_percent, rmse_ratio) or (None, None) if unavailable.
    """
    try:
        if fig is None or not hasattr(fig, "axes") or len(fig.axes) == 0:
            return None, None

        ax = fig.axes[0]
        lines = list(getattr(ax, "lines", []))
        if len(lines) < 2:
            return None, None

        # Try to identify mean vs target robustly using line styles/widths
        mean_line = None
        target_line = None

        for ln in lines:
            ls = (ln.get_linestyle() or "").strip()
            lw = float(ln.get_linewidth() or 0.0)

            if target_line is None and ls == "--" and lw >= 1.5:
                target_line = ln
            if mean_line is None and ls == "-" and lw >= 1.5:
                mean_line = ln

        # Fallback: assume first two plotted lines are mean and target (as in runners)
        mean_line = mean_line or lines[0]
        target_line = target_line or (lines[1] if len(lines) > 1 else None)
        if target_line is None:
            return None, None

        Tm = np.asarray(mean_line.get_xdata(), dtype=float)
        Sm = np.asarray(mean_line.get_ydata(), dtype=float)
        Tt = np.asarray(target_line.get_xdata(), dtype=float)
        St = np.asarray(target_line.get_ydata(), dtype=float)

        # Ensure common x-grid (usually identical); otherwise interpolate target onto mean grid
        if Tm.shape != Tt.shape or np.max(np.abs(Tm - Tt)) > 1e-9:
            St = np.interp(Tm, Tt, St)

        mask = (Tm >= float(T1)) & (Tm <= float(T2))
        if not np.any(mask):
            return None, None

        dm = Sm[mask]
        dt = St[mask]

        rmse = float(np.sqrt(np.mean((dm - dt) ** 2)))
        denom = float(np.sqrt(np.mean(dt ** 2)))
        if denom <= 0:
            return None, None

        rmse_ratio = rmse / denom
        score = max(0.0, (1.0 - rmse_ratio)) * 100.0
        return score, rmse_ratio
    except Exception:
        return None, None


# -----------------------------
# Result caching (keeps results visible after download_button reruns)
# -----------------------------
def _results_key(route_name: str) -> str:
    return f"results::{route_name}"


def _safe_key(route_name: str) -> str:
    # Streamlit widget keys must be stable and simple
    return (
        route_name.replace("::", "_")
        .replace(":", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def clear_cached_results(route_name: str):
    """Remove cached results for a given route."""
    st.session_state.pop(_results_key(route_name), None)


def store_cached_results(route_name: str, out_df, fig=None, caption=None, file_name=None, zip_bytes=None, zip_file_name=None, zip_mime=None):
    """Store latest selection outputs so they persist across reruns (e.g., download_button clicks)."""
    st.session_state[_results_key(route_name)] = {
        "out_df": out_df,
        "fig": fig,
        "caption": caption,
        "file_name": file_name,
        "zip_bytes": zip_bytes,
        "zip_file_name": zip_file_name,
        "zip_mime": zip_mime,
    }


def render_cached_results(route_name: str, default_file_name: str):
    """Render cached selection outputs (if available) and a download button."""
    payload = st.session_state.get(_results_key(route_name))
    if not payload:
        return

    out_df = payload.get("out_df")
    fig = payload.get("fig")
    caption = payload.get("caption")
    file_name = payload.get("file_name") or default_file_name

    zip_bytes = payload.get("zip_bytes")
    zip_file_name = payload.get("zip_file_name")
    zip_mime = payload.get("zip_mime")

    if fig is not None:
        st.pyplot(fig, clear_figure=False)

    if out_df is not None:
        st.dataframe(out_df, use_container_width=True)

        if zip_bytes is not None and zip_file_name:
            st.download_button(
                "Download results",
                data=zip_bytes,
                file_name=zip_file_name,
                mime=zip_mime or "application/zip",
                key=f"dl_cached_{_safe_key(route_name)}",
            )
        else:
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results",
                data=csv_bytes,
                file_name=file_name,
                mime="text/csv",
                key=f"dl_cached_{_safe_key(route_name)}",
            )

    if caption:
        st.caption(caption)




# -----------------------------
# Scenario-based: package CSV + processed records into a single ZIP (cached)
# -----------------------------
from zipfile import ZipFile, BadZipFile

@st.cache_data(show_spinner=False)
def build_scenario_zip_cached(event_ids: tuple, station_ids: tuple, csv_text: str) -> bytes:
    """Build a ZIP containing metadata_filtered_SEE.csv + normalized E/N/V records for each selection."""
    import READER_ESM as readE  # local import to keep app startup light

    N = len(event_ids)
    failed = []  # (index, event_id, station_id, reason)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        base_folder = tmpdir / "Scenario_based"
        records_folder = tmpdir / "Scenario_based_records"
        base_folder.mkdir(parents=True, exist_ok=True)
        records_folder.mkdir(parents=True, exist_ok=True)

        durationE = np.zeros(N)
        durationN = np.zeros(N)
        durationV = np.zeros(N)
        dtE = np.zeros(N)
        dtN = np.zeros(N)
        dtV = np.zeros(N)

        for i in range(N):
            url = (
                "https://esm-db.eu/esmws/eventdata/1/query"
                f"?eventid={event_ids[i]}&station={station_ids[i]}&format=ascii&data-type=DIS"
            )

            out_dir = base_folder / str(i + 1)
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Download zip (ESM returns a .zip even when format=ascii)
                zip_path = out_dir / "query.zip"
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                zip_path.write_bytes(r.content)

                # Extract
                try:
                    with ZipFile(zip_path, "r") as zf:
                        zf.extractall(out_dir)
                except BadZipFile:
                    failed.append((i + 1, event_ids[i], station_ids[i], "BadZipFile (not a zip)"))
                    continue

                # Collect extracted files (exclude the zip itself)
                extracted = [p for p in out_dir.iterdir() if p.is_file() and p.name.lower() != "query.zip"]
                if len(extracted) == 0:
                    failed.append((i + 1, event_ids[i], station_ids[i], "No files extracted"))
                    continue

                # Try to identify E/N/V components robustly (HNE/HNN/HNZ naming is common in ESM)
                def _pick_component(tag: str):
                    tag = tag.upper()
                    for p in extracted:
                        name = p.name.upper()
                        if tag in name:
                            return p
                    return None

                fileE = _pick_component("HNE") or _pick_component("_E") or _pick_component("E.")
                fileN = _pick_component("HNN") or _pick_component("_N") or _pick_component("N.")
                fileV = (
                    _pick_component("HNZ")
                    or _pick_component("HNV")
                    or _pick_component("_Z")
                    or _pick_component("_V")
                    or _pick_component("Z.")
                    or _pick_component("V.")
                )
                if fileE is None or fileN is None or fileV is None:
                    # Fallback: use stable ordering only for missing components.
                    extracted_sorted = sorted(extracted, key=lambda p: p.name)
                    if len(extracted_sorted) >= 3:
                        fileE = fileE or extracted_sorted[0]
                        fileN = fileN or extracted_sorted[1]
                        fileV = fileV or extracted_sorted[2]
                    else:
                        failed.append((i + 1, event_ids[i], station_ids[i], "Not enough components"))
                        continue

                # Write normalized records (PGA=1) into Scenario_based_records/<i+1>/
                rec_dir = records_folder / str(i + 1)
                rec_dir.mkdir(parents=True, exist_ok=True)

                
                # Use displacement (cm) and derive PGA-normalized acceleration with dt=0.005 s
                dt = 0.005

                # East component
                dt_in, t_in, dispE_cm = readE.readESM(str(fileE))
                nE = len(dispE_cm)
                timeE = np.arange(nE, dtype=float) * dt
                dispE_m = np.asarray(dispE_cm, dtype=float) * 0.01
                velE_ms = np.gradient(dispE_m, dt)
                accE_ms2 = np.gradient(velE_ms, dt)
                accE_g = accE_ms2 / 9.80665
                PGA_E = float(np.max(np.abs(accE_g))) if nE > 0 and np.max(np.abs(accE_g)) != 0 else 1.0

                durationE[i] = timeE[-1] if nE > 0 else 0.0
                dtE[i] = dt

                # PGA-normalize (apply to disp/vel/acc consistently)
                dispE_mn = dispE_m / PGA_E
                velE_msn = velE_ms / PGA_E
                accE_gn = accE_g / PGA_E  # PGA-normalized -> max(|acc|)=1

                outE = np.column_stack((timeE, dispE_mn, velE_msn, accE_gn))
                np.savetxt(
                    rec_dir / f"{i+1}_E.txt",
                    outE,
                    fmt="%.3f %.6e %.6e %.6e",
                    header="t [s]    Dis [m]    Vel [m/s]    Acc [g]",
                    comments="",
                )

                # North component
                dt_in, t_in, dispN_cm = readE.readESM(str(fileN))
                nN = len(dispN_cm)
                timeN = np.arange(nN, dtype=float) * dt
                dispN_m = np.asarray(dispN_cm, dtype=float) * 0.01
                velN_ms = np.gradient(dispN_m, dt)
                accN_ms2 = np.gradient(velN_ms, dt)
                accN_g = accN_ms2 / 9.80665
                PGA_N = float(np.max(np.abs(accN_g))) if nN > 0 and np.max(np.abs(accN_g)) != 0 else 1.0

                durationN[i] = timeN[-1] if nN > 0 else 0.0
                dtN[i] = dt

                dispN_mn = dispN_m / PGA_N
                velN_msn = velN_ms / PGA_N
                accN_gn = accN_g / PGA_N

                outN = np.column_stack((timeN, dispN_mn, velN_msn, accN_gn))
                np.savetxt(
                    rec_dir / f"{i+1}_N.txt",
                    outN,
                    fmt="%.3f %.6e %.6e %.6e",
                    header="t [s]    Dis [m]    Vel [m/s]    Acc [g]",
                    comments="",
                )

                # Vertical component
                dt_in, t_in, dispV_cm = readE.readESM(str(fileV))
                nV = len(dispV_cm)
                timeV = np.arange(nV, dtype=float) * dt
                dispV_m = np.asarray(dispV_cm, dtype=float) * 0.01
                velV_ms = np.gradient(dispV_m, dt)
                accV_ms2 = np.gradient(velV_ms, dt)
                accV_g = accV_ms2 / 9.80665
                PGA_V = float(np.max(np.abs(accV_g))) if nV > 0 and np.max(np.abs(accV_g)) != 0 else 1.0

                durationV[i] = timeV[-1] if nV > 0 else 0.0
                dtV[i] = dt

                dispV_mn = dispV_m / PGA_V
                velV_msn = velV_ms / PGA_V
                accV_gn = accV_g / PGA_V

                outV = np.column_stack((timeV, dispV_mn, velV_msn, accV_gn))
                np.savetxt(
                    rec_dir / f"{i+1}_V.txt",
                    outV,
                    fmt="%.3f %.6e %.6e %.6e",
                    header="t [s]    Dis [m]    Vel [m/s]    Acc [g]",
                    comments="",
                )

                # Save quick-look plot (E/N/V acceleration)
                import matplotlib.pyplot as plt
                figp = plt.figure()
                plt.plot(timeE, accE_gn, label="E")
                plt.plot(timeN, accN_gn, label="N")
                plt.plot(timeV, accV_gn, label="V")
                plt.xlabel("t [s]")
                plt.ylabel("Acc [g]")
                plt.legend()
                plt.tight_layout()
                figp.savefig(rec_dir / f"{i+1}.png", dpi=150)
                plt.close(figp)
                
                # Save quick-look plot (E/N/V velocity)
                figp = plt.figure()
                plt.plot(timeE, velE_msn, label="E")
                plt.plot(timeN, velN_msn, label="N")
                plt.plot(timeV, velV_msn, label="V")
                plt.xlabel("t [s]")
                plt.ylabel("Vel [m/s]")
                plt.legend()
                plt.tight_layout()
                figp.savefig(rec_dir / f"{i+1}_vel.png", dpi=150)
                plt.close(figp)

                # Save quick-look plot (E/N/V displacement)
                figp = plt.figure()
                plt.plot(timeE, dispE_mn, label="E")
                plt.plot(timeN, dispN_mn, label="N")
                plt.plot(timeV, dispV_mn, label="V")
                plt.xlabel("t [s]")
                plt.ylabel("Disp [m]")
                plt.legend()
                plt.tight_layout()
                figp.savefig(rec_dir / f"{i+1}_disp.png", dpi=150)
                plt.close(figp)

                # Save PGA-normalized elastic response spectra for E/N/V components
                save_response_spectrum_plot_3d(
                    accE_g=accE_gn,
                    accN_g=accN_gn,
                    accV_g=accV_gn,
                    dt=dt,
                    out_path=rec_dir / f"{i+1}_RS.png",
                    # title="Normalized response spectra (5% damping)",
                )
            except Exception as e:
                failed.append((i + 1, event_ids[i], station_ids[i], f"{type(e).__name__}: {e}"))
                continue

        # Package everything into a single zip (in-memory)
        buf = io.BytesIO()
        with ZipFile(buf, "w") as zf:
            zf.writestr("metadata_filtered_SEE.csv", csv_text)

            # Include failure report (if any)
            if len(failed) > 0:
                report = "index,event_id,station_id,reason\n" + "\n".join(
                    f"{idx},{eid},{sid},{reason}" for idx, eid, sid, reason in failed
                ) + "\n"
                zf.writestr("FAILED_DOWNLOADS.csv", report)

            for p in records_folder.rglob("*"):
                if p.is_file():
                    arcname = str(p.relative_to(tmpdir))
                    zf.write(p, arcname=arcname)

        buf.seek(0)
        return buf.getvalue()

# -----------------------------
# 1D (spectral matching): package CSV + processed records into a single ZIP (cached)
# -----------------------------
from zipfile import ZipFile, BadZipFile

@st.cache_data(show_spinner=False)
def build_1d_zip_cached(event_ids: tuple, station_ids: tuple, comp: tuple, x: tuple, csv_text: str) -> bytes:
    """Build a ZIP containing selection CSV + processed 1D records (single component) for OpenSees."""
    import READER_ESM as readE  # local import to keep app startup light

    N = len(event_ids)
    if len(station_ids) != N or len(comp) != N or len(x) != N:
        raise ValueError("event_ids, station_ids, comp, and x must have the same length.")

    failed = []  # (index, event_id, station_id, reason)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        base_folder = tmpdir / "1D"
        records_folder = tmpdir / "1D_records"
        base_folder.mkdir(parents=True, exist_ok=True)
        records_folder.mkdir(parents=True, exist_ok=True)

        durationE = np.zeros(N)
        dtE = np.zeros(N)

        for i in range(N):
            url = (
                "https://esm-db.eu/esmws/eventdata/1/query"
                f"?eventid={event_ids[i]}&station={station_ids[i]}&format=ascii&data-type=DIS"
            )

            out_dir = base_folder / str(i + 1)
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Download zip (ESM returns a .zip even when format=ascii)
                zip_path = out_dir / "query.zip"
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                zip_path.write_bytes(r.content)

                # Extract
                try:
                    with ZipFile(zip_path, "r") as zf:
                        zf.extractall(out_dir)
                except BadZipFile:
                    failed.append((i + 1, event_ids[i], station_ids[i], "BadZipFile (not a zip)"))
                    continue

                # Collect extracted files (exclude the zip itself), stable order
                extracted = [p for p in out_dir.iterdir() if p.is_file() and p.name.lower() != "query.zip"]
                extracted_sorted = sorted(extracted, key=lambda p: p.name)
                if len(extracted_sorted) == 0:
                    failed.append((i + 1, event_ids[i], station_ids[i], "No files extracted"))
                    continue

                # Select component by index (comp[i]) using a stable ordering
                ci = int(comp[i])
                if ci < 0 or ci >= len(extracted_sorted):
                    ci = max(0, min(ci, len(extracted_sorted) - 1))
                selected_file = extracted_sorted[ci]

                # Process record: transform cm/s2 to m/s2 for OpenSees and apply scaling x[i]
                rec_dir = records_folder / str(i + 1)
                rec_dir.mkdir(parents=True, exist_ok=True)

                
                # Read displacement (cm), apply scaling, convert to m, then derive v/a with dt=0.005 s
                dt_in, t_in, disp_cm = readE.readESM(str(selected_file))
                # Note: ESM "DIS" files are provided in cm. We use a fixed dt=0.005 s for numerical differentiation.
                dt = 0.005
                n = len(disp_cm)
                time = np.arange(n, dtype=float) * dt

                scale = float(x[i])
                disp_m = np.asarray(disp_cm, dtype=float) * scale * 0.01  # Dis [m]

                # Numerical differentiation (central differences via np.gradient)
                vel_ms = np.gradient(disp_m, dt)            # Vel [m/s]
                acc_ms2 = np.gradient(vel_ms, dt)           # Acc [m/s2]
                acc_g = acc_ms2 / 9.80665                   # Acc [g]

                durationE[i] = time[-1] if n > 0 else 0.0
                dtE[i] = dt

                out = np.column_stack((time, disp_m, vel_ms, acc_g))
                np.savetxt(
                    rec_dir / f"{i+1}.txt",
                    out,
                    fmt="%.3f %.6e %.6e %.6e",
                    header="t [s]    Dis [m]    Vel [m/s]    Acc [g]",
                    comments="",
                )

                # Save quick-look plot (acceleration)
                import matplotlib.pyplot as plt
                figp = plt.figure()
                plt.plot(time, acc_g)
                plt.xlabel("t [s]")
                plt.ylabel("Acc [g]")
                plt.tight_layout()
                figp.savefig(rec_dir / f"{i+1}.png", dpi=150)
                plt.close(figp)
                
                # Save quick-look plot (velocity)
                import matplotlib.pyplot as plt
                figp = plt.figure()
                plt.plot(time, vel_ms)
                plt.xlabel("t [s]")
                plt.ylabel("Vel [m/s]")
                plt.tight_layout()
                figp.savefig(rec_dir / f"{i+1}_vel.png", dpi=150)
                plt.close(figp)

                # Save quick-look plot (displacement)
                import matplotlib.pyplot as plt
                figp = plt.figure()
                plt.plot(time, disp_m)
                plt.xlabel("t [s]")
                plt.ylabel("Disp [m]")
                plt.tight_layout()
                figp.savefig(rec_dir / f"{i+1}_disp.png", dpi=150)
                plt.close(figp)
                
                # Save scaled elastic response spectrum computed from the final scaled acceleration
                save_response_spectrum_plot_1d(
                    acc_g=acc_g,
                    dt=dt,
                    out_path=rec_dir / f"{i+1}_RS.png",
                )
            except Exception as e:
                failed.append((i + 1, event_ids[i], station_ids[i], f"{type(e).__name__}: {e}"))
                continue

        # Package everything into a single zip (in-memory)
        buf = io.BytesIO()
        with ZipFile(buf, "w") as zf:
            zf.writestr("metadata_filtered_SEE.csv", csv_text)

            # Include failure report (if any)
            if len(failed) > 0:
                report = "index,event_id,station_id,reason\n" + "\n".join(
                    f"{idx},{eid},{sid},{reason}" for idx, eid, sid, reason in failed
                ) + "\n"
                zf.writestr("FAILED_DOWNLOADS.csv", report)

            for p in records_folder.rglob("*"):
                if p.is_file():
                    arcname = str(p.relative_to(tmpdir))
                    zf.write(p, arcname=arcname)

        buf.seek(0)
        return buf.getvalue()



# -----------------------------
# 2D (spectral matching): package CSV + processed records into a single ZIP (cached)
# -----------------------------
from zipfile import ZipFile, BadZipFile

@st.cache_data(show_spinner=False)
def build_2d_zip_cached(event_ids: tuple, station_ids: tuple, x: tuple, csv_text: str) -> bytes:
    """Build a ZIP containing selection CSV + processed 2D records (E/N components) for OpenSees."""
    import READER_ESM as readE  # local import to keep app startup light

    N = len(event_ids)
    if len(station_ids) != N or len(x) != N:
        raise ValueError("event_ids, station_ids, and x must have the same length.")

    failed = []  # (index, event_id, station_id, reason)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        base_folder = tmpdir / "2D"
        records_folder = tmpdir / "2D_records"
        base_folder.mkdir(parents=True, exist_ok=True)
        records_folder.mkdir(parents=True, exist_ok=True)

        durationE = np.zeros(N)
        durationN = np.zeros(N)
        dtE = np.zeros(N)
        dtN = np.zeros(N)

        for i in range(N):
            url = (
                "https://esm-db.eu/esmws/eventdata/1/query"
                f"?eventid={event_ids[i]}&station={station_ids[i]}&format=ascii&data-type=DIS"
            )

            out_dir = base_folder / str(i + 1)
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Download zip (ESM returns a .zip even when format=ascii)
                zip_path = out_dir / "query.zip"
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                zip_path.write_bytes(r.content)

                # Extract
                try:
                    with ZipFile(zip_path, "r") as zf:
                        zf.extractall(out_dir)
                except BadZipFile:
                    failed.append((i + 1, event_ids[i], station_ids[i], "BadZipFile (not a zip)"))
                    continue

                # Collect extracted files (exclude the zip itself)
                extracted = [p for p in out_dir.iterdir() if p.is_file() and p.name.lower() != "query.zip"]
                if len(extracted) == 0:
                    failed.append((i + 1, event_ids[i], station_ids[i], "No files extracted"))
                    continue

                # Try to identify E/N components robustly (HNE/HNN naming is common in ESM)
                def _pick_component(tag: str):
                    tag = tag.upper()
                    for p in extracted:
                        name = p.name.upper()
                        if tag in name:
                            return p
                    return None

                fileE = _pick_component("HNE") or _pick_component("_E") or _pick_component("E.")
                fileN = _pick_component("HNN") or _pick_component("_N") or _pick_component("N.")
                if fileE is None or fileN is None:
                    extracted_sorted = sorted(extracted, key=lambda p: p.name)
                    if len(extracted_sorted) >= 2:
                        fileE = fileE or extracted_sorted[0]
                        fileN = fileN or extracted_sorted[1]
                    else:
                        failed.append((i + 1, event_ids[i], station_ids[i], "Not enough components"))
                        continue

                # Process records: transform cm/s2 to m/s2 for OpenSees and apply scaling x[i]
                rec_dir = records_folder / str(i + 1)
                rec_dir.mkdir(parents=True, exist_ok=True)

                
                scale = float(x[i])

                # Read displacement (cm), apply scaling, convert to m, then derive v/a with dt=0.005 s
                dt = 0.005

                # East component
                dt_in, t_in, dispE_cm = readE.readESM(str(fileE))
                nE = len(dispE_cm)
                timeE = np.arange(nE, dtype=float) * dt
                dispE_m = np.asarray(dispE_cm, dtype=float) * scale * 0.01
                velE_ms = np.gradient(dispE_m, dt)
                accE_ms2 = np.gradient(velE_ms, dt)
                accE_g = accE_ms2 / 9.80665

                durationE[i] = timeE[-1] if nE > 0 else 0.0
                dtE[i] = dt

                outE = np.column_stack((timeE, dispE_m, velE_ms, accE_g))
                np.savetxt(
                    rec_dir / f"{i+1}_E.txt",
                    outE,
                    fmt="%.3f %.6e %.6e %.6e",
                    header="t [s]    Dis [m]    Vel [m/s]    Acc [g]",
                    comments="",
                )

                # North component
                dt_in, t_in, dispN_cm = readE.readESM(str(fileN))
                nN = len(dispN_cm)
                timeN = np.arange(nN, dtype=float) * dt
                dispN_m = np.asarray(dispN_cm, dtype=float) * scale * 0.01
                velN_ms = np.gradient(dispN_m, dt)
                accN_ms2 = np.gradient(velN_ms, dt)
                accN_g = accN_ms2 / 9.80665

                durationN[i] = timeN[-1] if nN > 0 else 0.0
                dtN[i] = dt

                outN = np.column_stack((timeN, dispN_m, velN_ms, accN_g))
                np.savetxt(
                    rec_dir / f"{i+1}_N.txt",
                    outN,
                    fmt="%.3f %.6e %.6e %.6e",
                    header="t [s]    Dis [m]    Vel [m/s]    Acc [g]",
                    comments="",
                )

                # Save quick-look plot (both components acceleration)
                import matplotlib.pyplot as plt
                figp = plt.figure()
                plt.plot(timeE, accE_g, label="E")
                plt.plot(timeN, accN_g, label="N")
                plt.xlabel("t [s]")
                plt.ylabel("Acc [g]")
                plt.legend()
                plt.tight_layout()
                figp.savefig(rec_dir / f"{i+1}.png", dpi=150)
                plt.close(figp)
                
                # Save quick-look plot (both components velocity)
                figp = plt.figure()
                plt.plot(timeE, velE_ms, label="E")
                plt.plot(timeN, velN_ms, label="N")
                plt.xlabel("t [s]")
                plt.ylabel("Vel [m/s]")
                plt.legend()
                plt.tight_layout()
                figp.savefig(rec_dir / f"{i+1}_vel.png", dpi=150)
                plt.close(figp)

                # Save quick-look plot (both components displacement)
                figp = plt.figure()
                plt.plot(timeE, dispE_m, label="E")
                plt.plot(timeN, dispN_m, label="N")
                plt.xlabel("t [s]")
                plt.ylabel("Disp [m]")
                plt.legend()
                plt.tight_layout()
                figp.savefig(rec_dir / f"{i+1}_disp.png", dpi=150)
                plt.close(figp)

                # Save scaled elastic response spectra for both horizontal components
                save_response_spectrum_plot_2d(
                    accE_g=accE_g,
                    accN_g=accN_g,
                    dt=dt,
                    out_path=rec_dir / f"{i+1}_RS.png",
                )
            except Exception as e:
                failed.append((i + 1, event_ids[i], station_ids[i], f"{type(e).__name__}: {e}"))
                continue

        # Package everything into a single zip (in-memory)
        buf = io.BytesIO()
        with ZipFile(buf, "w") as zf:
            zf.writestr("metadata_filtered_SEE.csv", csv_text)

            # Include failure report (if any)
            if len(failed) > 0:
                report = "index,event_id,station_id,reason\n" + "\n".join(
                    f"{idx},{eid},{sid},{reason}" for idx, eid, sid, reason in failed
                ) + "\n"
                zf.writestr("FAILED_DOWNLOADS.csv", report)

            for p in records_folder.rglob("*"):
                if p.is_file():
                    arcname = str(p.relative_to(tmpdir))
                    zf.write(p, arcname=arcname)

        buf.seek(0)
        return buf.getvalue()

# -----------------------------
# Router state
# -----------------------------
if "route" not in st.session_state:
    st.session_state.route = "home"

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .titlebar {
        background:#6f6f6f;
        color:white;
        padding:14px 18px;
        border-radius:4px;
        font-size:28px;
        font-weight:800;
        margin-bottom:10px;
    }
    div.stButton > button {
        width: 100%;
        height: 72px;
        font-size: 22px !important;
        font-weight: 800 !important;
        border-radius: 14px !important;
    }

    /* 🔴 ADD THIS */
    .stop-server-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# st.markdown('<div class="titlebar">PEUSP_beta</div>', unsafe_allow_html=True)
app_dir = Path(__file__).parent

h1, h2 = st.columns([1, 10])

with h1:
    st.image(str(app_dir / "PEUSP.png"), width=200)

with h2:
    st.markdown('<div class="titlebar">PEUSP_beta</div>', unsafe_allow_html=True)


# ---- Stop server button (kills the Streamlit process)
st.markdown('<div class="stop-server-container">', unsafe_allow_html=True)
if st.button("⛔ Stop server", key="stop_server"):
    st.warning("Stopping Streamlit server...")
    os._exit(0)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# HOME
# -----------------------------
if st.session_state.route == "home":
    with st.container():
        st.markdown('<div class="box">', unsafe_allow_html=True)

        st.markdown(
            "<h3 style='margin-top:0; margin-bottom:20px;'>Select workflow</h3>",
            unsafe_allow_html=True
        )

        col = st.columns([1, 3, 1])[1]
        with col:
            if st.button("Spectral matching", key="home_spectral"):
                go("spectral")
            st.write("")
            if st.button("Scenario-based", key="home_scenario"):
                go("scenario")  # placeholder

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# SPECTRAL MATCHING (choose 1D/2D)
# -----------------------------
elif st.session_state.route == "spectral":
    st.subheader("Spectral matching")
    st.caption("Selection dimensionality")

    app_dir = Path(__file__).parent  # same folder as app.py

    c1, c2 = st.columns(2)

    with c1:
        st.image(str(app_dir / "1D.png"), width=170)
        if st.button("1D", key="spec_1d"):
            go("spectral_1d_menu")

    with c2:
        st.image(str(app_dir / "2D.png"), width=195)
        if st.button("2D", key="spec_2d"):
            go("spectral_2d_menu")

    st.write("")
    if st.button("⬅️ Back", key="back_from_spectral"):
        go("home")


# -----------------------------
# 1D MENU (choose 2004/2024)
# -----------------------------
elif st.session_state.route == "spectral_1d_menu":
    st.subheader("Spectral matching → 1D")
    st.caption("Choose code version")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("EN 1998-1-1:2004", key="1d_2004"):
            go("spectral_1d_2004")
    with c2:
        if st.button("EN 1998-1-1:2024", key="1d_2024"):
            go("spectral_1d_2024")
    with c3:
        if st.button("User-defined target", key="1d_user"):
            go("spectral_1d_user")

    st.write("")
    if st.button("⬅️ Back", key="back_1d_menu"):
        go("spectral")

# -----------------------------
# 2D MENU (choose 2004/2024)
# -----------------------------
elif st.session_state.route == "spectral_2d_menu":
    st.subheader("Spectral matching → 2D")
    st.caption("Choose code version")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("EN 1998-1-1:2004", key="2d_2004"):
            go("spectral_2d_2004")
    with c2:
        if st.button("EN 1998-1-1:2024", key="2d_2024"):
            go("spectral_2d_2024")
    with c3:
        if st.button("User-defined target", key="2d_user"):
            go("spectral_2d_user")

    st.write("")
    if st.button("⬅️ Back", key="back_2d_menu"):
        go("spectral")


# -----------------------------
# 1D MENU (User-defined target) — placeholder
# -----------------------------
elif st.session_state.route == "spectral_1d_user":
    st.subheader("Spectral matching → 1D → User-defined target")
    st.caption("User-defined target and conditions — record selection & scaling (single horizontal component selection)")

    app_dir = Path(__file__).parent
    default_metadata = app_dir / "metadata.csv"

    # Fixed internal period grid (must match user_1D_runner)
    T_grid = np.arange(start=0, stop=4.05, step=0.05)

    with st.sidebar:
        st.markdown("## Records & matching conditions")
        st.caption("Use '.' for decimals (',' also accepted). Target file can be space- or tab-delimited.")

        N_records = st.number_input(
            "Number of records",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            key="USER1D_N_records",
            help="Number of ground motions to select; preferable ≥ 7.",
        )

        T1 = parse_float_or_blank(
            st.text_input(
                label="T1 [s]",
                value="",
                key="USER1D_T1",
                help="Lower bound of the matching range. This value should be structure dependent, e.g., 0.00 for masonry (i.e., full domain).",
            )
        )
        T2 = parse_float_or_blank(
            st.text_input(
                label="T2 [s]",
                value="",
                key="USER1D_T2",
                help="Upper bound of the matching range. This value should be structure dependent, e.g., 4.00 for masonry (i.e., full domain).",
            )
        )

        Soil_Type_filtering = st.selectbox(
            "Soil type filtering",
            ["A", "B", "C", "D", "ALL"],
            index=4,
            key="USER1D_Soil_Type_filtering",
            help="Used to filter the database by Vs30 bounds.",
        )

        st.markdown("---")
        st.markdown("## User-defined target spectrum")
        target_file = st.file_uploader(
            "Upload target spectrum (.txt)",
            type=["txt", "csv", "dat"],
            key="USER1D_target_upload",
            help="Two columns: T [s] and Sa [g]; preferable ΔT = 0.05 s but any spacing is allowed.",
        )

        Min_scalling = parse_float_required(
            st.text_input(
                "Min scaling",
                value="0.50",
                key="USER1D_Min_scalling",
                help="Lower bound of the scaling factor for each record. A value of 0.50 is recommended.",
            )
        )
        Max_scalling = parse_float_required(
            st.text_input(
                "Max scaling",
                value="2.00",
                key="USER1D_Max_scalling",
                help="Upper bound of the scaling factor for each record. A value of 2.00 is recommended.",
            )
        )

        Min_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean lower limit",
                value="0.80",
                key="USER1D_Min_Target_MEAN",
                help="Lower bound for the mean spectrum (relative to target). Values in the range 0.75–0.90 are recommended.",
            )
        )
        Max_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean upper limit",
                value="1.20",
                key="USER1D_Max_Target_MEAN",
                help="Upper bound for the mean spectrum (relative to target). Values in the range 1.10–1.30 are recommended.",
            )
        )

        Min_Target_ind = parse_float_required(
            st.text_input(
                "Individual lower limit",
                value="0.50",
                key="USER1D_Min_Target_ind",
                help="Lower bound for each individual spectrum (relative to target). A value of 0.50 is recommended (i.e. ±50 % of the target spectrum).",
            )
        )
        Max_Target_ind = parse_float_required(
            st.text_input(
                "Individual upper limit",
                value="2.00",
                key="USER1D_Max_Target_ind",
                help="Upper bound for each individual spectrum (relative to target). A value of 2.00 is recommended (i.e. ±50 % of the target spectrum).",
            )
        )

        st.markdown("---")
        st.markdown("## Seismological filtering conditions")
        st.caption("Optional fields")

        Style_of_Faulting = mechanism_checkboxes("USER1D")

        Mw_lower_bound = parse_float_or_blank(
            st.text_input(
                label="Mw lower bound",
                value="",
                key="USER1D_Mw_lower",
                help="Moment magnitude lower bound. Leave blank to ignore.",
            )
        )
        Mw_upper_bound = parse_float_or_blank(
            st.text_input(
                label="Mw upper bound",
                value="",
                key="USER1D_Mw_upper",
                help="Moment magnitude upper bound. Leave blank to ignore.",
            )
        )

        Rjb_lower_bound = parse_float_or_blank(
            st.text_input(
                label="R lower bound [km]",
                value="",
                key="USER1D_Rjb_lower",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )
        Rjb_upper_bound = parse_float_or_blank(
            st.text_input(
                label="R upper bound [km]",
                value="",
                key="USER1D_Rjb_upper",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )

        depth_lower_bound = parse_float_or_blank(
            st.text_input("Depth lower bound [km]", value="", key="USER1D_depth_lower")
        )
        depth_upper_bound = parse_float_or_blank(
            st.text_input("Depth upper bound [km]", value="", key="USER1D_depth_upper")
        )

        st.markdown("---")
        metadata_path = st.text_input(
            "metadata.csv path", value=str(default_metadata), key="USER1D_metadata_path"
        )

        # seed = st.number_input(
        #     "Random seed",
        #     min_value=0,
        #     max_value=10_000,
        #     value=1,
        #     step=1,
        #     key="USER1D_seed",
        #     help="Seed for reproducibility.",
        # )

    # ---- Read + interpolate user spectrum
    Target_user = None
    if target_file is not None:
        try:
            raw = target_file.getvalue().decode("utf-8", errors="ignore")
            arr = np.genfromtxt(io.StringIO(raw), comments="#", delimiter=None)

            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError("Target file must contain at least two columns: T and Sa.")

            T_in = arr[:, 0].astype(float)
            Sa_in = arr[:, 1].astype(float)

            # Drop NaNs
            mask = np.isfinite(T_in) & np.isfinite(Sa_in)
            T_in, Sa_in = T_in[mask], Sa_in[mask]

            if T_in.size < 2:
                raise ValueError("Target file must contain at least two valid rows.")

            # Sort by period and remove duplicates (keep first)
            order = np.argsort(T_in)
            T_in, Sa_in = T_in[order], Sa_in[order]
            _, uniq_idx = np.unique(T_in, return_index=True)
            T_in, Sa_in = T_in[uniq_idx], Sa_in[uniq_idx]

            if T_in.min() > 0.0 + 1e-9 or T_in.max() < 4.0 - 1e-9:
                raise ValueError("Target T range must cover at least 0.0 to 4.0 seconds.")

            Sa_interp = np.interp(T_grid, T_in, Sa_in).astype(float)
            Target_user = Sa_interp

            st.success("Target spectrum loaded and interpolated to ΔT = 0.05 s (0–4 s).")
            with st.expander("Preview target spectrum"):
                st.write("Interpolated target (first 10 points):", Sa_interp[:10])
        except Exception as e:
            st.error(f"Could not read target file: {e}")
            Target_user = None

    # Main buttons
    c_run, c_back = st.columns([2, 1])
    with c_run:
        run = st.button("Run selection", type="primary", key="run_1d_user")
    with c_back:
        if st.button("⬅️ Back", key="back_1d_user"):
            clear_cached_results("spectral_1d_user")
            go("spectral_1d_menu")
            st.stop()

    if run:
        if Target_user is None:
            st.error("Please upload a valid target spectrum file before running.")
        else:
            try:
                with st.spinner("Running User-defined target (1D)..."):
                    out_df, fig, result, comp, x = user_1D_runner.run_user_1d(
                        metadata_path=metadata_path,
                        N_records=int(N_records),
                        T1=float(T1),
                        T2=float(T2),
                        Soil_Type_filtering=str(Soil_Type_filtering),
                        Target_user=Target_user,
                        Min_scalling=float(Min_scalling),
                        Max_scalling=float(Max_scalling),
                        Min_Target_MEAN=float(Min_Target_MEAN),
                        Max_Target_MEAN=float(Max_Target_MEAN),
                        Min_Target_ind=float(Min_Target_ind),
                        Max_Target_ind=float(Max_Target_ind),
                        Style_of_Faulting=Style_of_Faulting,
                        Mw_lower_bound=Mw_lower_bound,
                        Mw_upper_bound=Mw_upper_bound,
                        Rjb_lower_bound=Rjb_lower_bound,
                        Rjb_upper_bound=Rjb_upper_bound,
                        depth_lower_bound=depth_lower_bound,
                        depth_upper_bound=depth_upper_bound,
                        # seed=int(seed),
                    )

                st.success("Selection completed.")

                if fig is not None:
                    st.pyplot(fig, clear_figure=False)

                st.dataframe(out_df, use_container_width=True)

                score, rmse_ratio = compute_matching_score_from_fig(fig, float(T1), float(T2))
                score_txt = f"{score:.1f}%" if score is not None else "n/a"

                caption_txt = (
                    f"Optimizer success: {getattr(result, 'success', None)} | "
                    f"iterations: {getattr(result, 'nit', None)} | "
                    f"matching score (T1–T2): {score_txt}"
                )
                # One-click deliverable: ZIP = metadata_filtered_SEE.csv + processed (OpenSees-ready) records
                csv_text = out_df.to_csv(index=False)
                event_ids = tuple(out_df["Event_ID"].astype(str).to_numpy())
                station_ids = tuple(out_df["Station_ID"].astype(str).to_numpy())
                comp_t = tuple(np.asarray(comp, dtype=int).tolist())
                x_t = tuple(np.asarray(x, dtype=float).tolist())

                with st.spinner("Preparing download package (CSV + records)..."):
                    zip_bytes = build_1d_zip_cached(event_ids, station_ids, comp_t, x_t, csv_text)

                store_cached_results(
                    route_name="spectral_1d_user",
                    out_df=out_df,
                    fig=fig,
                    caption=caption_txt,
                    file_name="PEUSP_UserTarget_1D_selection.csv",
                    zip_bytes=zip_bytes,
                    zip_file_name="PEUSP_UserTarget_1D_selection.zip",
                    zip_mime="application/zip",
                )

                st.download_button(
                    "Download results",
                    data=zip_bytes,
                    file_name="PEUSP_UserTarget_1D_selection.zip",
                    mime="application/zip",
                    key="dl_1d_user_zip",
                )

                st.caption(
                    f"Optimizer success: {getattr(result, 'success', None)} | "
                    f"iterations: {getattr(result, 'nit', None)} | "
                    f"matching score (T1–T2): {score_txt}"
                )

            except Exception as e:
                st.error(friendly_filtering_error(e))

    else:
        render_cached_results("spectral_1d_user", "PEUSP_UserTarget_1D_selection.csv")


# ============================================================
# IMPLEMENTED: Spectral matching → 1D → EN 1998-1-1:2004
# ============================================================
elif st.session_state.route == "spectral_1d_2004":
    st.subheader("Spectral matching → 1D → EN 1998-1-1:2004")
    st.caption("EC8:2004 — record selection & scaling (single horizontal component selection)")

    app_dir = Path(__file__).parent
    default_metadata = app_dir / "metadata.csv"

    with st.sidebar:
        st.markdown("## Records & matching conditions")
        st.caption("Use '.' for decimals (',' also accepted).")

        N_records = st.number_input(
            "Number of records",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            help="When only three input motions are used, the structural demand is determined from the most unfavourable value produced from the corresponding three dynamic analyses.",
        )

        T1 = parse_float_or_blank(
            st.text_input(
                label="T1 [s]",
                value="",
                key="1D2004_T1",
                help="Lower bound of the matching range, recommended in EN 1998-1-1:2004 as T2 = 0.2T, with T as the fundamental period of the structure.",
            )
        )
        T2 = parse_float_or_blank(
            st.text_input(
                label="T2 [s]",
                value="",
                key="1D2004_T2",
                help="Upper bound of the matching range, recommended in EN 1998-1-1:2004 as T2 = 2T, with T as the fundamental period of the structure.",
            )
        )

        Soil_Type_filtering = st.selectbox(
            "Soil type filtering",
            ["A", "B", "C", "D", "ALL"],
            index=4,
            key="1D2004_Soil_Type_filtering",
            help="Used to filter the database by Vs30 bounds.",
        )

        # ---- EC8:2004 target spectrum parameters
        agr = parse_float_or_blank(
            st.text_input(
                "agR [g]",
                value="",
                key="1D2004_agr",
                help="Reference peak ground acceleration on type A ground (in g).",
            )
        )
        Ifactor = parse_float_required(
            st.text_input(
                "γI",
                value="1.00",
                key="1D2004_Ifactor",
                help="Importance factor.",
            )
        )
        Type = st.selectbox(
            "Spectrum type",
            ["Type1", "Type2"],
            index=0,
            key="1D2004_Type",
            help="Type1 for Mw > 5.5, Type2 for Mw < 5.5.",
        )
        Soil_Type_spectrum = st.selectbox(
            "Soil type (target spectrum)",
            ["A", "B", "C", "D", "E"],
            index=0,
            key="1D2004_Soil_Type_spectrum",
            help="Used to compute the EN 1998-1-1:2004 target response spectrum.",
        )

        Min_scalling = parse_float_or_blank(
            st.text_input(
                "Min scaling",
                value="0.50",
                key="1D2004_Min_scalling",
                help="No recommendations are provided in EN 1998-1-1:2004. A value of 0.50 is recomended in EN 1998-1-1:2024 for the lower bound of the scaling factor of each signal.",
            )
        )
        Max_scalling = parse_float_required(
            st.text_input(
                "Max scaling",
                value="2.00",
                key="1D2004_Max_scalling",
                help="No recommendations are provided in EN 1998-1-1:2004. A value of 2.00 is recomended in EN 1998-1-1:2024 for the upper bound of the scaling factor of each signal.",
            )
        )

        Min_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean lower limit",
                value="0.90",
                key="1D2004_Min_Target_MEAN",
                help="Lower bound for the mean spectrum (relative to target). A value of 0.90 is recomended in EN 1998-1-1:2004.",
            )
        )
        Max_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean upper limit",
                value="1.20",
                key="1D2004_Max_Target_MEAN",
                help="Upper bound for the mean spectrum (relative to target). A value of 1.20 is recomended in EN 1998-1-1:2004.",
            )
        )

        Min_Target_ind = parse_float_required(
            st.text_input(
                "Individual lower limit",
                value="0.50",
                key="1D2004_Min_Target_ind",
                help="Lower bound for each individual spectrum (relative to target). No recommendations are provided in EN 1998-1-1:2004. A value of 0.50 is recomended in EN 1998-1-1:2024 (i.e. ±50 % of the target spectrum).",
            )
        )
        Max_Target_ind = parse_float_required(
            st.text_input(
                "Individual upper limit",
                value="2.00",
                key="1D2004_Max_Target_ind",
                help="Upper bound for each individual spectrum (relative to target). No recommendations are provided in EN 1998-1-1:2004. A value of 2.00 is recomended in EN 1998-1-1:2024 (i.e. ±50 % of the target spectrum).",
            )
        )

        st.markdown("---")
        st.markdown("## Seismological filtering conditions")
        st.caption("Optional fields")

        Style_of_Faulting = mechanism_checkboxes("1D2004")

        Mw_lower_bound = parse_float_or_blank(
            st.text_input(
                label="Mw lower bound",
                value="",
                key="1D2004_Mw_lower",
                help="Moment magnitude lower bound. Leave blank to ignore.",
            )
        )
        Mw_upper_bound = parse_float_or_blank(
            st.text_input(
                label="Mw upper bound",
                value="",
                key="1D2004_Mw_upper",
                help="Moment magnitude upper bound. Leave blank to ignore.",
            )
        )

        Rjb_lower_bound = parse_float_or_blank(
            st.text_input(
                label="R lower bound [km]",
                value="",
                key="1D2004_Rjb_lower",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )
        Rjb_upper_bound = parse_float_or_blank(
            st.text_input(
                label="R upper bound [km]",
                value="",
                key="1D2004_Rjb_upper",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )

        depth_lower_bound = parse_float_or_blank(
            st.text_input("Depth lower bound [km]", value="", key="1D2004_depth_lower")
        )
        depth_upper_bound = parse_float_or_blank(
            st.text_input("Depth upper bound [km]", value="", key="1D2004_depth_upper")
        )

        st.markdown("---")
        metadata_path = st.text_input(
            "metadata.csv path", value=str(default_metadata), key="1D2004_metadata_path"
        )

    # Main buttons
    c_run, c_back = st.columns([2, 1])
    with c_run:
        run = st.button("Run selection", type="primary", key="run_1d_2004")
    with c_back:
        if st.button("⬅️ Back", key="back_1d_2004"):
            clear_cached_results("spectral_1d_2004")
            go("spectral_1d_menu")
            st.stop()

    if run:
        try:
            with st.spinner("Running EC8:2004 (1D)..."):
                out_df, fig, result, comp, x = EC82004_1D_runner.run_EC82024_1d_ec82004(
                    metadata_path=metadata_path,
                    N_records=int(N_records),
                    T1=float(T1),
                    T2=float(T2),
                    Soil_Type_filtering=str(Soil_Type_filtering),
                    agr=float(agr),
                    Ifactor=float(Ifactor),
                    Type=str(Type),
                    Soil_Type_spectrum=str(Soil_Type_spectrum),
                    Min_scalling=float(Min_scalling),
                    Max_scalling=float(Max_scalling),
                    Min_Target_MEAN=float(Min_Target_MEAN),
                    Max_Target_MEAN=float(Max_Target_MEAN),
                    Min_Target_ind=float(Min_Target_ind),
                    Max_Target_ind=float(Max_Target_ind),
                    Style_of_Faulting=Style_of_Faulting,
                    Mw_lower_bound=Mw_lower_bound,
                    Mw_upper_bound=Mw_upper_bound,
                    Rjb_lower_bound=Rjb_lower_bound,
                    Rjb_upper_bound=Rjb_upper_bound,
                    depth_lower_bound=depth_lower_bound,
                    depth_upper_bound=depth_upper_bound,
                )

            st.success("Selection completed.")

            if fig is not None:
                st.pyplot(fig, clear_figure=False)

            st.dataframe(out_df, use_container_width=True)

            score, rmse_ratio = compute_matching_score_from_fig(fig, float(T1), float(T2))
            score_txt = f"{score:.1f}%" if score is not None else "n/a"

            caption_txt = (
                f"Optimizer success: {getattr(result, 'success', None)} | "
                f"iterations: {getattr(result, 'nit', None)} | "
                f"matching score (T1–T2): {score_txt}"
            )
            # One-click deliverable: ZIP = metadata_filtered_SEE.csv + processed (OpenSees-ready) records
            csv_text = out_df.to_csv(index=False)
            event_ids = tuple(out_df["Event_ID"].astype(str).to_numpy())
            station_ids = tuple(out_df["Station_ID"].astype(str).to_numpy())
            comp_t = tuple(np.asarray(comp, dtype=int).tolist())
            x_t = tuple(np.asarray(x, dtype=float).tolist())

            with st.spinner("Preparing download package (CSV + records)..."):
                zip_bytes = build_1d_zip_cached(event_ids, station_ids, comp_t, x_t, csv_text)

            store_cached_results(
                route_name="spectral_1d_2004",
                out_df=out_df,
                fig=fig,
                caption=caption_txt,
                file_name="PEUSP_EC8_2004_1D_selection.csv",
                zip_bytes=zip_bytes,
                zip_file_name="PEUSP_EC8_2004_1D_selection.zip",
                zip_mime="application/zip",
            )

            st.download_button(
                "Download results",
                data=zip_bytes,
                file_name="PEUSP_EC8_2004_1D_selection.zip",
                mime="application/zip",
                key="dl_1d_2004_zip",
            )

            st.caption(
                f"Optimizer success: {getattr(result, 'success', None)} | "
                f"iterations: {getattr(result, 'nit', None)} | "
                f"matching score (T1–T2): {score_txt}"
            )

        except Exception as e:
            st.error(friendly_filtering_error(e))

    else:
        render_cached_results("spectral_1d_2004", "PEUSP_EC8_2004_1D_selection.csv")

# ============================================================
# IMPLEMENTED: Spectral matching → 1D → EN 1998-1-1:2024
# (UNCHANGED)
# ============================================================
elif st.session_state.route == "spectral_1d_2024":
    st.subheader("Spectral matching → 1D → EN 1998-1-1:2024")
    st.caption("EC8:2024 Annex D — record selection & scaling (single horizontal component selection)")

    app_dir = Path(__file__).parent
    default_metadata = app_dir / "metadata.csv"

    with st.sidebar:
        st.markdown("## Records & matching conditions")
        st.caption("Use '.' for decimals (',' also accepted).")

        N_records = st.number_input(
            "Number of records",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            help="The minimum number of input motions may be reduced to three, considering the maximum response, only for low and very low seismic action classes.",
        )

        T1 = parse_float_or_blank(
            st.text_input(
                label="T1 [s]",
                value="",
                key="1D_T1",
                help="Lower bound of the matching range, recommended in EN 1998-1-1:2024 as T1 = 0.2T, with T as the fundamental period of the structure.",
            )
        )
        T2 = parse_float_or_blank(
            st.text_input(
                label="T2 [s]",
                value="",
                key="1D_T2",
                help="Upper bound of the matching range, recommended in EN 1998-1-1:2024 as T2 = 1.5T, with T as the fundamental period of the structure.",
            )
        )

        Soil_Type_filtering = st.selectbox(
            "Soil type filtering",
            ["A", "B", "C", "D", "ALL"],
            index=4,
            key="1D_Soil_Type_filtering",
            help="Used to filter the database by Vs30 bounds.",
        )

        S_alpha_RP = parse_float_or_blank(
            st.text_input(
                label="Sα,RP [g]",
                value="",
                key="1D_S_alpha_RP",
                help="Maximum response spectral acceleration, for 5% damping, at the constant acceleration range of the elastic response spectrum. Explore the latest European Seismic Hazard Model 2020 (ESHM20) at hazard.efehr.org",
            )
        )

        S_beta_RP = parse_float_or_blank(
            st.text_input(
                label="Sᵦ,RP [g]",
                value="",
                key="1D_S_beta_RP",
                help="5%-damped response spectral acceleration at the vibration period Tβ = 1 s. Explore the latest European Seismic Hazard Model 2020 (ESHM20) at hazard.efehr.org",
            )
        )

        Soil_Type_spectrum = st.selectbox(
            "Soil type (target spectrum)",
            ["A", "B", "C", "D"],
            index=0,
            key="1D_Soil_Type_spectrum",
            help="Used to compute the EN 1998-1-1:2024 target response spectrum.",
        )

        Min_scalling = parse_float_required(
            st.text_input(
                "Min scaling",
                value="0.50",
                key="1D_Min_scalling",
                help="Lower bound of the scaling factor of each signal, recommended as 0.50 in EN 1998-1-1:2024.",
            )
        )
        Max_scalling = parse_float_required(
            st.text_input(
                "Max scaling",
                value="2.00",
                key="1D_Max_scalling",
                help="Upper bound of the scaling factor of each signal, recommended as 2.00 in EN 1998-1-1:2024.",
            )
        )

        Min_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean lower limit",
                value="0.75",
                key="1D_Min_Target_MEAN",
                help="Lower bound for the mean spectrum (relative to target). A value of 0.75 is recomended in EN 1998-1-1:2024.",
            )
        )
        Max_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean upper limit",
                value="1.30",
                key="1D_Max_Target_MEAN",
                help="Upper bound for the mean spectrum (relative to target). A value of 1.30 is recomended in EN 1998-1-1:2024.",
            )
        )

        Min_Target_ind = parse_float_required(
            st.text_input(
                "Individual lower limit",
                value="0.50",
                key="1D_Min_Target_ind",
                help="Lower bound for each individual spectrum (relative to target), recommended as 0.50 in EN 1998-1-1:2024 (i.e. ±50 % of the target spectrum).",
            )
        )
        Max_Target_ind = parse_float_required(
            st.text_input(
                "Individual upper limit",
                value="2.00",
                key="1D_Max_Target_ind",
                help="Upper bound for each individual spectrum (relative to target), recommended as 2.00 in EN 1998-1-1:2024 (i.e. ±50 % of the target spectrum).",
            )
        )

        st.markdown("---")
        st.markdown("## Seismological filtering conditions")
        st.caption("Optional fields")

        Style_of_Faulting = mechanism_checkboxes("1D2024")

        Mw_lower_bound = parse_float_or_blank(
            st.text_input(
                label="Mw lower bound",
                value="",
                key="1D_Mw_lower",
                help="Moment magnitude lower bound. Leave blank to ignore.",
            )
        )
        Mw_upper_bound = parse_float_or_blank(
            st.text_input(
                label="Mw upper bound",
                value="",
                key="1D_Mw_upper",
                help="Moment magnitude upper bound. Leave blank to ignore.",
            )
        )

        Rjb_lower_bound = parse_float_or_blank(
            st.text_input(
                label="R lower bound [km]",
                value="",
                key="1D_Rjb_lower",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )
        Rjb_upper_bound = parse_float_or_blank(
            st.text_input(
                label="R upper bound [km]",
                value="",
                key="1D_Rjb_upper",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )

        depth_lower_bound = parse_float_or_blank(
            st.text_input("Depth lower bound [km]", value="", key="1D_depth_lower")
        )
        depth_upper_bound = parse_float_or_blank(
            st.text_input("Depth upper bound [km]", value="", key="1D_depth_upper")
        )

        st.markdown("---")
        metadata_path = st.text_input(
            "metadata.csv path", value=str(default_metadata), key="1D_metadata_path"
        )

    # Main buttons
    c_run, c_back = st.columns([2, 1])
    with c_run:
        run = st.button("Run selection", type="primary", key="run_1d_2024")
    with c_back:
        if st.button("⬅️ Back", key="back_1d_2024"):
            clear_cached_results("spectral_1d_2024")
            go("spectral_1d_menu")
            st.stop()

    if run:
        try:
            with st.spinner("Running Annex D (1D, EC8:2024)..."):
                out_df, fig, result, comp, x = AnnexD_1D_runner.run_annexD_1d_ec82024(
                    metadata_path=metadata_path,
                    N_records=int(N_records),
                    T1=float(T1),
                    T2=float(T2),
                    Soil_Type_filtering=str(Soil_Type_filtering),
                    S_alpha_RP=float(S_alpha_RP),
                    S_beta_RP=float(S_beta_RP),
                    Soil_Type_spectrum=str(Soil_Type_spectrum),
                    Min_scalling=float(Min_scalling),
                    Max_scalling=float(Max_scalling),
                    Min_Target_MEAN=float(Min_Target_MEAN),
                    Max_Target_MEAN=float(Max_Target_MEAN),
                    Min_Target_ind=float(Min_Target_ind),
                    Max_Target_ind=float(Max_Target_ind),
                    Style_of_Faulting=Style_of_Faulting,
                    Mw_lower_bound=Mw_lower_bound,
                    Mw_upper_bound=Mw_upper_bound,
                    Rjb_lower_bound=Rjb_lower_bound,
                    Rjb_upper_bound=Rjb_upper_bound,
                    depth_lower_bound=depth_lower_bound,
                    depth_upper_bound=depth_upper_bound,
                )

            st.success("Selection completed.")

            if fig is not None:
                st.pyplot(fig, clear_figure=False)

            st.dataframe(out_df, use_container_width=True)

            score, rmse_ratio = compute_matching_score_from_fig(fig, float(T1), float(T2))
            score_txt = f"{score:.1f}%" if score is not None else "n/a"

            caption_txt = (
                f"Optimizer success: {getattr(result, 'success', None)} | "
                f"iterations: {getattr(result, 'nit', None)} | "
                f"matching score (T1–T2): {score_txt}"
            )
            # One-click deliverable: ZIP = metadata_filtered_SEE.csv + processed (OpenSees-ready) records
            csv_text = out_df.to_csv(index=False)
            event_ids = tuple(out_df["Event_ID"].astype(str).to_numpy())
            station_ids = tuple(out_df["Station_ID"].astype(str).to_numpy())
            comp_t = tuple(np.asarray(comp, dtype=int).tolist())
            x_t = tuple(np.asarray(x, dtype=float).tolist())

            with st.spinner("Preparing download package (CSV + records)..."):
                zip_bytes = build_1d_zip_cached(event_ids, station_ids, comp_t, x_t, csv_text)

            store_cached_results(
                route_name="spectral_1d_2024",
                out_df=out_df,
                fig=fig,
                caption=caption_txt,
                file_name="PEUSP_EC8_2024_1D_AnnexD_selection.csv",
                zip_bytes=zip_bytes,
                zip_file_name="PEUSP_EC8_2024_1D_AnnexD_selection.zip",
                zip_mime="application/zip",
            )

            st.download_button(
                "Download results",
                data=zip_bytes,
                file_name="PEUSP_EC8_2024_1D_AnnexD_selection.zip",
                mime="application/zip",
                key="dl_1d_2024_zip",
            )

            st.caption(
                f"Optimizer success: {getattr(result, 'success', None)} | "
                f"iterations: {getattr(result, 'nit', None)} | "
                f"matching score (T1–T2): {score_txt}"
            )

        except Exception as e:
            st.error(friendly_filtering_error(e))

    else:
        render_cached_results("spectral_1d_2024", "PEUSP_EC8_2024_1D_AnnexD_selection.csv")

elif st.session_state.route == "spectral_2d_2004":
    st.subheader("Spectral matching → 2D → EN 1998-1-1:2004")
    st.caption("EC8:2004 — record selection & scaling (geometric mean of horizontal components)")

    app_dir = Path(__file__).parent
    default_metadata = app_dir / "metadata.csv"

    with st.sidebar:
        st.markdown("## Records & matching conditions")
        st.caption("Use '.' for decimals (',' also accepted).")

        N_records = st.number_input(
            "Number of records",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            help="When only three input motions are used, the structural demand is determined from the most unfavourable value produced from the corresponding three dynamic analyses.",
        )

        T1 = parse_float_or_blank(
            st.text_input(
                label="T1 [s]",
                value="",
                key="2D2004_T1",
                help="Lower bound of the matching range, recommended in EN 1998-1-1:2004 as T2 = 0.2T, with T as the fundamental period of the structure.",
            )
        )
        T2 = parse_float_or_blank(
            st.text_input(
                label="T2 [s]",
                value="",
                key="2D2004_T2",
                help="Upper bound of the matching range, recommended in EN 1998-1-1:2004 as T2 = 2T, with T as the fundamental period of the structure.",
            )
        )

        Soil_Type_filtering = st.selectbox(
            "Soil type filtering",
            ["A", "B", "C", "D", "ALL"],
            index=4,
            key="2D2004_Soil_Type_filtering",
            help="Used to filter the database by Vs30 bounds.",
        )

        # ---- EC8:2004 target spectrum parameters
        agr = parse_float_or_blank(
            st.text_input(
                "agR [g]",
                value="",
                key="2D2004_agr",
                help="Reference peak ground acceleration on type A ground (in g).",
            )
        )
        Ifactor = parse_float_required(
            st.text_input(
                "γI",
                value="1.00",
                key="2D2004_Ifactor",
                help="Importance factor.",
            )
        )
        Type = st.selectbox(
            "Spectrum type",
            ["Type1", "Type2"],
            index=0,
            key="2D2004_Type",
            help="Type1 for Mw > 5.5, Type2 for Mw < 5.5.",
        )
        Soil_Type_spectrum = st.selectbox(
            "Soil type (target spectrum)",
            ["A", "B", "C", "D", "E"],
            index=0,
            key="2D2004_Soil_Type_spectrum",
            help="Used to compute the EN 1998-1-1:2004 target response spectrum.",
        )

        Min_scalling = parse_float_or_blank(
            st.text_input(
                "Min scaling",
                value="0.50",
                key="2D2004_Min_scalling",
                help="No recommendations are provided in EN 1998-1-1:2004. A value of 0.50 is recomended in EN 1998-1-1:2024 for the lower bound of the scaling factor of each signal.",
            )
        )
        Max_scalling = parse_float_required(
            st.text_input(
                "Max scaling",
                value="2.00",
                key="2D2004_Max_scalling",
                help="No recommendations are provided in EN 1998-1-1:2004. A value of 2.00 is recomended in EN 1998-1-1:2024 for the upper bound of the scaling factor of each signal.",
            )
        )

        Min_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean lower limit",
                value="0.90",
                key="2D2004_Min_Target_MEAN",
                help="Lower bound for the mean spectrum (relative to target). A value of 0.90 is recomended in EN 1998-1-1:2004.",
            )
        )
        Max_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean upper limit",
                value="1.20",
                key="2D2004_Max_Target_MEAN",
                help="Upper bound for the mean spectrum (relative to target). A value of 1.20 is recomended in EN 1998-1-1:2004.",
            )
        )

        Min_Target_ind = parse_float_required(
            st.text_input(
                "Individual lower limit",
                value="0.50",
                key="2D2004_Min_Target_ind",
                help="Lower bound for each individual spectrum (relative to target). No recommendations are provided in EN 1998-1-1:2004. A value of 0.50 is recomended in EN 1998-1-1:2024 (i.e. ±50 % of the target spectrum).",
            )
        )
        Max_Target_ind = parse_float_required(
            st.text_input(
                "Individual upper limit",
                value="2.00",
                key="2D2004_Max_Target_ind",
                help="Upper bound for each individual spectrum (relative to target). No recommendations are provided in EN 1998-1-1:2004. A value of 2.00 is recomended in EN 1998-1-1:2024 (i.e. ±50 % of the target spectrum).",
            )
        )
        st.markdown("---")
        st.markdown("## Seismological filtering conditions")
        st.caption("Optional fields")

        Style_of_Faulting = mechanism_checkboxes("2D2004")

        Mw_lower_bound = parse_float_or_blank(
            st.text_input(
                label="Mw lower bound",
                value="",
                key="Mw_lower",
                help="Moment magnitude lower bound. Leave blank to ignore.",
            )
        )
        Mw_upper_bound = parse_float_or_blank(
            st.text_input(
                label="Mw upper bound",
                value="",
                key="Mw_upper",
                help="Moment magnitude upper bound. Leave blank to ignore.",
            )
        )

        Rjb_lower_bound = parse_float_or_blank(
            st.text_input(
                label="R lower bound [km]",
                value="",
                key="Rjb_lower",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )
        Rjb_upper_bound = parse_float_or_blank(
            st.text_input(
                label="R upper bound [km]",
                value="",
                key="Rjb_upper",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )

        depth_lower_bound = parse_float_or_blank(st.text_input("Depth lower bound [km]", value=""))
        depth_upper_bound = parse_float_or_blank(st.text_input("Depth upper bound [km]", value=""))

        st.markdown("---")
        metadata_path = st.text_input(
            "metadata.csv path",
            value=str(default_metadata),
            key="2D2004_metadata_path",
        )

    c_run, c_back = st.columns([2, 1])
    with c_run:
        run = st.button("Run selection", type="primary", key="run_2d_2004")
    with c_back:
        if st.button("⬅️ Back", key="back_2d_2004"):
            clear_cached_results("spectral_2d_2004")
            go("spectral_2d_menu")
            st.stop()

    if run:
        try:
            with st.spinner("Running EC8:2004 (2D)..."):
                out_df, fig, result = EC82004_2D_runner.run_EC82024_2d_ec82004(
                    metadata_path=metadata_path,
                    N_records=int(N_records),
                    T1=T1,
                    T2=T2,
                    Soil_Type_filtering=Soil_Type_filtering,
                    agr=agr,
                    Ifactor=Ifactor,
                    Type=Type,
                    Soil_Type_spectrum=Soil_Type_spectrum,
                    Min_scalling=Min_scalling,
                    Max_scalling=Max_scalling,
                    Min_Target_MEAN=Min_Target_MEAN,
                    Max_Target_MEAN=Max_Target_MEAN,
                    Min_Target_ind=Min_Target_ind,
                    Max_Target_ind=Max_Target_ind,
                    Style_of_Faulting=Style_of_Faulting,
                    Mw_lower_bound=Mw_lower_bound,
                    Mw_upper_bound=Mw_upper_bound,
                    Rjb_lower_bound=Rjb_lower_bound,
                    Rjb_upper_bound=Rjb_upper_bound,
                    depth_lower_bound=depth_lower_bound,
                    depth_upper_bound=depth_upper_bound,
                )

            st.success("Selection completed.")

            if fig is not None:
                st.pyplot(fig, clear_figure=False)

            st.dataframe(out_df, use_container_width=True)

            score, rmse_ratio = compute_matching_score_from_fig(fig, float(T1), float(T2))
            score_txt = f"{score:.1f}%" if score is not None else "n/a"

            caption_txt = (
                f"Optimizer success: {getattr(result, 'success', None)} | "
                f"iterations: {getattr(result, 'nit', None)} | "
                f"matching score (T1–T2): {score_txt}"
            )

            store_cached_results(
                route_name="spectral_2d_2004",
                out_df=out_df,
                fig=fig,
                caption=caption_txt,
                file_name="PEUSP_EC8_2004_2D_selection.csv",
            )

            # One-click deliverable: ZIP = metadata_filtered_SEE.csv + processed (OpenSees-ready) records
            csv_text = out_df.to_csv(index=False)
            event_ids = tuple(out_df["Event_ID"].astype(str).to_numpy())
            station_ids = tuple(out_df["Station_ID"].astype(str).to_numpy())
            x_t = tuple(out_df["scale_factor"].astype(float).to_numpy())

            with st.spinner("Preparing download package (CSV + records)..."):
                zip_bytes = build_2d_zip_cached(event_ids, station_ids, x_t, csv_text)

            store_cached_results(
                route_name="spectral_2d_2004",
                out_df=out_df,
                fig=fig,
                caption=caption_txt,
                file_name="PEUSP_EC8_2004_2D_selection.csv",
                zip_bytes=zip_bytes,
                zip_file_name="PEUSP_EC8_2004_2D_selection.zip",
                zip_mime="application/zip",
            )

            st.download_button(
                "Download results",
                data=zip_bytes,
                file_name="PEUSP_EC8_2004_2D_selection.zip",
                mime="application/zip",
                key="dl_2d_2004_zip",
            )

            st.caption(
                f"Optimizer success: {getattr(result, 'success', None)} | "
                f"iterations: {getattr(result, 'nit', None)} | "
                f"matching score (T1–T2): {score_txt}"
            )

        except Exception as e:
            st.error(friendly_filtering_error(e))

    else:
        render_cached_results("spectral_2d_2004", "PEUSP_EC8_2004_2D_selection.csv")


# ============================================================
# IMPLEMENTED: Spectral matching → 2D → EN 1998-1-1:2024
# (UNCHANGED)
# ============================================================
elif st.session_state.route == "spectral_2d_2024":
    st.subheader("Spectral matching → 2D → EN 1998-1-1:2024")
    st.caption("EC8:2024 Annex D — record selection & scaling (geometric mean of horizontal components)")

    app_dir = Path(__file__).parent
    default_metadata = app_dir / "metadata.csv"

    with st.sidebar:
        st.markdown("## Records & matching conditions")
        st.caption("Use '.' for decimals (',' also accepted).")

        N_records = st.number_input(
            "Number of records",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            help="The minimum number of input motions may be reduced to three, considering the maximum response, only for low and very low seismic action classes.",
        )

        T1 = parse_float_or_blank(
            st.text_input(
                label="T1 [s]",
                value="",
                help="Lower bound of the matching range, recomended in EN 1998-1-1:2024 as T1 = 0.2T, with T as the fundamental period of the structure.",
            )
        )
        T2 = parse_float_or_blank(
            st.text_input(
                label="T2 [s]",
                value="",
                help="Upper bound of the matching range, recomended in EN 1998-1-1:2024 as T2 = 1.5T, with T as the fundamental period of the structure.",
            )
        )

        Soil_Type_filtering = st.selectbox(
            "Soil type filtering",
            ["A", "B", "C", "D", "ALL"],
            index=4,
            help="Used to filter the database by Vs30 bounds."
        )

        S_alpha_RP = parse_float_or_blank(
            st.text_input(
                label="Sα,RP [g]",
                value="",
                key="S_alpha_RP",
                help="Maximum response spectral acceleration, for 5% damping, at the constant acceleration range of the elastic response spectrum. Explore the latest European Seismic Hazard Model 2020 (ESHM20) at hazard.efehr.org",
            )
        )

        S_beta_RP = parse_float_or_blank(
            st.text_input(
                label="Sᵦ,RP [g]",
                value="",
                key="S_beta_RP",
                help="5%-damped response spectral acceleration at the vibration period Tβ = 1 s. Explore the latest European Seismic Hazard Model 2020 (ESHM20) at hazard.efehr.org",
            )
        )
        
        Soil_Type_spectrum = st.selectbox(
            "Soil type (target spectrum)",
            ["A", "B", "C", "D"],
            index=0,
            help="Used to compute the EN 1998-1-1:2024 target response spectrum."
        )
        
        Min_scalling = parse_float_required(
            st.text_input(
                "Min scaling",
                value="0.50",
                help="Lower bound of the scalling factor of each signal. A value of 0.50 is recommended in EN 1998-1-1:2024.",
            )
        )
        Max_scalling = parse_float_required(
            st.text_input(
                "Max scaling",
                value="2.00",
                help="Upper bound of the scalling factor of each signal. A value of 2.00 is recommended in EN 1998-1-1:2024",
            )
        )

        Min_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean lower limit",
                value="0.75",
                help="Lower bound for the mean spectrum (relative to target). A value of 0.75 is recomended in EN 1998-1-1:2024.",
            )
        )
        Max_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean upper limit",
                value="1.30",
                help="Upper bound for the mean spectrum (relative to target). A value of 1.30 is recomended in EN 1998-1-1:2024.",
            )
        )

        Min_Target_ind = parse_float_required(
            st.text_input(
                "Individual lower limit",
                value="0.50",
                help="Lower bound for each individual spectrum (relative to target), recommended as 0.50 in EN 1998-1-1:2024 (i.e. ±50 % of the target spectrum).",
            )
        )
        Max_Target_ind = parse_float_required(
            st.text_input(
                "Individual upper limit",
                value="2.00",
                help="Upper bound for each individual spectrum (relative to target), recommended as 2.00 in EN 1998-1-1:2024 (i.e. ±50 % of the target spectrum).",
            )
        )

        st.markdown("---")
        st.markdown("## Seismological filtering conditions")
        st.caption("Optional fields")

        Style_of_Faulting = mechanism_checkboxes("2D2024")

        Mw_lower_bound = parse_float_or_blank(
            st.text_input(
                label="Mw lower bound",
                value="",
                key="Mw_lower",
                help="Moment magnitude lower bound. Leave blank to ignore.",
            )
        )
        Mw_upper_bound = parse_float_or_blank(
            st.text_input(
                label="Mw upper bound",
                value="",
                key="Mw_upper",
                help="Moment magnitude upper bound. Leave blank to ignore.",
            )
        )

        Rjb_lower_bound = parse_float_or_blank(
            st.text_input(
                label="R lower bound [km]",
                value="",
                key="Rjb_lower",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )
        Rjb_upper_bound = parse_float_or_blank(
            st.text_input(
                label="R upper bound [km]",
                value="",
                key="Rjb_upper",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )

        depth_lower_bound = parse_float_or_blank(st.text_input("Depth lower bound [km]", value=""))
        depth_upper_bound = parse_float_or_blank(st.text_input("Depth upper bound [km]", value=""))

        st.markdown("---")
        metadata_path = st.text_input("metadata.csv path", value=str(default_metadata))

    # Main buttons
    c_run, c_back = st.columns([2, 1])
    with c_run:
        run = st.button("Run selection", type="primary")
    with c_back:
        if st.button("⬅️ Back", key="back_2d_2024"):
            clear_cached_results("spectral_2d_2024")
            go("spectral_2d_menu")
            st.stop()

    if run:
        try:
            with st.spinner("Running Annex D (2D, EC8:2024)..."):
                out_df, fig, result = AnnexD_2D_runner.run_annexD_2d_ec82024(
                    metadata_path=metadata_path,
                    N_records=int(N_records),
                    T1=float(T1),
                    T2=float(T2),
                    Soil_Type_filtering=str(Soil_Type_filtering),
                    S_alpha_RP=float(S_alpha_RP),
                    S_beta_RP=float(S_beta_RP),
                    Soil_Type_spectrum=str(Soil_Type_spectrum),
                    Min_scalling=float(Min_scalling),
                    Max_scalling=float(Max_scalling),
                    Min_Target_MEAN=float(Min_Target_MEAN),
                    Max_Target_MEAN=float(Max_Target_MEAN),
                    Min_Target_ind=float(Min_Target_ind),
                    Max_Target_ind=float(Max_Target_ind),
                    Style_of_Faulting=Style_of_Faulting,
                    Mw_lower_bound=Mw_lower_bound,
                    Mw_upper_bound=Mw_upper_bound,
                    Rjb_lower_bound=Rjb_lower_bound,
                    Rjb_upper_bound=Rjb_upper_bound,
                    depth_lower_bound=depth_lower_bound,
                    depth_upper_bound=depth_upper_bound,
                )

            st.success("Selection completed.")

            if fig is not None:
                st.pyplot(fig, clear_figure=False)

            st.dataframe(out_df, use_container_width=True)

            score, rmse_ratio = compute_matching_score_from_fig(fig, float(T1), float(T2))
            score_txt = f"{score:.1f}%" if score is not None else "n/a"

            caption_txt = (
                f"Optimizer success: {getattr(result, 'success', None)} | "
                f"iterations: {getattr(result, 'nit', None)} | "
                f"matching score (T1–T2): {score_txt}"
            )
            store_cached_results(
                route_name="spectral_2d_2024",
                out_df=out_df,
                fig=fig,
                caption=caption_txt,
                file_name="PEUSP_EC8_2024_2D_AnnexD_selection.csv",
            )

            # One-click deliverable: ZIP = metadata_filtered_SEE.csv + processed (OpenSees-ready) records
            csv_text = out_df.to_csv(index=False)
            event_ids = tuple(out_df["Event_ID"].astype(str).to_numpy())
            station_ids = tuple(out_df["Station_ID"].astype(str).to_numpy())
            x_t = tuple(out_df["scale_factor"].astype(float).to_numpy())

            with st.spinner("Preparing download package (CSV + records)..."):
                zip_bytes = build_2d_zip_cached(event_ids, station_ids, x_t, csv_text)

            store_cached_results(
                route_name="spectral_2d_2024",
                out_df=out_df,
                fig=fig,
                caption=caption_txt,
                file_name="PEUSP_EC8_2024_2D_AnnexD_selection.csv",
                zip_bytes=zip_bytes,
                zip_file_name="PEUSP_EC8_2024_2D_AnnexD_selection.zip",
                zip_mime="application/zip",
            )
            st.download_button(
                "Download results",
                data=zip_bytes,
                file_name="PEUSP_EC8_2024_2D_AnnexD_selection.zip",
                mime="application/zip",
                key="dl_2d_2024_zip",
            )

            st.caption(
                f"Optimizer success: {getattr(result, 'success', None)} | "
                f"iterations: {getattr(result, 'nit', None)} | "
                f"matching score (T1–T2): {score_txt}"
            )

        except Exception as e:
            st.error(friendly_filtering_error(e))

    else:
        render_cached_results("spectral_2d_2024", "PEUSP_EC8_2024_2D_AnnexD_selection.csv")


# -----------------------------
# 2D MENU (User-defined target) — placeholder
# -----------------------------
elif st.session_state.route == "spectral_2d_user":
    st.subheader("Spectral matching → 2D → User-defined target")
    st.caption(
        "User-defined target and conditions — record selection & scaling (geometric mean of horizontal components)"
    )

    app_dir = Path(__file__).parent
    default_metadata = app_dir / "metadata.csv"

    # Fixed internal period grid (must match user_2D_runner)
    T_grid = np.arange(start=0, stop=4.05, step=0.05)

    with st.sidebar:
        st.markdown("## Records & matching conditions")
        st.caption("Use '.' for decimals (',' also accepted). Target file can be space- or tab-delimited.")

        N_records = st.number_input(
            "Number of records",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            key="USER2D_N_records",
            help="Number of ground motions to select; preferable ≥ 7.",
        )

        T1 = parse_float_or_blank(
            st.text_input(
                label="T1 [s]",
                value="",
                key="USER2D_T1",
                help="Lower bound of the matching range. This value should be structure-dependent, e.g., 0.00 for masonry (i.e., full domain).",
            )
        )
        T2 = parse_float_or_blank(
            st.text_input(
                label="T2 [s]",
                value="",
                key="USER2D_T2",
                help="Upper bound of the matching range. This value should be structure-dependent, e.g., 4.00 for masonry (i.e., full domain).",
            )
        )

        Soil_Type_filtering = st.selectbox(
            "Soil type filtering",
            ["A", "B", "C", "D", "ALL"],
            index=4,
            key="USER2D_Soil_Type_filtering",
            help="Used to filter the database by Vs30 bounds.",
        )

        st.markdown("---")
        st.markdown("## User-defined target spectrum")
        target_file = st.file_uploader(
            "Upload target spectrum (.txt)",
            type=["txt", "csv", "dat"],
            key="USER2D_target_upload",
            help="Two columns: T [s] and Sa [g]; preferable ΔT = 0.05 s but any spacing is allowed.",
        )

        Min_scalling = parse_float_required(
            st.text_input(
                "Min scaling",
                value="0.50",
                key="USER2D_Min_scalling",
                help="Lower bound of the scaling factor for each record. A value of 0.50 is recommended.",
            )
        )
        Max_scalling = parse_float_required(
            st.text_input(
                "Max scaling",
                value="2.00",
                key="USER2D_Max_scalling",
                help="Upper bound of the scaling factor for each record. A value of 2.00 is recommended.",
            )
        )

        Min_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean lower limit",
                value="0.80",
                key="USER2D_Min_Target_MEAN",
                help="Lower bound for the mean spectrum (relative to target). Values in the range 0.75–0.90 are recommended.",
            )
        )
        Max_Target_MEAN = parse_float_required(
            st.text_input(
                "Mean upper limit",
                value="1.20",
                key="USER2D_Max_Target_MEAN",
                help="Upper bound for the mean spectrum (relative to target). Values in the range 1.10–1.30 are recommended.",
            )
        )

        Min_Target_ind = parse_float_required(
            st.text_input(
                "Individual lower limit",
                value="0.50",
                key="USER2D_Min_Target_ind",
                help="Lower bound for each individual spectrum (relative to target). A value of 0.50 is recommended (i.e. ±50 % of the target spectrum).",
            )
        )
        Max_Target_ind = parse_float_required(
            st.text_input(
                "Individual upper limit",
                value="2.00",
                key="USER2D_Max_Target_ind",
                help="Upper bound for each individual spectrum (relative to target). A value of 2.00 is recommended (i.e. ±50 % of the target spectrum).",
            )
        )

        st.markdown("---")
        st.markdown("## Seismological filtering conditions")
        st.caption("Optional fields")

        Style_of_Faulting = mechanism_checkboxes("USER2D")

        Mw_lower_bound = parse_float_or_blank(
            st.text_input(
                label="Mw lower bound",
                value="",
                key="USER2D_Mw_lower",
                help="Moment magnitude lower bound. Leave blank to ignore.",
            )
        )
        Mw_upper_bound = parse_float_or_blank(
            st.text_input(
                label="Mw upper bound",
                value="",
                key="USER2D_Mw_upper",
                help="Moment magnitude upper bound. Leave blank to ignore.",
            )
        )

        Rjb_lower_bound = parse_float_or_blank(
            st.text_input(
                label="R lower bound [km]",
                value="",
                key="USER2D_Rjb_lower",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )
        Rjb_upper_bound = parse_float_or_blank(
            st.text_input(
                label="R upper bound [km]",
                value="",
                key="USER2D_Rjb_upper",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )

        depth_lower_bound = parse_float_or_blank(
            st.text_input("Depth lower bound [km]", value="", key="USER2D_depth_lower")
        )
        depth_upper_bound = parse_float_or_blank(
            st.text_input("Depth upper bound [km]", value="", key="USER2D_depth_upper")
        )

        st.markdown("---")
        metadata_path = st.text_input(
            "metadata.csv path", value=str(default_metadata), key="USER2D_metadata_path"
        )

        # seed = st.number_input(
        #     "Random seed",
        #     min_value=0,
        #     max_value=10_000,
        #     value=1,
        #     step=1,
        #     key="USER2D_seed",
        #     help="Seed for reproducibility.",
        # )

    # ---- Read + interpolate user spectrum
    Target_user = None
    if target_file is not None:
        try:
            raw = target_file.getvalue().decode("utf-8", errors="ignore")
            arr = np.genfromtxt(io.StringIO(raw), comments="#", delimiter=None)

            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError("Target file must contain at least two columns: T and Sa.")

            T_in = arr[:, 0].astype(float)
            Sa_in = arr[:, 1].astype(float)

            # Drop NaNs
            mask = np.isfinite(T_in) & np.isfinite(Sa_in)
            T_in, Sa_in = T_in[mask], Sa_in[mask]

            if T_in.size < 2:
                raise ValueError("Target file must contain at least two valid rows.")

            # Sort by period and remove duplicates (keep first)
            order = np.argsort(T_in)
            T_in, Sa_in = T_in[order], Sa_in[order]
            _, uniq_idx = np.unique(T_in, return_index=True)
            T_in, Sa_in = T_in[uniq_idx], Sa_in[uniq_idx]

            if T_in.min() > 0.0 + 1e-9 or T_in.max() < 4.0 - 1e-9:
                raise ValueError("Target T range must cover at least 0.0 to 4.0 seconds.")

            Sa_interp = np.interp(T_grid, T_in, Sa_in).astype(float)
            Target_user = Sa_interp

            st.success("Target spectrum loaded and interpolated to ΔT = 0.05 s (0–4 s).")
            with st.expander("Preview target spectrum"):
                st.write("Interpolated target (first 10 points):", Sa_interp[:10])
        except Exception as e:
            st.error(f"Could not read target file: {e}")
            Target_user = None

    # Main buttons
    c_run, c_back = st.columns([2, 1])
    with c_run:
        run = st.button("Run selection", type="primary", key="run_2d_user")
    with c_back:
        if st.button("⬅️ Back", key="back_2d_user"):
            clear_cached_results("spectral_2d_user")
            go("spectral_2d_menu")
            st.stop()

    if run:
        if Target_user is None:
            st.error("Please upload a valid target spectrum file before running.")
        else:
            try:
                with st.spinner("Running User-defined target (2D)..."):
                    out_df, fig, result = user_2D_runner.run_user_2d(
                        metadata_path=metadata_path,
                        N_records=int(N_records),
                        T1=float(T1),
                        T2=float(T2),
                        Soil_Type_filtering=str(Soil_Type_filtering),
                        Target_user=Target_user,
                        Min_scalling=float(Min_scalling),
                        Max_scalling=float(Max_scalling),
                        Min_Target_MEAN=float(Min_Target_MEAN),
                        Max_Target_MEAN=float(Max_Target_MEAN),
                        Min_Target_ind=float(Min_Target_ind),
                        Max_Target_ind=float(Max_Target_ind),
                        Style_of_Faulting=Style_of_Faulting,
                        Mw_lower_bound=Mw_lower_bound,
                        Mw_upper_bound=Mw_upper_bound,
                        Rjb_lower_bound=Rjb_lower_bound,
                        Rjb_upper_bound=Rjb_upper_bound,
                        depth_lower_bound=depth_lower_bound,
                        depth_upper_bound=depth_upper_bound,
                        # seed=int(seed),
                    )

                st.success("Selection completed.")

                if fig is not None:
                    st.pyplot(fig, clear_figure=False)

                st.dataframe(out_df, use_container_width=True)

                score, rmse_ratio = compute_matching_score_from_fig(fig, float(T1), float(T2))
                score_txt = f"{score:.1f}%" if score is not None else "n/a"

                caption_txt = (
                    f"Optimizer success: {getattr(result, 'success', None)} | "
                    f"iterations: {getattr(result, 'nit', None)} | "
                    f"matching score (T1–T2): {score_txt}"
                )
                store_cached_results(
                    route_name="spectral_2d_user",
                    out_df=out_df,
                    fig=fig,
                    caption=caption_txt,
                    file_name="PEUSP_UserTarget_2D_selection.csv",
                )

                # One-click deliverable: ZIP = metadata_filtered_SEE.csv + processed (OpenSees-ready) records
                csv_text = out_df.to_csv(index=False)
                event_ids = tuple(out_df["Event_ID"].astype(str).to_numpy())
                station_ids = tuple(out_df["Station_ID"].astype(str).to_numpy())
                x_t = tuple(out_df["scale_factor"].astype(float).to_numpy())

                with st.spinner("Preparing download package (CSV + records)..."):
                    zip_bytes = build_2d_zip_cached(event_ids, station_ids, x_t, csv_text)

                store_cached_results(
                    route_name="spectral_2d_user",
                    out_df=out_df,
                    fig=fig,
                    caption=caption_txt,
                    file_name="PEUSP_UserTarget_2D_selection.csv",
                    zip_bytes=zip_bytes,
                    zip_file_name="PEUSP_UserTarget_2D_selection.zip",
                    zip_mime="application/zip",
                )

                st.download_button(
                    "Download results",
                    data=zip_bytes,
                    file_name="PEUSP_UserTarget_2D_selection.zip",
                    mime="application/zip",
                    key="dl_2d_user_zip",
                )

                st.caption(
                    f"Optimizer success: {getattr(result, 'success', None)} | "
                    f"iterations: {getattr(result, 'nit', None)} | "
                    f"objective: {getattr(result, 'fun', None)}"
                )

            except Exception as e:
                st.error(friendly_filtering_error(e))

    else:
        render_cached_results("spectral_2d_user", "PEUSP_UserTarget_2D_selection.csv")

# -----------------------------
# Scenario placeholder
# -----------------------------
elif st.session_state.route == "scenario":
    st.subheader("Scenario-based selection")
    st.caption("Scenario (Mw, R) selection with optional filtering (soil type, style of faulting, depth).")

    app_dir = Path(__file__).parent
    default_metadata = app_dir / "metadata.csv"

    with st.sidebar:
        st.markdown("## Number of records")
        st.caption("Use '.' for decimals (',' also accepted).")

        N_records = st.number_input(
            "Number of records",
            min_value=1,
            max_value=100,
            value=21,
            step=1,
            key="SCEN_N_records",
            help="Number of ground motions to select (up to 100).",
        )

        st.markdown("---")
        st.markdown("## Basic scenario definition (Mw, R)")
        Mw_lower_bound = parse_float_or_blank(
            st.text_input(
                label="Mw lower bound",
                value="",
                key="SCEN_Mw_lower",
                help="Moment magnitude lower bound. Leave blank to ignore.",
            )
        )
        Mw_upper_bound = parse_float_or_blank(
            st.text_input(
                label="Mw upper bound",
                value="",
                key="SCEN_Mw_upper",
                help="Moment magnitude upper bound. Leave blank to ignore.",
            )
        )

        Rjb_lower_bound = parse_float_or_blank(
            st.text_input(
                label="R lower bound [km]",
                value="",
                key="SCEN_Rjb_lower",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )
        Rjb_upper_bound = parse_float_or_blank(
            st.text_input(
                label="R upper bound [km]",
                value="",
                key="SCEN_Rjb_upper",
                help="Preferable Joyner–Boore distance. Leave blank to ignore.",
            )
        )

        st.markdown("---")
        st.markdown("## Additional filtering")

        Style_of_Faulting = mechanism_checkboxes("SCEN")
        
        Soil_Type_filtering = st.selectbox(
            "Soil type filtering",
            ["A", "B", "C", "D", "ALL"],
            index=4,
            key="SCEN_Soil_Type_filtering",
            help="Used to filter the database by Vs30 bounds.",
        )


        depth_lower_bound = parse_float_or_blank(
            st.text_input(
                "Depth lower bound [km]",
                value="",
                key="SCEN_depth_lower",
                help="Leave blank to ignore.",
            )
        )
        depth_upper_bound = parse_float_or_blank(
            st.text_input(
                "Depth upper bound [km]",
                value="",
                key="SCEN_depth_upper",
                help="Leave blank to ignore.",
            )
        )

        st.markdown("---")
        metadata_path = st.text_input(
            "metadata.csv path",
            value=str(default_metadata),
            key="SCEN_metadata_path",
        )

    c_run, c_back = st.columns([2, 1])
    with c_run:
        run = st.button("Run selection", type="primary", key="run_scenario")
    with c_back:
        if st.button("⬅️ Back", key="back_scenario"):
            clear_cached_results("scenario")
            go("home")
            st.stop()

    if run:
        try:
            with st.spinner("Running scenario-based selection..."):
                out_df, fig = Scenario_based_runner.run_scenario_based(
                    metadata_path=metadata_path,
                    N_records=int(N_records),
                    Mw_lower_bound=Mw_lower_bound,
                    Mw_upper_bound=Mw_upper_bound,
                    Rjb_lower_bound=Rjb_lower_bound,
                    Rjb_upper_bound=Rjb_upper_bound,
                    Soil_Type_filtering=str(Soil_Type_filtering),
                    Style_of_Faulting=Style_of_Faulting,
                    depth_lower_bound=depth_lower_bound,
                    depth_upper_bound=depth_upper_bound,
                )

            st.success("Selection completed.")

            if fig is not None:
                st.pyplot(fig, clear_figure=False)

            st.dataframe(out_df, use_container_width=True)

            store_cached_results(
                route_name="scenario",
                out_df=out_df,
                fig=fig,
                caption=None,
                file_name="PEUSP_Scenario_based_selection.csv",
            )

            # One-click deliverable: ZIP = metadata_filtered_SEE.csv + processed (PGA-normalized) records
            csv_text = out_df.to_csv(index=False)
            event_ids = tuple(out_df["Event_ID"].astype(str).to_numpy())
            station_ids = tuple(out_df["Station_ID"].astype(str).to_numpy())

            with st.spinner("Preparing download package (CSV + records)..."):
                zip_bytes = build_scenario_zip_cached(event_ids, station_ids, csv_text)

            store_cached_results(
                route_name="scenario",
                out_df=out_df,
                fig=fig,
                caption=None,
                file_name="PEUSP_Scenario_based_selection.csv",
                zip_bytes=zip_bytes,
                zip_file_name="PEUSP_Scenario_based_selection.zip",
                zip_mime="application/zip",
            )

            st.download_button(
                "Download results",
                data=zip_bytes,
                file_name="PEUSP_Scenario_based_selection.zip",
                mime="application/zip",
                key="dl_scenario_zip",
            )

        except Exception as e:
            st.error(friendly_filtering_error(e))

    else:
        render_cached_results("scenario", "PEUSP_Scenario_based_selection.csv")
