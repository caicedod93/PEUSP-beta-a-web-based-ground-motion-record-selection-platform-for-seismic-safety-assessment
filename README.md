# PEUSP-beta: A web-based ground motion record selection platform for seismic safety assessment

**PEUSP-beta** is a Python-based web application for the selection and scaling of earthquake ground motion records for seismic safety assessment and performance-based earthquake engineering applications.

The platform supports the selection of recorded European strong-motion data through two main workflows: **spectral matching** and **scenario-based selection**. The spectral-matching workflow allows users to select one-dimensional (1D) or two-dimensional (2D) horizontal ground motion components and scale them to target spectra defined according to **EN 1998-1:2004**, **EN 1998-1-1:2024**, or a **user-defined target spectrum**. The scenario-based workflow enables ground motion selection based on seismological and site-related parameters, including moment magnitude, Joyner–Boore distance, VS30-based soil class, faulting mechanism, and focal depth.

PEUSP-beta is implemented in Python using **Streamlit** as a browser-based graphical user interface. The selection and scaling procedures are formulated as optimisation problems, using metaheuristic algorithms to identify suites of ground motions that satisfy the imposed spectral and seismological constraints. The platform is linked to the **European Strong-Motion (ESM) database**, allowing the automatic retrieval and preparation of selected records.

The output of the software includes selected record metadata, scaling factors, processed acceleration time histories, and quick-look plots of acceleration, velocity, displacement, and response spectra. The exported records are intended to facilitate subsequent nonlinear dynamic analyses in structural engineering applications.

---

## Associated publication

This repository contains the source code associated with the SoftwareX publication:

**PEUSP-beta: a web-based ground motion record selection platform for seismic safety assessment**

**Authors:** Daniel Caicedo, Shaghayegh Karimzadeh, Vasco Bernardo, Paulo B. Lourenço  
**Journal:** SoftwareX  
**Year:** 2026  
**Volume:** 35  
**Article number:** 102790  
**DOI:** [10.1016/j.softx.2026.102790](https://doi.org/10.1016/j.softx.2026.102790)

Repository archive:

[https://doi.org/10.5281/zenodo.19697334](https://doi.org/10.5281/zenodo.19697334)

---

## Authors

- **Daniel Caicedo**  
  University of Minho, ISISE, ARISE, Department of Civil Engineering, Guimarães, Portugal

- **Shaghayegh Karimzadeh**  
  University of Minho, ISISE, ARISE, Department of Civil Engineering, Guimarães, Portugal

- **Vasco Bernardo**  
  Earthquake Engineering and Structural Dynamics Unit, Structures Department, National Laboratory for Civil Engineering, Lisbon, Portugal

- **Paulo B. Lourenço**  
  University of Minho, ISISE, ARISE, Department of Civil Engineering, Guimarães, Portugal

---

## Main features

- Web-based graphical user interface developed with Streamlit.
- Ground motion record selection for seismic safety assessment.
- Spectral matching using constant-amplitude scaling.
- Support for 1D and 2D horizontal ground motion selection.
- Code-based target spectra according to:
  - EN 1998-1:2004;
  - EN 1998-1-1:2024.
- User-defined target spectrum option.
- Scenario-based selection using:
  - moment magnitude, Mw;
  - Joyner–Boore distance, Rjb;
  - VS30-based soil class;
  - faulting mechanism;
  - focal depth.
- Optimisation-based record selection and scaling.
- Automatic download and processing of selected records from the European Strong-Motion database.
- Export of selected metadata, processed records, and response spectrum plots.
- Preparation of records for subsequent nonlinear dynamic analyses.

---

## Software workflows

PEUSP-beta currently supports two main workflows:

1. **Spectral matching**
2. **Scenario-based selection**

---

## Spectral-matching workflow

The spectral-matching workflow selects and scales ground motions to match a target response spectrum over a user-defined period range.

The user can select between:

- **1D selection**, based on one horizontal component;
- **2D selection**, based on the geometric mean of the two horizontal components.

The target spectrum can be defined according to:

- **EN 1998-1:2004**;
- **EN 1998-1-1:2024**;
- a **user-defined target spectrum** uploaded by the user.

The user can also define:

- number of records;
- matching period range;
- soil class filtering;
- scaling factor limits;
- admissible lower and upper bounds for the mean spectrum;
- admissible lower and upper bounds for individual spectra;
- moment magnitude range;
- distance range;
- faulting mechanism;
- focal depth range.

The optimisation procedure identifies a suite of records and corresponding scaling factors that minimise the mismatch between the mean spectrum of the selected suite and the target spectrum, while satisfying the imposed constraints.

---

## Scenario-based workflow

The scenario-based workflow selects records according to user-defined seismological and site-related conditions. This workflow is especially useful when the preservation of seismological consistency is important, for example in advanced performance-based earthquake engineering applications.

The scenario-based workflow allows filtering according to:

- moment magnitude;
- Joyner–Boore distance;
- VS30-based soil class;
- faulting mechanism;
- focal depth.

The selected records are normalised with respect to their individual peak ground acceleration. This allows users to subsequently scale the records to different intensity levels, for example in incremental dynamic analysis or cloud-based nonlinear dynamic analyses.

---

## Repository structure

The repository is organised as follows:

```text
PEUSP-beta/
│
├── app.py
├── PEUSP_launcher.py
├── requirements.txt
├── LICENSE
├── README.md
│
├── RS_EC8.py
├── RS_EC82024.py
├── READER_ESM.py
│
├── AnnexD_1D_runner.py
├── AnnexD_2D_runner.py
├── EC82004_1D_runner.py
├── EC82004_2D_runner.py
├── user_1D_runner.py
├── user_2D_runner.py
├── Scenario_based_runner.py
│
├── PEUSP.png
├── 1D.png
├── 2D.png
│
└── metadata.csv
