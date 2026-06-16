# PEUSP-beta-a-web-based-ground-motion-record-selection-platform-for-seismic-safety-assessment

**PEUSP_beta** is a Python-based web application for the selection and scaling of earthquake ground motion records for seismic safety assessment and performance-based earthquake engineering applications.

The platform supports the selection of recorded European strong-motion data through two main workflows: **spectral matching** and **scenario-based selection**. The spectral-matching workflow allows users to select one-dimensional (1D) or two-dimensional (2D) horizontal ground motion components and scale them to target spectra defined according to **EN 1998-1:2004**, **EN 1998-1-1:2024**, or a **user-defined target spectrum**. The scenario-based workflow enables ground motion selection based on seismological and site-related parameters, including moment magnitude, Joyner–Boore distance, VS30-based soil class, faulting mechanism, and focal depth.

PEUSP_beta is implemented in Python using **Streamlit** as a browser-based graphical user interface. The selection and scaling procedures are formulated as optimisation problems, using metaheuristic algorithms to identify suites of ground motions that satisfy the imposed spectral and seismological constraints. The platform is directly linked to the **European Strong-Motion (ESM) database**, allowing the automatic retrieval and preparation of selected records.

The output of the software includes selected record metadata, scaling factors, processed acceleration time histories, and quick-look plots of acceleration, velocity, displacement, and response spectra. The exported records are intended to facilitate subsequent nonlinear dynamic analyses in structural engineering applications.
