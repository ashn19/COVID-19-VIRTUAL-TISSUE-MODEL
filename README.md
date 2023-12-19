## Hey, I'm [Ashith Nirmal P!](https://github.com/ashn19) 👋.

### 📕 COVID 19 VIRTUAL TISSUE MODEL
- The current global pandemic of COVID-19, caused by the novel coronavirus Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2), has motivated the study of betacoronavirus diseases at multiple spatial and temporal computational modeling scales. The time course, severity of symptoms and complications from SARS-CoV-2 infection are highly
variable from patient to patient. Mathematical modeling methods integrate the available host- and pathogen-level data on disease dynamics that are required to understand the complex biology of infection and immune response to optimize therapeutic interventions. Mathematical models and computer simulations built on spatial and ODE frameworks have
been extensively used to study in-host progression of viral infection , with a recent acceleration in the development of spatial COVID-19 viral infection models in response to
the global pandemic. Building multiscale models of acute primary viral infection requires integration of submodels of multiple biological components across scales (e.g., viral replication and internalization, immune system responses). Non-spatial, coupled ordinary differential equation (ODE)
models can represent many aspects of pathogen-host interaction. In the context of viral infection dynamics, specialized ODE models can describe both the entire virus-host response at the tissue and organ levels and different stages of the viral replication cycle within cells, such as binding and internalization, viral genome replication and translation, assembly, packaging and release. By fitting ODE models to clinical or experimental data, researchers
have been able to estimate important parameters, such as the turnover rate of target cells, average lifetimes of viral particles and infected cells and the rate of production of new viral particles by infected cells. Other ODE models include pharmacokinetic models of drug availability

### 📕 METHODOLOGY
We begin by presenting our base multicellular model of viral infection in an epithelial tissue, along with a simulation for a baseline set of parameters and basic analyses. We then explore the simulation’s dependence on critical parameters that are important to the dynamics
of acute primary viral infection in airway epithelial cells. All simulations and spatial, population and system-level metrics presented in this section follow the specifications in Simulation Specifications. We performed simulations using the open-source modeling
environment CompuCell3D. Downloading and running the simulation provides instructions on how to run these simulations. We initialize the simulations with periodic boundary conditions parallel to the plane of the
sheet and Neumann boundary conditions perpendicular to the plane of the sheet. Initially, the extracellular environment does not contain any extracellular virus, cytokines, oxidative
agents or immune cells. We introduce infection by creating a single infected epithelial cell at the center of the epithelial sheet, comparably to (but less than) initial conditions of similar
works that model discrete cellular distributions. To illustrate the full range of dynamics of viral infection in the presence of an immune response, we established a baseline set of
parameters for which the immune response is strong enough to slow the spread of the infection, but insufficient to prevent widespread infection and death of all epithelial cells. While we have adjusted the parameters for the viral replication model to agree with reported
time scales for SARS-CoV-2 replication in vitro, and we have selected parameter values in physiologically reasonable ranges, we have not attempted to match other model parameters to a specific tissue, virus or host species. Furthermore, this baseline parameter set is not unique
with respect to its purpose, in that many sets of parameters can generate appreciable but insufficient inhibition of spread of infection. Rather, as is shown in subsequent sections, this
parameter set presents emergent dynamics of a theoretical virus and host immune response near, but not in, a regime of successful prevention of widespread infection, which is critical to showing the effects of underlying mechanisms on emergent dynamics and resulting outcome
