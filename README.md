*** project in progress
# Identification of the 11-Parameter Functional Form Model for Photovoltaic Modules Using Manufacturer-Provided Ratings




## Abstrac
In light of the ongoing decline in photovoltaic (PV) generation costs and its growing competitiveness with retail electricity prices, accurately predicting PV performance is increasingly important. While manufacturers have typically rated PV modules at standard test conditions (STC), their ratings are now being enhanced by reporting module data at low irradiance conditions (LIC) and nominal operating cell temperature (NOCT). Recently, an enhanced PV model was proposed [[1]][[1]](https://ieeexplore.ieee.org/document/10187624), capable of reproducing the behavior of a PV module across a wide range of atmospheric conditions. Although the superiority of this model is thoroughly discussed in [[1]](https://ieeexplore.ieee.org/document/10187624), the identification of its characterizing parameters from ratings provided by manufacturers is not addressed. This paper proposes a parameter identification methodology relying on STC, LIC, and NOCT ratings. The problem at hand involves solving a complex system of eleven nonlinear equations, and is approached by progressively reducing the search space and generating adjustment functions. The methodology is tested in an automated fashion over the entire California Energy Commission PV database, which currently contains 17,710 modules, achieving a convergence rate of 99.8\%. The quality of the identified model is assessed by comparing energy predictions against experimental measurements, including state-of-the-art models available in the literature. Results indicate that the identified model reduces prediction errors by about 9\% compared to the best competitive model.


![Example of search space and uniqueness of solution](https://raw.githubusercontent.com/DIE-UTFSM-AA/ParameterIdentification_11PFF/refs/heads/main/FIgs/fig1.png)





## Installation
For the installation and start-up of the repository, it is necessary to have the following libraries:


* [ipython==8.12.3](https://ipython.org/)
* [matplotlib==3.10.3](https://matplotlib.org/)
* [mpmath==1.3.0](https://mpmath.org/)
* [numpy==2.2.5](https://numpy.org/install/)
* [pandas==2.2.3](https://pandas.pydata.org/docs/getting_started/install.html)
* [PyQt6==6.9.0](https://doc.qt.io/qtforpython-6/)
* [scipy==1.15.3](https://scipy.org/)
* [seaborn==0.13.2](https://seaborn.pydata.org/)
* [sympy==1.13.1](https://www.sympy.org/en/index.html)



### Database

The data used to validate the convergence of the algorithmic proposal correspond to [PV Module List](https://www.energy.ca.gov/media/2367) from the California Energy Commission (CEC).

The data used for out-of-sample validation in this work are obtained from [Data for Validating Models for PV Module Performance](https://datahub.duramat.org/dataset/data-for-validating-models-for-pv-module-performance) from National Renewable Energy Laboratory (NREL).











## References
[[1]](https://ieeexplore.ieee.org/document/10187624)


## Citation
    @ARTICLE{10935289,
    author={Angulo, Alejandro and Huerta, Miguel and Mancilla–David, Fernando},
    journal={IEEE Transactions on Industrial Informatics}, 
    title={Identification of the 11-Parameter Functional Form Model for Photovoltaic Modules Using Manufacturer-Provided Ratings}, 
    year={2025},
    volume={},
    number={},
    pages={1-9},
    abstract={In light of the ongoing decline in photovoltaic (PV) generation costs and its growing competitiveness with retail electricity prices, accurately predicting PV performance is increasingly important. While manufacturers have typically rated PV modules at standard test conditions (STCs), their ratings are now being enhanced by reporting module data at low irradiance conditions (LICs) and nominal operating cell temperature (NOCT). Recently, an enhanced PV model was proposed Angulo et al., 2024, capable of reproducing the behavior of a PV module across a wide range of atmospheric conditions. Although the superiority of this model is thoroughly discussed in Angulo et al., 2024, the identification of its characterizing parameters from ratings provided by manufacturers is not addressed. This paper proposes a parameter identification methodology relying on STC, LIC, and NOCT ratings. The problem at hand involves solving a complex system of eleven nonlinear equations, and is approached by progressively reducing the search space and generating adjustment functions. The methodology is tested in an automated fashion over the entire California Energy Commission PV database, which currently contains 17 710 modules, achieving a convergence rate of 99.8%. The quality of the identified model is assessed by comparing energy predictions against experimental measurements, including state-of-the-art models available in the literature. Results indicate that the identified model reduces prediction errors by about 9% compared to the best competitive model.},
    keywords={Mathematical models;Atmospheric modeling;Computational modeling;Databases;Vectors;Standards;Predictive models;Parameter estimation;Silicon;IEC Standards;Parameter identification;photovoltaic (PV) module characterization;single-diode model (SDM);transcendental equations},
    doi={10.1109/TII.2025.3545086},
    ISSN={1941-0050},
    month={},}


    
