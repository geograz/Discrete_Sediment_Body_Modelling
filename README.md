# Discrete_Sediment_Body_Modelling
This repository contains the codes for the paper:

_Stochastic 3D Modelling of Discrete Sediment Bodies for Geotechnical Applications_

by Georg H. Erharter, Franz Tschuchnigg and Gerhard Poscher

published in the __Applied Computing and Geosciences (Vol. 11; September 2021)__

DOI: https://doi.org/10.1016/j.acags.2021.100066

Use the `sediment_generator.py` Python code to generate your own sediment bodies as it is described in the paper.

## Case study and exemplary discrete sediment body models
The folder `case study` contains .stl geometry / mesh files and is made up as follows:
```
case study
├── Section_5_samples
│   └── set1.stl
│   └── set2.stl
│   └── set3.stl
├── discrete_sediment_bodies.stl
├── slope_excavated.stl
└── slope_unexcavated.stl
```

`set1.stl`, `set2.stl`, `set3.stl` are the exemplary discrete sediment body models as they are presented in section 5 of the publication and shown in figure 7

`discrete_sediment_bodies.stl` are the sediment bodies that were used for the FEM case study in section 6 and `slope_excavated.stl` and `slope_unexcavated.stl` are the geometries of the case study's slope before and after excavation.
