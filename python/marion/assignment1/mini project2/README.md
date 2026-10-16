# Uganda Numerical Projects

A repository containing modular Python software and numerical modeling workflows set in Ugandan contexts.

## Repository Structure
```text
uganda-numerical-projects/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── population.py        # Domain class & routines for Project 1 (Demographics)
│   ├── microgrid.py         # Domain class & solver for Project 2 (Kasese Solar Micro-Grid)
│   ├── fish_model.py        # Domain classes for Project 3 (Lake Victoria Fish Stock)
│   └── utils.py             # Shared plot stylers & formatting helpers
├── notebooks/
│   ├── project1_population.ipynb
│   ├── project2_analysis.ipynb
│   └── project3_fish_model.ipynb
└── tests/
    ├── test_population.py
    ├── test_microgrid.py
    └── test_fish_model.py