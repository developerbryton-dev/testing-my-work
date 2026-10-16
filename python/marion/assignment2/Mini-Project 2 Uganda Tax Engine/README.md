# Mini-Project 2: Uganda Tax Engine (URA Compliance & OOP Design)

## Overview
This project implements an extensible, robust, and highly testable Object-Oriented tax calculation engine modeling the Uganda Revenue Authority (URA) statutory tax rules. It is engineered with strict financial precision, custom exception handling, and comprehensive automated testing.

---

## Regulatory Baseline & Citations
The tax rules implemented in this engine follow official URA guidelines and statutory frameworks:
- **PAYE (Pay-As-You-Earn):** Progressive monthly resident bands (Nil up to UGX 335,000; 20% up to UGX 410,000; 25% up to UGX 485,000; 30% up to UGX 10,000,000; and a 10% surcharge on amounts exceeding UGX 10,000,000) and non-resident flat schedule rules. 
- **NSSF (National Social Security Fund):** Statutory contributions comprising 5% employee deduction and 10% employer contribution (*Source: NSSF Uganda Act / Sunpro Calculator baseline*).
- **VAT (Value Added Tax):** Standard rate of 18% applied to taxable supplies, supporting input VAT credits and net payable computations.
- **Corporate Income Tax (CIT):** 30% standard rate applied to chargeable corporate income (Revenue minus allowable deductions).
- **Local Service Tax (LST):** Configurable optional fixed local deduction applied prior to PAYE computation.

---

## Repository Structure
- `src/`: Contains the core OOP engine (`tax_engine.py`), implementing the `Money` value object, custom exceptions, abstract `Tax` base class, and concrete calculators.
- `tests/`: Contains robust test suites (`test_tax_engine.py`) including unit tests, band boundary assertions, and property-based tests via `hypothesis`.
- `notebooks/`: Contains the interactive Jupyter Notebook (`project2_tax.ipynb`) demonstrating pipeline executions and answering design discussion questions.

---

## Setup and Installation

1. **Open your terminal/command prompt** inside the project folder.
2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt