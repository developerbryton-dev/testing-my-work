from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any, Optional, Union

# --- 1. Custom Exception Hierarchy ---
class TaxError(Exception):
    """Base exception for all tax engine errors."""
    pass

class InvalidIncomeError(TaxError):
    """Raised when income or monetary values are negative or invalid."""
    pass

class NoScheduleForDateError(TaxError):
    """Raised when tax rates or schedules are unavailable for a given date/period."""
    pass

# --- 2. Money Value Object ---
class Money:
    """
    Represents monetary values in UGX using decimal.Decimal exclusively.
    Implements explicit rounding and required dunder methods.
    """
    def __init__(self, amount: Union[Decimal, int, float, str]):
        if isinstance(amount, float):
            raise TypeError("Float is unsafe for financial computations. Use Decimal, int, or str.")
        self.amount = Decimal(amount).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def __add__(self, other: 'Money') -> 'Money':
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self.amount + other.amount)

    def __sub__(self, other: 'Money') -> 'Money':
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self.amount - other.amount)

    def __mul__(self, factor: Union[Decimal, int, float]) -> 'Money':
        if isinstance(factor, float):
            raise TypeError("Multiplication by float is restricted in Money. Use Decimal or int.")
        return Money(self.amount * Decimal(str(factor)))

    def __lt__(self, other: 'Money') -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount < other.amount

    def __le__(self, other: 'Money') -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount <= other.amount

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return False
        return self.amount == other.amount

    def __repr__(self) -> str:
        return f"Money(UGX {self.amount:,.2f})"

# --- 3. Context & Result Data Classes ---
@dataclass
class TaxContext:
    """Holds input parameters for tax calculations."""
    gross_income: Money = field(default_factory=lambda: Money(0))
    allowances: Money = field(default_factory=lambda: Money(0))
    is_resident: bool = True
    deductions: Money = field(default_factory=lambda: Money(0))
    revenue: Money = field(default_factory=lambda: Money(0))
    allowable_deductions: Money = field(default_factory=lambda: Money(0))

@dataclass
class TaxResult:
    """Holds computed tax outputs."""
    tax_name: str
    tax_amount: Money
    breakdown: Dict[str, Any]

# --- 4. Abstract Base Class ---
class Tax(ABC):
    """Abstract base class for all tax calculators."""
    
    @abstractmethod
    def calculate(self, ctx: TaxContext) -> TaxResult:
        pass

# --- 5. Concrete Tax Implementations ---
class NSSFCalculator(Tax):
    """Calculates NSSF contributions (Employee 5%, Employer 10%)."""
    
    def calculate(self, ctx: TaxContext) -> TaxResult:
        if ctx.gross_income < Money(0):
            raise InvalidIncomeError("Gross income cannot be negative.")
        
        employee_contrib = ctx.gross_income * Decimal("0.05")
        employer_contrib = ctx.gross_income * Decimal("0.10")
        total_contrib = employee_contrib + employer_contrib
        
        return TaxResult(
            tax_name="NSSF",
            tax_amount=employee_contrib,
            breakdown={
                "employee_contribution": employee_contrib,
                "employer_contribution": employer_contrib,
                "total_nssf": total_contrib
            }
        )

class PAYECalculator(Tax):
    """
    Computes monthly PAYE for resident and non-resident employees 
    based on URA progressive bands and the 10,000,000 UGX 10% surcharge.
    """
    
    def calculate(self, ctx: TaxContext) -> TaxResult:
        taxable_income = ctx.gross_income + ctx.allowances - ctx.deductions
        if taxable_income < Money(0):
            raise InvalidIncomeError("Taxable income cannot be negative.")
        
        amt = taxable_income.amount
        tax = Decimal("0")
        
        if ctx.is_resident:
            if amt <= Decimal("335000"):
                tax = Decimal("0")
            elif amt <= Decimal("410000"):
                tax = (amt - Decimal("335000")) * Decimal("0.20")
            elif amt <= Decimal("485000"):
                tax = Decimal("15000") + (amt - Decimal("410000")) * Decimal("0.25")
            elif amt <= Decimal("10000000"):
                tax = Decimal("33750") + (amt - Decimal("485000")) * Decimal("0.30")
            else:
                base_tax_10m = Decimal("33750") + (Decimal("10000000") - Decimal("485000")) * Decimal("0.30")
                surcharge = (amt - Decimal("10000000")) * Decimal("0.40")
                tax = base_tax_10m + surcharge
        else:
            tax = amt * Decimal("0.30")
            
        tax_money = Money(tax)
        return TaxResult(
            tax_name="PAYE",
            tax_amount=tax_money,
            breakdown={"taxable_income": taxable_income, "resident": ctx.is_resident}
        )

class VATCalculator(Tax):
    """Calculates VAT (Standard 18%)."""
    
    def calculate(self, ctx: TaxContext) -> TaxResult:
        output_vat = ctx.revenue * Decimal("0.18")
        input_vat = ctx.allowable_deductions * Decimal("0.18")
        net_vat = output_vat - input_vat
        
        if net_vat.amount < Decimal("0"):
            net_vat = Money(0)
            
        return TaxResult(
            tax_name="VAT",
            tax_amount=net_vat,
            breakdown={"output_vat": output_vat, "input_vat": input_vat}
        )

class CorporateIncomeTaxCalculator(Tax):
    """Calculates Corporate Income Tax at 30% for resident companies."""
    
    def calculate(self, ctx: TaxContext) -> TaxResult:
        chargeable_income = ctx.revenue - ctx.allowable_deductions
        if chargeable_income < Money(0):
            chargeable_income = Money(0)
            
        tax_amount = chargeable_income * Decimal("0.30")
        return TaxResult(
            tax_name="Corporate Income Tax",
            tax_amount=tax_amount,
            breakdown={"chargeable_income": chargeable_income}
        )

# --- 6. Payroll Pipeline (Composition & Order of Deductions) ---
class PayrollPipeline:
    """
    Computes complete monthly payroll:
    Order of deductions: NSSF (Employee) -> LST (Local Service Tax) -> PAYE -> Net Pay.
    """
    def __init__(self, lst_amount: Union[Decimal, int, str] = 0):
        self.nssf_calc = NSSFCalculator()
        self.paye_calc = PAYECalculator()
        self.lst = Money(lst_amount)

    def process(self, gross_income: Money, allowances: Money = Money(0), is_resident: bool = True) -> Dict[str, Any]:
        ctx = TaxContext(gross_income=gross_income, allowances=allowances, is_resident=is_resident)
        
        # 1. NSSF
        nssf_res = self.nssf_calc.calculate(ctx)
        employee_nssf = nssf_res.tax_amount
        
        # 2. Taxable income after NSSF & LST deductions
        taxable_base = gross_income + allowances - employee_nssf - self.lst
        
        # 3. PAYE
        paye_ctx = TaxContext(gross_income=taxable_base, is_resident=is_resident)
        paye_res = self.paye_calc.calculate(paye_ctx)
        paye_tax = paye_res.tax_amount
        
        # 4. Net Pay
        total_deductions = employee_nssf + self.lst + paye_tax
        net_pay = gross_income + allowances - total_deductions
        
        return {
            "gross_income": gross_income,
            "allowances": allowances,
            "employee_nssf": employee_nssf,
            "local_service_tax": self.lst,
            "taxable_income_for_paye": taxable_base,
            "paye": paye_tax,
            "net_pay": net_pay
        }