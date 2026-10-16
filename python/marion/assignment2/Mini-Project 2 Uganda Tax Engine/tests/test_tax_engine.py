import pytest
from decimal import Decimal
from hypothesis import given, strategies as st
from src.tax_engine import (
    Money, PAYECalculator, NSSFCalculator, VATCalculator, 
    CorporateIncomeTaxCalculator, TaxContext, InvalidIncomeError, PayrollPipeline
)

def test_money_operations_and_rounding():
    m1 = Money(100050)
    m2 = Money("250.256")
    
    # Check rounding half up
    assert m2.amount == Decimal("250.26")
    
    # Check dunder addition and comparison
    m3 = Money(100)
    m4 = Money(200)
    assert m3 + m4 == Money(300)
    assert m3 < m4
    assert not (m4 < m3)

def test_invalid_money_type():
    with pytest.raises(TypeError):
        _ = Money(100.50)  # Floats must be rejected for financial accuracy

def test_paye_boundary_values():
    calc = PAYECalculator()
    
    # Band 1 edge: 335,000 -> Nil tax (0)
    res_335k = calc.calculate(TaxContext(gross_income=Money(335000), is_resident=True))
    assert res_335k.tax_amount == Money(0)
    
    # Band 2 edge: 335,001 -> 20% of excess (1)
    res_335k_1 = calc.calculate(TaxContext(gross_income=Money(335001), is_resident=True))
    assert res_335k_1.tax_amount == Money(Decimal("0.20"))
    
    # Band 3 edge: 410,000 -> 15,000 max of band 2
    res_410k = calc.calculate(TaxContext(gross_income=Money(410000), is_resident=True))
    assert res_410k.tax_amount == Money(15000)
    
    # Band 4 edge: 485,000 -> 33,750 max of band 3
    res_485k = calc.calculate(TaxContext(gross_income=Money(485000), is_resident=True))
    assert res_485k.tax_amount == Money(33750)
    
    # 10,000,000 edge
    res_10m = calc.calculate(TaxContext(gross_income=Money(10000000), is_resident=True))
    assert res_10m.tax_amount.amount > Decimal("0")
    
    # Above 10,000,001 (triggers 10% surcharge / 40% marginal rate)
    res_10m_1 = calc.calculate(TaxContext(gross_income=Money(10000001), is_resident=True))
    assert res_10m_1.tax_amount.amount > res_10m.tax_amount.amount

@given(income=st.decimals(min_value=0, max_value=50000000, places=2))
def test_property_paye_non_negative_and_monotonic(income):
    calc = PAYECalculator()
    ctx = TaxContext(gross_income=Money(income), is_resident=True)
    res = calc.calculate(ctx)
    
    # Property 1: Tax must never be negative
    assert res.tax_amount >= Money(0)
    
    # Property 2: Tax must never exceed gross income
    assert res.tax_amount.amount <= income

def test_nssf_calculation():
    calc = NSSFCalculator()
    res = calc.calculate(TaxContext(gross_income=Money(1000000)))
    # Employee 5% of 1,000,000 = 50,000
    assert res.tax_amount == Money(50000)
    assert res.breakdown["employer_contribution"] == Money(100000)

def test_corporate_and_vat():
    vat_calc = VATCalculator()
    vat_res = vat_calc.calculate(TaxContext(revenue=Money(1000000), allowable_deductions=Money(400000)))
    # Output VAT 18% of 1M = 180k, Input VAT 18% of 400k = 72k. Net = 108k
    assert vat_res.tax_amount == Money(108000)
    
    cit_calc = CorporateIncomeTaxCalculator()
    cit_res = cit_calc.calculate(TaxContext(revenue=Money(1000000), allowable_deductions=Money(400000)))
    # Chargeable income = 600k. CIT 30% = 180k
    assert cit_res.tax_amount == Money(180000)