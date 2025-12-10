"""Unit tests for intanalysis mappings module."""

import pytest
from intanalysis.mappings import (
    COMPANY_TO_STOCK, SECTOR_TO_COMPANIES, REGULATORS,
    get_stock_symbol, get_companies_in_sector, get_sectors_for_company
)


class TestCompanyToStock:
    """Tests for COMPANY_TO_STOCK mapping."""
    
    def test_mapping_structure(self):
        """Test that mapping has expected structure."""
        for key, value in COMPANY_TO_STOCK.items():
            assert isinstance(key, str)
            assert isinstance(value, tuple)
            assert len(value) == 3
            symbol, name, aliases = value
            assert isinstance(symbol, str)
            assert isinstance(name, str)
            assert isinstance(aliases, list)
    
    def test_known_companies_exist(self):
        """Test that known companies are in mapping."""
        assert "hdfc bank" in COMPANY_TO_STOCK
        assert "icici bank" in COMPANY_TO_STOCK
        assert "sbi" in COMPANY_TO_STOCK
        assert "tcs" in COMPANY_TO_STOCK
        assert "infosys" in COMPANY_TO_STOCK
    
    def test_hdfc_bank_mapping(self):
        """Test HDFC Bank mapping details."""
        symbol, name, aliases = COMPANY_TO_STOCK["hdfc bank"]
        assert symbol == "HDFCBANK"
        assert name == "HDFC Bank Limited"
        assert "hdfc" in aliases or "hdfcbank" in aliases


class TestSectorToCompanies:
    """Tests for SECTOR_TO_COMPANIES mapping."""
    
    def test_mapping_structure(self):
        """Test that mapping has expected structure."""
        for sector, companies in SECTOR_TO_COMPANIES.items():
            assert isinstance(sector, str)
            assert isinstance(companies, list)
            assert all(isinstance(c, str) for c in companies)
    
    def test_known_sectors_exist(self):
        """Test that known sectors are in mapping."""
        assert "Banking" in SECTOR_TO_COMPANIES
        assert "IT" in SECTOR_TO_COMPANIES
        assert "Aviation" in SECTOR_TO_COMPANIES
        assert "Automobile" in SECTOR_TO_COMPANIES
    
    def test_banking_sector_companies(self):
        """Test Banking sector has expected companies."""
        banking_companies = SECTOR_TO_COMPANIES["Banking"]
        assert "HDFCBANK" in banking_companies
        assert "ICICIBANK" in banking_companies
        assert "SBIN" in banking_companies
    
    def test_it_sector_companies(self):
        """Test IT sector has expected companies."""
        it_companies = SECTOR_TO_COMPANIES["IT"]
        assert "TCS" in it_companies
        assert "INFY" in it_companies


class TestRegulators:
    """Tests for REGULATORS mapping."""
    
    def test_mapping_structure(self):
        """Test that mapping has expected structure."""
        for key, info in REGULATORS.items():
            assert isinstance(key, str)
            assert isinstance(info, dict)
            assert "full_name" in info
            assert "aliases" in info
            assert "sectors" in info
    
    def test_rbi_mapping(self):
        """Test RBI regulator mapping."""
        assert "rbi" in REGULATORS
        rbi = REGULATORS["rbi"]
        assert rbi["full_name"] == "Reserve Bank of India"
        assert "Banking" in rbi["sectors"] or "Financial Services" in rbi["sectors"]
    
    def test_sebi_mapping(self):
        """Test SEBI regulator mapping."""
        assert "sebi" in REGULATORS
        sebi = REGULATORS["sebi"]
        assert sebi["full_name"] == "Securities and Exchange Board of India"


class TestGetStockSymbol:
    """Tests for get_stock_symbol function."""
    
    def test_exact_company_match(self):
        """Test exact company name match."""
        result = get_stock_symbol("hdfc bank")
        assert result is not None
        symbol, name, confidence = result
        assert symbol == "HDFCBANK"
        assert name == "HDFC Bank Limited"
    
    def test_partial_match_via_alias(self):
        """Test match via alias."""
        result = get_stock_symbol("hdfc")
        assert result is not None
        symbol, name, confidence = result
        assert symbol == "HDFCBANK"
    
    def test_case_insensitive(self):
        """Test that matching is case insensitive."""
        result1 = get_stock_symbol("HDFC Bank")
        result2 = get_stock_symbol("hdfc bank")
        assert result1 == result2
    
    def test_no_match_returns_none(self):
        """Test that unknown company returns None."""
        result = get_stock_symbol("unknown company xyz")
        assert result is None
    
    def test_tcs_match(self):
        """Test TCS matching."""
        result = get_stock_symbol("tcs wins deal")
        assert result is not None
        assert result[0] == "TCS"
    
    def test_infosys_match(self):
        """Test Infosys matching."""
        result = get_stock_symbol("infosys reports earnings")
        assert result is not None
        assert result[0] == "INFY"


class TestGetCompaniesInSector:
    """Tests for get_companies_in_sector function."""
    
    def test_banking_sector(self):
        """Test getting companies in Banking sector."""
        companies = get_companies_in_sector("Banking")
        assert isinstance(companies, list)
        assert len(companies) > 0
        assert "HDFCBANK" in companies
        assert "ICICIBANK" in companies
    
    def test_it_sector(self):
        """Test getting companies in IT sector."""
        companies = get_companies_in_sector("IT")
        assert "TCS" in companies
        assert "INFY" in companies
    
    def test_unknown_sector_returns_empty(self):
        """Test that unknown sector returns empty list."""
        companies = get_companies_in_sector("Unknown Sector")
        assert companies == []
    
    def test_aviation_sector(self):
        """Test getting companies in Aviation sector."""
        companies = get_companies_in_sector("Aviation")
        assert "INDIGO" in companies


class TestGetSectorsForCompany:
    """Tests for get_sectors_for_company function."""
    
    def test_hdfc_bank_sectors(self):
        """Test getting sectors for HDFC Bank."""
        sectors = get_sectors_for_company("HDFCBANK")
        assert isinstance(sectors, list)
        assert "Banking" in sectors
    
    def test_tcs_sectors(self):
        """Test getting sectors for TCS."""
        sectors = get_sectors_for_company("TCS")
        assert "IT" in sectors
    
    def test_company_in_multiple_sectors(self):
        """Test company that might be in multiple sectors."""
        # HDFC Bank is in both Banking and Financial Services
        sectors = get_sectors_for_company("HDFCBANK")
        assert len(sectors) >= 1
    
    def test_unknown_company_returns_empty(self):
        """Test that unknown company returns empty list."""
        sectors = get_sectors_for_company("UNKNOWNSYMBOL")
        assert sectors == []
