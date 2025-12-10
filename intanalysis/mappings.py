"""Stock and entity mappings for Indian markets."""

# Company name -> (symbol, full_name, aliases)
COMPANY_TO_STOCK: dict[str, tuple[str, str, list[str]]] = {
    "hdfc bank": ("HDFCBANK", "HDFC Bank Limited", ["hdfc", "hdfcbank"]),
    "icici bank": ("ICICIBANK", "ICICI Bank Limited", ["icici"]),
    "sbi": ("SBIN", "State Bank of India", ["state bank"]),
    "axis bank": ("AXISBANK", "Axis Bank Limited", ["axis"]),
    "kotak": ("KOTAKBANK", "Kotak Mahindra Bank", ["kotak bank", "kotak mahindra"]),
    "reliance": ("RELIANCE", "Reliance Industries", ["ril", "reliance industries"]),
    "tcs": ("TCS", "Tata Consultancy Services", ["tata consultancy"]),
    "infosys": ("INFY", "Infosys Limited", ["infy"]),
    "wipro": ("WIPRO", "Wipro Limited", []),
    "hcl": ("HCLTECH", "HCL Technologies", ["hcl tech"]),
    "indigo": ("INDIGO", "InterGlobe Aviation", ["interglobe"]),
    "air india": ("AIRINDIA", "Air India Limited", []),
    "spicejet": ("SPICEJET", "SpiceJet Limited", []),
    "tata motors": ("TATAMOTORS", "Tata Motors Limited", ["tata motor"]),
    "maruti": ("MARUTI", "Maruti Suzuki", ["maruti suzuki"]),
    "bajaj auto": ("BAJAJ-AUTO", "Bajaj Auto Limited", ["bajaj"]),
}

# Sector -> list of stock symbols
SECTOR_TO_COMPANIES: dict[str, list[str]] = {
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH"],
    "Aviation": ["INDIGO", "SPICEJET"],
    "Automobile": ["TATAMOTORS", "MARUTI", "BAJAJ-AUTO"],
    "Financial Services": ["HDFCBANK", "ICICIBANK", "SBIN"],
}

# Regulator mappings
REGULATORS: dict[str, dict] = {
    "rbi": {"full_name": "Reserve Bank of India", "aliases": ["reserve bank", "central bank"], "sectors": ["Banking", "Financial Services"]},
    "sebi": {"full_name": "Securities and Exchange Board of India", "aliases": ["securities board"], "sectors": ["Financial Services"]},
    "dgca": {"full_name": "Directorate General of Civil Aviation", "aliases": ["aviation regulator"], "sectors": ["Aviation"]},
    "irdai": {"full_name": "Insurance Regulatory and Development Authority", "aliases": ["insurance regulator"], "sectors": ["Insurance"]},
}


def get_stock_symbol(text: str) -> tuple[str, str, float] | None:
    """Find stock symbol from text. Returns (symbol, name, confidence)."""
    text_lower = text.lower()
    for key, (symbol, name, aliases) in COMPANY_TO_STOCK.items():
        if key in text_lower:
            return (symbol, name, 1.0)
        for alias in aliases:
            if alias in text_lower:
                return (symbol, name, 0.9)
    return None


def get_companies_in_sector(sector: str) -> list[str]:
    """Get all company symbols in a sector."""
    return SECTOR_TO_COMPANIES.get(sector, [])


def get_sectors_for_company(symbol: str) -> list[str]:
    """Get sectors a company belongs to."""
    return [s for s, companies in SECTOR_TO_COMPANIES.items() if symbol in companies]
