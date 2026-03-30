"""Stock and entity mappings for supported markets."""

# Company name -> (symbol, full_name, aliases)
COMPANY_TO_STOCK: dict[str, tuple[str, str, list[str]]] = {
    # Hong Kong / China ADRs
    "泡泡玛特": ("9992.HK", "泡泡玛特", ["泡泡马特", "pop mart", "popmart"]),
    "腾讯控股": ("0700.HK", "腾讯控股", ["腾讯", "tencent", "tencent holdings"]),
    "阿里巴巴": ("9988.HK", "阿里巴巴", ["阿里", "阿里巴巴集团", "alibaba", "baba"]),
    "美团": ("3690.HK", "美团", ["美团-w", "meituan"]),
    "京东集团": ("9618.HK", "京东集团", ["京东", "jd", "jd.com"]),
    "小米集团": ("1810.HK", "小米集团", ["小米", "xiaomi"]),
    "比亚迪股份": ("1211.HK", "比亚迪股份", ["比亚迪", "byd"]),
    "网易": ("9999.HK", "网易", ["netease", "网易-s"]),
    "百度集团": ("9888.HK", "百度集团", ["百度", "baidu"]),
    "快手": ("1024.HK", "快手", ["kuaishou", "快手-w"]),
    "哔哩哔哩": ("9626.HK", "哔哩哔哩", ["b站", "bilibili"]),
    "海尔智家": ("6690.HK", "海尔智家", ["haier", "haier smart home"]),
    "拼多多": ("PDD", "拼多多", ["pdd", "pinduoduo"]),
    "理想汽车": ("2015.HK", "理想汽车", ["理想", "li auto"]),
    "小鹏汽车": ("9868.HK", "小鹏汽车", ["小鹏", "xpeng"]),
    "蔚来": ("9866.HK", "蔚来", ["nio"]),
    "宁德时代": ("300750.SZ", "宁德时代", ["catl"]),
    "贵州茅台": ("600519.SH", "贵州茅台", ["茅台", "kweichow moutai"]),
    "工商银行": ("601398.SH", "工商银行", ["工行", "icbc"]),
    "建设银行": ("601939.SH", "建设银行", ["建行", "ccb"]),
    "农业银行": ("601288.SH", "农业银行", ["农行", "abc bank"]),
    "中国银行": ("601988.SH", "中国银行", ["中行", "bank of china"]),

    # India
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
    "Internet": ["0700.HK", "9988.HK", "3690.HK", "9618.HK", "9999.HK", "9888.HK", "1024.HK", "9626.HK", "PDD"],
    "E-Commerce": ["9988.HK", "3690.HK", "9618.HK", "PDD"],
    "Consumer": ["9992.HK", "600519.SH"],
    "Consumer Electronics": ["1810.HK"],
    "EV": ["1211.HK", "2015.HK", "9868.HK", "9866.HK", "300750.SZ"],
    "Home Appliances": ["6690.HK"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    "Chinese Banking": ["601398.SH", "601939.SH", "601288.SH", "601988.SH"],
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH"],
    "Aviation": ["INDIGO", "SPICEJET"],
    "Automobile": ["TATAMOTORS", "MARUTI", "BAJAJ-AUTO"],
    "Financial Services": ["HDFCBANK", "ICICIBANK", "SBIN"],
}

# Regulator mappings
REGULATORS: dict[str, dict] = {
    "中国证监会": {"full_name": "中国证监会", "aliases": ["证监会", "csrc"], "sectors": ["Financial Services", "Chinese Banking"]},
    "香港证监会": {"full_name": "香港证监会", "aliases": ["香港证券及期货事务监察委员会", "sfc"], "sectors": ["Financial Services"]},
    "港交所": {"full_name": "香港交易及结算所有限公司", "aliases": ["香港交易所", "hkex"], "sectors": ["Financial Services"]},
    "中国人民银行": {"full_name": "中国人民银行", "aliases": ["人民银行", "央行", "pboc"], "sectors": ["Chinese Banking", "Financial Services"]},
    "rbi": {"full_name": "Reserve Bank of India", "aliases": ["reserve bank", "central bank"], "sectors": ["Banking", "Financial Services"]},
    "sebi": {"full_name": "Securities and Exchange Board of India", "aliases": ["securities board"], "sectors": ["Financial Services"]},
    "dgca": {"full_name": "Directorate General of Civil Aviation", "aliases": ["aviation regulator"], "sectors": ["Aviation"]},
    "irdai": {"full_name": "Insurance Regulatory and Development Authority", "aliases": ["insurance regulator"], "sectors": ["Insurance"]},
}


def find_stock_symbols(text: str) -> list[tuple[str, str, float]]:
    """Find all matching stock symbols from text ordered by match quality."""
    text_lower = text.lower()
    matches: list[tuple[str, str, float, int]] = []

    for key, (symbol, name, aliases) in COMPANY_TO_STOCK.items():
        if key in text_lower:
            matches.append((symbol, name, 1.0, len(key)))
        for alias in aliases:
            if alias in text_lower:
                matches.append((symbol, name, 0.9, len(alias)))

    if not matches:
        return []

    # Keep the strongest match for each symbol, then sort globally.
    best_by_symbol: dict[str, tuple[str, str, float, int]] = {}
    for match in matches:
        symbol = match[0]
        existing = best_by_symbol.get(symbol)
        if existing is None or match[2] > existing[2] or (match[2] == existing[2] and match[3] > existing[3]):
            best_by_symbol[symbol] = match

    ranked = sorted(best_by_symbol.values(), key=lambda item: (item[2], item[3]), reverse=True)
    return [(symbol, name, confidence) for symbol, name, confidence, _ in ranked]


def get_stock_symbol(text: str) -> tuple[str, str, float] | None:
    """Find the strongest stock symbol match from text."""
    matches = find_stock_symbols(text)
    if not matches:
        return None
    return matches[0]


def get_companies_in_sector(sector: str) -> list[str]:
    """Get all company symbols in a sector."""
    return SECTOR_TO_COMPANIES.get(sector, [])


def get_sectors_for_company(symbol: str) -> list[str]:
    """Get sectors a company belongs to."""
    return [s for s, companies in SECTOR_TO_COMPANIES.items() if symbol in companies]
