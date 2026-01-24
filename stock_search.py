"""
Stock Search Engine with Fuzzy Matching
Supports searching by ticker, company name, or partial matches
"""

from rapidfuzz import fuzz, process
from typing import List, Dict, Optional
import pandas as pd


class StockSearchEngine:
    """Intelligent stock search with fuzzy matching capabilities"""
    
    # Comprehensive stock database with ticker, name, and sector
    STOCK_DATABASE = [
        # Current supported stocks
        {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
        {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
        {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
        {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Technology"},
        {"ticker": "ASML", "name": "ASML Holding", "sector": "Technology"},
        {"ticker": "TSM", "name": "Taiwan Semiconductor", "sector": "Technology"},
        {"ticker": "WMT", "name": "Walmart Inc.", "sector": "Consumer"},
        {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial"},
        {"ticker": "MCD", "name": "McDonald's Corporation", "sector": "Consumer"},
        {"ticker": "TM", "name": "Toyota Motor Corporation", "sector": "Automotive"},
        
        # Extended stock universe for search
        {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Automotive"},
        {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Technology"},
        {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
        {"ticker": "NFLX", "name": "Netflix Inc.", "sector": "Technology"},
        {"ticker": "DIS", "name": "The Walt Disney Company", "sector": "Entertainment"},
        {"ticker": "BA", "name": "Boeing Company", "sector": "Aerospace"},
        {"ticker": "INTC", "name": "Intel Corporation", "sector": "Technology"},
        {"ticker": "CSCO", "name": "Cisco Systems Inc.", "sector": "Technology"},
        {"ticker": "ORCL", "name": "Oracle Corporation", "sector": "Technology"},
        {"ticker": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
        {"ticker": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
        {"ticker": "PYPL", "name": "PayPal Holdings Inc.", "sector": "Financial"},
        {"ticker": "V", "name": "Visa Inc.", "sector": "Financial"},
        {"ticker": "MA", "name": "Mastercard Incorporated", "sector": "Financial"},
        {"ticker": "BAC", "name": "Bank of America Corp", "sector": "Financial"},
        {"ticker": "GS", "name": "Goldman Sachs Group", "sector": "Financial"},
        {"ticker": "MS", "name": "Morgan Stanley", "sector": "Financial"},
        {"ticker": "WFC", "name": "Wells Fargo & Company", "sector": "Financial"},
        {"ticker": "C", "name": "Citigroup Inc.", "sector": "Financial"},
        {"ticker": "KO", "name": "Coca-Cola Company", "sector": "Consumer"},
        {"ticker": "PEP", "name": "PepsiCo Inc.", "sector": "Consumer"},
        {"ticker": "NKE", "name": "Nike Inc.", "sector": "Consumer"},
        {"ticker": "SBUX", "name": "Starbucks Corporation", "sector": "Consumer"},
        {"ticker": "PG", "name": "Procter & Gamble", "sector": "Consumer"},
        {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
        {"ticker": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare"},
        {"ticker": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare"},
        {"ticker": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare"},
        {"ticker": "MRK", "name": "Merck & Co. Inc.", "sector": "Healthcare"},
        {"ticker": "TMO", "name": "Thermo Fisher Scientific", "sector": "Healthcare"},
        {"ticker": "XOM", "name": "Exxon Mobil Corporation", "sector": "Energy"},
        {"ticker": "CVX", "name": "Chevron Corporation", "sector": "Energy"},
        {"ticker": "COP", "name": "ConocoPhillips", "sector": "Energy"},
        {"ticker": "SLB", "name": "Schlumberger Limited", "sector": "Energy"},
        {"ticker": "T", "name": "AT&T Inc.", "sector": "Telecommunications"},
        {"ticker": "VZ", "name": "Verizon Communications", "sector": "Telecommunications"},
        {"ticker": "CMCSA", "name": "Comcast Corporation", "sector": "Telecommunications"},
        {"ticker": "AVGO", "name": "Broadcom Inc.", "sector": "Technology"},
        {"ticker": "QCOM", "name": "Qualcomm Incorporated", "sector": "Technology"},
        {"ticker": "TXN", "name": "Texas Instruments", "sector": "Technology"},
        {"ticker": "IBM", "name": "IBM Corporation", "sector": "Technology"},
        {"ticker": "NOW", "name": "ServiceNow Inc.", "sector": "Technology"},
        {"ticker": "SHOP", "name": "Shopify Inc.", "sector": "Technology"},
        {"ticker": "SQ", "name": "Block Inc.", "sector": "Financial"},
        {"ticker": "COIN", "name": "Coinbase Global Inc.", "sector": "Financial"},
        {"ticker": "GE", "name": "General Electric", "sector": "Industrial"},
        {"ticker": "CAT", "name": "Caterpillar Inc.", "sector": "Industrial"},
        {"ticker": "MMM", "name": "3M Company", "sector": "Industrial"},
        {"ticker": "HON", "name": "Honeywell International", "sector": "Industrial"},
        {"ticker": "LMT", "name": "Lockheed Martin Corp", "sector": "Aerospace"},
        {"ticker": "RTX", "name": "Raytheon Technologies", "sector": "Aerospace"},
        {"ticker": "UPS", "name": "United Parcel Service", "sector": "Logistics"},
        {"ticker": "FDX", "name": "FedEx Corporation", "sector": "Logistics"},
        {"ticker": "BABA", "name": "Alibaba Group", "sector": "Technology"},
        {"ticker": "BIDU", "name": "Baidu Inc.", "sector": "Technology"},
        {"ticker": "NIO", "name": "NIO Inc.", "sector": "Automotive"},
        {"ticker": "F", "name": "Ford Motor Company", "sector": "Automotive"},
        {"ticker": "GM", "name": "General Motors", "sector": "Automotive"},
        {"ticker": "RIVN", "name": "Rivian Automotive", "sector": "Automotive"},
        {"ticker": "LCID", "name": "Lucid Group Inc.", "sector": "Automotive"},
        {"ticker": "HD", "name": "The Home Depot", "sector": "Retail"},
        {"ticker": "LOW", "name": "Lowe's Companies", "sector": "Retail"},
        {"ticker": "TGT", "name": "Target Corporation", "sector": "Retail"},
        {"ticker": "COST", "name": "Costco Wholesale", "sector": "Retail"},
        {"ticker": "CVS", "name": "CVS Health Corporation", "sector": "Healthcare"},
        {"ticker": "WBA", "name": "Walgreens Boots Alliance", "sector": "Healthcare"},
        {"ticker": "DHR", "name": "Danaher Corporation", "sector": "Healthcare"},
        {"ticker": "ABT", "name": "Abbott Laboratories", "sector": "Healthcare"},
        {"ticker": "LLY", "name": "Eli Lilly and Company", "sector": "Healthcare"},
        {"ticker": "BMY", "name": "Bristol-Myers Squibb", "sector": "Healthcare"},
        {"ticker": "GILD", "name": "Gilead Sciences", "sector": "Healthcare"},
        {"ticker": "AMGN", "name": "Amgen Inc.", "sector": "Healthcare"},
        {"ticker": "BIIB", "name": "Biogen Inc.", "sector": "Healthcare"},
        {"ticker": "REGN", "name": "Regeneron Pharmaceuticals", "sector": "Healthcare"},
        {"ticker": "VRTX", "name": "Vertex Pharmaceuticals", "sector": "Healthcare"},
        {"ticker": "ZTS", "name": "Zoetis Inc.", "sector": "Healthcare"},
        {"ticker": "ISRG", "name": "Intuitive Surgical", "sector": "Healthcare"},
        {"ticker": "BLK", "name": "BlackRock Inc.", "sector": "Financial"},
        {"ticker": "SCHW", "name": "Charles Schwab Corp", "sector": "Financial"},
        {"ticker": "CB", "name": "Chubb Limited", "sector": "Financial"},
        {"ticker": "AXP", "name": "American Express", "sector": "Financial"},
        {"ticker": "SPGI", "name": "S&P Global Inc.", "sector": "Financial"},
        {"ticker": "CME", "name": "CME Group Inc.", "sector": "Financial"},
        {"ticker": "ICE", "name": "Intercontinental Exchange", "sector": "Financial"},
        {"ticker": "TRV", "name": "The Travelers Companies", "sector": "Financial"},
        {"ticker": "PGR", "name": "Progressive Corporation", "sector": "Financial"},
        {"ticker": "ALL", "name": "Allstate Corporation", "sector": "Financial"},
        {"ticker": "MET", "name": "MetLife Inc.", "sector": "Financial"},
        {"ticker": "PRU", "name": "Prudential Financial", "sector": "Financial"},
        {"ticker": "AIG", "name": "American International Group", "sector": "Financial"},
    ]
    
    def __init__(self):
        """Initialize the search engine"""
        self.df = pd.DataFrame(self.STOCK_DATABASE)
        # Create searchable text combining ticker and name
        self.df['searchable'] = self.df['ticker'] + ' ' + self.df['name']
    
    def search_stocks(self, query: str, top_n: int = 5) -> List[Dict]:
        """
        Search for stocks using fuzzy matching
        
        Args:
            query: Search query (ticker, name, or partial)
            top_n: Number of top matches to return
            
        Returns:
            List of matching stocks with similarity scores
        """
        if not query or not query.strip():
            return []
        
        query = query.strip()
        
        # Search in searchable text (ticker + name)
        choices = self.df['searchable'].tolist()
        
        # Use rapidfuzz for fuzzy matching
        matches = process.extract(
            query,
            choices,
            scorer=fuzz.WRatio,  # Weighted ratio for better partial matching
            limit=top_n
        )
        
        results = []
        for match_text, score, idx in matches:
            if score > 40:  # Minimum similarity threshold
                stock = self.df.iloc[idx]
                results.append({
                    'ticker': stock['ticker'],
                    'name': stock['name'],
                    'sector': stock['sector'],
                    'similarity_score': round(score, 1)
                })
        
        return results
    
    def get_stock_info(self, ticker: str) -> Optional[Dict]:
        """
        Get detailed information for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Stock information dict or None if not found
        """
        ticker = ticker.upper().strip()
        match = self.df[self.df['ticker'] == ticker]
        
        if match.empty:
            return None
        
        stock = match.iloc[0]
        return {
            'ticker': stock['ticker'],
            'name': stock['name'],
            'sector': stock['sector']
        }
    
    def is_supported(self, ticker: str) -> bool:
        """
        Check if a ticker is in the supported list
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if supported, False otherwise
        """
        ticker = ticker.upper().strip()
        return ticker in self.df['ticker'].values
    
    def get_all_tickers(self) -> List[str]:
        """Get list of all supported tickers"""
        return sorted(self.df['ticker'].tolist())
    
    def get_stocks_by_sector(self, sector: str) -> List[Dict]:
        """
        Get all stocks in a specific sector
        
        Args:
            sector: Sector name
            
        Returns:
            List of stocks in that sector
        """
        matches = self.df[self.df['sector'].str.lower() == sector.lower()]
        return matches[['ticker', 'name', 'sector']].to_dict('records')


# Singleton instance for easy import
search_engine = StockSearchEngine()


# Convenience functions for backward compatibility
def search_stocks(query: str, top_n: int = 5) -> List[Dict]:
    """Search for stocks using fuzzy matching"""
    return search_engine.search_stocks(query, top_n)


def get_stock_info(ticker: str) -> Optional[Dict]:
    """Get stock information by ticker"""
    return search_engine.get_stock_info(ticker)


def is_supported(ticker: str) -> bool:
    """Check if ticker is supported"""
    return search_engine.is_supported(ticker)
