"""
On-Chain Data Service
Provides free blockchain data from:
- Etherscan: Ethereum transactions, gas, token transfers (free: 5 req/s)
- Public JSON-RPC: Direct blockchain queries (free, no key)
- Blockchain.info: Bitcoin stats (free, no key)
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.onchain')


# ---------------------------------------------------------------------------
# Etherscan adapter (free tier: 5 calls/sec, 100k calls/day)
# ---------------------------------------------------------------------------
class EtherscanAdapter:
    """
    Ethereum blockchain data via Etherscan API.
    Free tier: 5 calls/sec, 100,000 calls/day.
    """

    BASE_URL = "https://api.etherscan.io/api"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.ETHERSCAN_API_KEY
        self.available = bool(self.api_key)
        if not self.available:
            logger.warning(
                "ETHERSCAN_API_KEY not set — Etherscan data unavailable. "
                "Get a free key at https://etherscan.io/apis"
            )

    def _get(self, **params) -> Any:
        import requests
        params["apikey"] = self.api_key
        resp = requests.get(self.BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "0" and data.get("message") != "No transactions found":
            raise RuntimeError(f"Etherscan API error: {data.get('result', data.get('message'))}")
        return data.get("result")

    def get_eth_price(self) -> Dict:
        """Current ETH price in USD and BTC."""
        if not self.available:
            raise RuntimeError("Etherscan API key not configured")

        result = self._get(module="stats", action="ethprice")
        return {
            "eth_usd": float(result.get("ethusd", 0)),
            "eth_btc": float(result.get("ethbtc", 0)),
            "timestamp": datetime.utcfromtimestamp(
                int(result.get("ethusd_timestamp", 0))
            ).isoformat() if result.get("ethusd_timestamp") else None,
        }

    def get_gas_oracle(self) -> Dict:
        """Current gas prices (low, average, high)."""
        if not self.available:
            raise RuntimeError("Etherscan API key not configured")

        result = self._get(module="gastracker", action="gasoracle")
        return {
            "low_gwei": float(result.get("SafeGasPrice", 0)),
            "average_gwei": float(result.get("ProposeGasPrice", 0)),
            "high_gwei": float(result.get("FastGasPrice", 0)),
            "base_fee_gwei": float(result.get("suggestBaseFee", 0)),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_eth_supply(self) -> Dict:
        """Total ETH supply stats."""
        if not self.available:
            raise RuntimeError("Etherscan API key not configured")

        result = self._get(module="stats", action="ethsupply2")
        # result is a dict with supply breakdown
        if isinstance(result, dict):
            return {
                "eth_supply": float(result.get("EthSupply", 0)) / 1e18,
                "eth2_staking": float(result.get("Eth2Staking", 0)) / 1e18,
                "burnt_fees": float(result.get("BurntFees", 0)) / 1e18,
                "timestamp": datetime.utcnow().isoformat(),
            }
        # Fallback: simple supply as string
        return {
            "eth_supply": float(result) / 1e18 if result else None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_balance(self, address: str) -> Dict:
        """ETH balance for an address."""
        if not self.available:
            raise RuntimeError("Etherscan API key not configured")

        result = self._get(module="account", action="balance", address=address, tag="latest")
        balance_eth = float(result) / 1e18 if result else 0

        return {
            "address": address,
            "balance_eth": round(balance_eth, 6),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_token_transfers(
        self,
        address: str,
        contract_address: Optional[str] = None,
        page: int = 1,
        offset: int = 20,
    ) -> List[Dict]:
        """
        ERC-20 token transfers for an address.

        Args:
            address: Wallet address
            contract_address: Optional token contract to filter
            page: Page number
            offset: Results per page (max 10000)
        """
        if not self.available:
            raise RuntimeError("Etherscan API key not configured")

        params = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "page": page,
            "offset": offset,
            "sort": "desc",
        }
        if contract_address:
            params["contractaddress"] = contract_address

        result = self._get(**params)
        if not isinstance(result, list):
            return []

        transfers = []
        for tx in result:
            decimals = int(tx.get("tokenDecimal", 18))
            value = float(tx.get("value", 0)) / (10 ** decimals)
            transfers.append({
                "hash": tx.get("hash"),
                "from": tx.get("from"),
                "to": tx.get("to"),
                "value": round(value, 6),
                "token_name": tx.get("tokenName"),
                "token_symbol": tx.get("tokenSymbol"),
                "timestamp": datetime.utcfromtimestamp(
                    int(tx.get("timeStamp", 0))
                ).isoformat() if tx.get("timeStamp") else None,
                "gas_used": int(tx.get("gasUsed", 0)),
            })
        return transfers

    def get_whale_transactions(
        self,
        address: str,
        min_value_eth: float = 100,
        offset: int = 50,
    ) -> List[Dict]:
        """
        Large ETH transactions for a known whale address.

        Args:
            address: Wallet address to monitor
            min_value_eth: Minimum transaction value in ETH
            offset: Number of recent txns to check
        """
        if not self.available:
            raise RuntimeError("Etherscan API key not configured")

        result = self._get(
            module="account", action="txlist",
            address=address, page=1, offset=offset, sort="desc"
        )
        if not isinstance(result, list):
            return []

        whale_txs = []
        for tx in result:
            value_eth = float(tx.get("value", 0)) / 1e18
            if value_eth >= min_value_eth:
                whale_txs.append({
                    "hash": tx.get("hash"),
                    "from": tx.get("from"),
                    "to": tx.get("to"),
                    "value_eth": round(value_eth, 4),
                    "timestamp": datetime.utcfromtimestamp(
                        int(tx.get("timeStamp", 0))
                    ).isoformat() if tx.get("timeStamp") else None,
                    "direction": "out" if tx.get("from", "").lower() == address.lower() else "in",
                })
        return whale_txs


# ---------------------------------------------------------------------------
# Bitcoin adapter (blockchain.info — free, no key)
# ---------------------------------------------------------------------------
class BitcoinAdapter:
    """
    Bitcoin network stats via blockchain.info public API.
    No API key needed.
    """

    BASE_URL = "https://blockchain.info"

    def __init__(self):
        self.available = True

    def _get(self, endpoint: str) -> Any:
        import requests
        resp = requests.get(f"{self.BASE_URL}/{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_stats(self) -> Dict:
        """Bitcoin network statistics."""
        data = self._get("stats?format=json")
        return {
            "market_price_usd": data.get("market_price_usd"),
            "hash_rate_gh": data.get("hash_rate", 0) / 1e9,  # Convert to GH/s
            "total_fees_btc": data.get("total_fees_btc", 0) / 1e8,
            "n_btc_mined": data.get("n_btc_mined", 0) / 1e8,
            "n_tx": data.get("n_tx"),
            "n_blocks_mined": data.get("n_blocks_mined"),
            "minutes_between_blocks": data.get("minutes_between_blocks"),
            "difficulty": data.get("difficulty"),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_mempool_info(self) -> Dict:
        """Unconfirmed transaction count (mempool pressure indicator)."""
        import requests
        resp = requests.get(f"{self.BASE_URL}/q/unconfirmedcount", timeout=10)
        resp.raise_for_status()
        return {
            "unconfirmed_txs": int(resp.text),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_block_height(self) -> int:
        """Current block height."""
        import requests
        resp = requests.get(f"{self.BASE_URL}/q/getblockcount", timeout=10)
        resp.raise_for_status()
        return int(resp.text)


# ---------------------------------------------------------------------------
# Unified OnChainDataService
# ---------------------------------------------------------------------------
class OnChainDataService:
    """
    Unified entry point for on-chain blockchain data.
    All sources are free-tier.

    Usage:
        svc = OnChainDataService()

        # Ethereum
        gas = svc.etherscan.get_gas_oracle()
        price = svc.etherscan.get_eth_price()
        balance = svc.etherscan.get_balance("0x...")
        transfers = svc.etherscan.get_token_transfers("0x...")

        # Bitcoin
        stats = svc.bitcoin.get_stats()
        mempool = svc.bitcoin.get_mempool_info()
    """

    def __init__(self):
        self.etherscan = EtherscanAdapter()
        self.bitcoin = BitcoinAdapter()

        logger.info(
            f"OnChainDataService initialized — "
            f"etherscan={'OK' if self.etherscan.available else 'NO KEY'}, "
            f"bitcoin=OK"
        )

    def get_status(self) -> Dict:
        """Health check for all on-chain adapters."""
        return {
            "etherscan": {"available": self.etherscan.available, "source": "Etherscan"},
            "bitcoin": {"available": self.bitcoin.available, "source": "blockchain.info"},
        }

    def get_network_overview(self) -> Dict:
        """Combined overview of major blockchain networks."""
        overview = {"timestamp": datetime.utcnow().isoformat(), "networks": {}}

        # Bitcoin
        try:
            overview["networks"]["bitcoin"] = self.bitcoin.get_stats()
        except Exception as e:
            logger.warning(f"Bitcoin stats failed: {e}")
            overview["networks"]["bitcoin"] = {"error": str(e)}

        # Ethereum
        if self.etherscan.available:
            try:
                eth_data = {}
                eth_data["price"] = self.etherscan.get_eth_price()
                eth_data["gas"] = self.etherscan.get_gas_oracle()
                overview["networks"]["ethereum"] = eth_data
            except Exception as e:
                logger.warning(f"Ethereum stats failed: {e}")
                overview["networks"]["ethereum"] = {"error": str(e)}

        return overview
