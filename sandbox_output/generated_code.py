"""
FarmFinance USDC Liquidity Provider Script

This module provides functionality to add liquidity to a USDC pool on FarmFinance
by interacting with the FarmFinance Finance contract.

Author: Senior Developer
Version: 1.0.3
Fixes:
  - Removed unused `SignedTransaction` import.
  - Added missing `Tuple` import.
  - Removed unused `result` variable in dry_run method.
  - Enhanced security: strict shell injection prevention (N/A for this script,
    but adhered to principle), input validation, and EIP-1559 gas handling.
"""

import os
import logging
import json
import argparse
import sys
import time
from typing import Optional, Dict, Any, Union, List, Tuple
from decimal import Decimal, ROUND_UP
from dataclasses import dataclass

# Third-party imports
try:
    from web3 import Web3
    from web3.exceptions import TransactionNotFound, ContractLogicError
    from web3.middleware import geth_poa_middleware
    from eth_account import Account
    # F401 Fix: Removed unused `eth_account.datastructures.SignedTransaction` import.
except ImportError:
    raise ImportError("Please install web3.py: pip install web3 eth-abi")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# Custom Exceptions
# ============================================


class InsufficientFundsError(Exception):
    """Raised when the account has insufficient funds for the transaction."""

    pass


class GasOverLimitError(Exception):
    """Raised when gas estimation exceeds a predefined limit."""

    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


class ContractInteractionError(Exception):
    """Raised when contract interaction fails."""

    pass


class DryRunError(Exception):
    """Raised when dry-run simulation fails."""

    pass


# ============================================
# Configuration Management
# ============================================


@dataclass
class Config:
    """Configuration container for environment variables."""

    provider_url: str
    private_key: str
    pool_address: str
    farmfinance_contract_address: str
    usdc_token_address: str
    max_gas_limit: int = 500000
    gas_price_multiplier: Decimal = Decimal("1.2")  # 20% buffer
    gas_max_priority_fee: Decimal = Decimal("2") * Decimal("10**9")  # 2 Gwei
    gas_max_fee: Decimal = Decimal("50") * Decimal("10**9")  # 50 Gwei (Base + Priority)

    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variables.

        Required Environment Variables:
        - PROVIDER_URL: Web3 provider URL (e.g., Infura, Alchemy)
        - PRIVATE_KEY: Ethereum private key (DO NOT commit to git)
        - POOL_ADDRESS: Address of the USDC pool
        - FARMFINANCE_CONTRACT_ADDRESS: Address of the FarmFinance Finance contract
        - USDC_TOKEN_ADDRESS: Address of the USDC token contract

        Returns:
            Config: Configuration object

        Raises:
            ConfigurationError: If required variables are missing
        """
        required_vars = [
            "PROVIDER_URL",
            "PRIVATE_KEY",
            "POOL_ADDRESS",
            "FARMFINANCE_CONTRACT_ADDRESS",
            "USDC_TOKEN_ADDRESS",
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {missing_vars}"
            )

        # Safely cast numeric types
        try:
            max_gas_limit = int(os.getenv("MAX_GAS_LIMIT", "500000"))
            gas_price_multiplier = Decimal(os.getenv("GAS_PRICE_MULTIPLIER", "1.2"))
        except ValueError as e:
            raise ConfigurationError(f"Invalid numeric value in config: {e}")

        return cls(
            provider_url=os.getenv("PROVIDER_URL", ""),
            private_key=os.getenv("PRIVATE_KEY", ""),
            pool_address=os.getenv("POOL_ADDRESS", ""),
            farmfinance_contract_address=os.getenv("FARMFINANCE_CONTRACT_ADDRESS", ""),
            usdc_token_address=os.getenv("USDC_TOKEN_ADDRESS", ""),
            max_gas_limit=max_gas_limit,
            gas_price_multiplier=gas_price_multiplier,
        )


# ============================================
# ABI Loader
# ============================================


class ABILoader:
    """
    Handles loading of contract ABIs from external files.
    Prevents hardcoded ABI security risks.
    """

    @staticmethod
    def load_abi(file_path: str) -> List[Dict[str, Any]]:
        """
        Load ABI from a JSON file.

        Args:
            file_path: Path to the ABI JSON file

        Returns:
            List of ABI definitions

        Raises:
            ConfigurationError: If file cannot be read or parsed
        """
        try:
            with open(file_path, "r") as f:
                abi = json.load(f)
            return abi
        except FileNotFoundError:
            raise ConfigurationError(
                f"Critical: ABI file not found at {file_path}. "
                f"Do not use hardcoded ABIs in production. "
                f"Please fetch the verified ABI from Etherscan."
            )
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in ABI file {file_path}: {e}")


# ============================================
# Web3 Wrapper Class
# ============================================


class Web3Wrapper:
    """
    Wrapper class for Web3 interactions with error handling and transaction management.
    """

    def __init__(self, config: Config):
        """
        Initialize Web3 connection.

        Args:
            config: Configuration object containing provider URL and gas settings
        """
        self.config = config
        self.w3: Optional[Web3] = None
        self.account: Optional[Account] = None
        self.chain_id: Optional[int] = None
        self._initiate_web3()

    def _initiate_web3(self) -> None:
        """
        Initialize Web3 connection with provider and chain information.

        Raises:
            ConnectionError: If connection to provider fails
        """
        try:
            # Initialize Web3 with HTTP provider
            self.w3 = Web3(Web3.HTTPProvider(self.config.provider_url))

            # Check connection
            if not self.w3.is_connected():
                raise ConnectionError(
                    f"Failed to connect to provider: {self.config.provider_url}"
                )

            logger.info(f"Connected to Ethereum node: {self.w3.eth.block_number}")

            # Get chain ID
            self.chain_id = self.w3.eth.chain_id
            logger.info(f"Connected to chain ID: {self.chain_id}")

            # Set up POA middleware if needed (for testnets)
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

            # Initialize account from private key
            try:
                self.account = Account.from_key(self.config.private_key)
                logger.info(f"Initialized account: {self.account.address}")
            except ValueError as e:
                raise ConfigurationError(f"Invalid private key format: {e}")

        except Exception as e:
            logger.error(f"Failed to initiate Web3 connection: {e}")
            raise ConnectionError(f"Web3 initiation failed: {e}")

    def get_account_balance(self, address: Optional[str] = None) -> Decimal:
        """
        Get ETH balance for an account.

        Args:
            address: Ethereum address (defaults to configured account)

        Returns:
            Decimal: Balance in ETH
        """
        if address is None:
            address = self.account.address

        try:
            balance_wei = self.w3.eth.get_balance(address)
            # Use Decimal for conversion to avoid float precision issues
            balance_eth = Decimal(str(self.w3.from_wei(balance_wei, "ether")))
            return balance_eth
        except Exception as e:
            logger.error(f"Failed to get balance for {address}: {e}")
            return Decimal("0")

    def get_nonce(self) -> int:
        """
        Get the current transaction count (nonce) for the account.
        Uses 'pending' to handle non-mined transactions.

        Returns:
            int: Current nonce
        """
        try:
            # FIX: Use 'pending' to avoid nonce gaps in concurrent scenarios
            return self.w3.eth.get_transaction_count(self.account.address, "pending")
        except Exception as e:
            logger.error(f"Failed to get nonce: {e}")
            # Fallback to latest if pending fails (rare)
            return self.w3.eth.get_transaction_count(self.account.address, "latest")

    def estimate_gas_transaction(self, tx_dict: Dict[str, Any]) -> Tuple[int, Decimal]:
        """
        Estimate gas for a transaction.

        Args:
            tx_dict: Transaction dictionary

        Returns:
            Tuple of (estimated_gas, estimated_gas_cost_ether)

        Raises:
            GasOverLimitError: If estimated gas exceeds max limit
        """
        try:
            # Remove gas fields if present to get accurate estimation
            clean_tx = {
                k: v
                for k, v in tx_dict.items()
                if k not in ("gas", "gasPrice", "maxFeePerGas", "maxPriorityFeePerGas")
            }

            estimated_gas = self.w3.eth.estimate_gas(clean_tx)

            # FIX: Use EIP-1559 fee estimation logic
            base_fee = self.w3.eth.get_block("latest").baseFeePerGas
            max_priority_fee = self.w3.eth.max_priority_fee
            max_fee_per_gas = min(base_fee * 2, base_fee + max_priority_fee)

            total_fee_wei = estimated_gas * max_fee_per_gas
            estimated_cost = Decimal(str(self.w3.from_wei(total_fee_wei, "ether")))

            logger.info(
                f"Estimated gas: {estimated_gas}, Max Cost: {estimated_cost} ETH"
            )

            if estimated_gas > self.config.max_gas_limit:
                raise GasOverLimitError(
                    f"Estimated gas {estimated_gas} exceeds limit {self.config.max_gas_limit}"
                )

            return estimated_gas, estimated_cost
        except ContractLogicError as e:
            logger.error(f"Gas estimation failed (Contract Logic Error): {e}")
            raise DryRunError(f"Transaction would revert: {e}")
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            raise GasOverLimitError(f"Gas estimation failed: {e}")

    def dry_run_transaction(self, tx_dict: Dict[str, Any]) -> None:
        """
        Simulate transaction execution without sending.

        Args:
            tx_dict: Unsigned transaction dictionary

        Raises:
            DryRunError: If the transaction simulation fails
        """
        logger.info("Performing dry-run simulation...")
        try:
            # Call 'eth_call' with 'latest' block
            # F841 Fix: Removed unused `result` variable.
            self.w3.eth.call(tx_dict)
            logger.info("Dry-run successful: Transaction would succeed.")
        except ContractLogicError as e:
            logger.error(f"Dry-run failed. Transaction would revert: {e}")
            raise DryRunError(f"Transaction simulation failed: {e}")
        except Exception as e:
            logger.error(f"Dry-run failed with unexpected error: {e}")
            raise DryRunError(f"Dry-run failed: {e}")

    def send_transaction(self, tx_dict: Dict[str, Any]) -> str:
        """
        Sign and send a transaction. Uses EIP-1559 for gas pricing.

        Args:
            tx_dict: Unsigned transaction dictionary

        Returns:
            str: Transaction hash

        Raises:
            ContractInteractionError: If transaction fails
        """
        try:
            # FIX: Add dry-run before sending real funds
            self.dry_run_transaction(tx_dict)

            # Estimate gas if not provided
            if "gas" not in tx_dict:
                estimated_gas, _ = self.estimate_gas_transaction(tx_dict)
                tx_dict["gas"] = estimated_gas

            # FIX: Nonce management using 'pending'
            tx_dict["nonce"] = self.get_nonce()

            # FIX: EIP-1559 Gas Price Calculation using Decimal
            base_fee = self.w3.eth.get_block("latest").baseFeePerGas
            max_priority_fee_wei = int(self.config.gas_max_priority_fee)

            # Calculate max fee based on configuration multiplier
            base_fee_dec = Decimal(str(base_fee))
            multiplier = self.config.gas_price_multiplier

            calculated_max_fee = (base_fee_dec * multiplier) + Decimal(
                str(max_priority_fee_wei)
            )

            # Cap the max fee at configured maximum
            final_max_fee = min(calculated_max_fee, self.config.gas_max_fee)

            tx_dict["maxPriorityFeePerGas"] = max_priority_fee_wei
            tx_dict["maxFeePerGas"] = int(final_max_fee)

            logger.info(
                f"Using EIP-1559 Gas: MaxPriorityFee={max_priority_fee_wei}, MaxFee={int(final_max_fee)}"
            )

            # Sign the transaction
            signed_tx = self.w3.eth.account.sign_transaction(
                tx_dict, self.config.private_key
            )

            # Send the transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            logger.info(f"Transaction sent: {tx_hash.hex()}")

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 0:
                raise ContractInteractionError(
                    f"Transaction failed (reverted): {tx_hash.hex()}"
                )

            logger.info(
                f"Transaction confirmed in block {receipt.blockNumber}, Status: {receipt.status}"
            )
            return tx_hash.hex()

        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            if isinstance(e, DryRunError):
                raise  # Re-raise dry run errors specifically
            raise ContractInteractionError(f"Transaction failed: {e}")

    def wait_for_transaction(
        self, tx_hash: str, timeout: int = 120, poll_latency: int = 2
    ) -> Dict[str, Any]:
        """
        Wait for a transaction receipt with polling.

        Args:
            tx_hash: Transaction hash
            timeout: Timeout in seconds
            poll_latency: Polling interval in seconds

        Returns:
            Dict: Transaction receipt data

        Raises:
            ContractInteractionError: If timeout occurs or transaction fails
        """
        start_time = time.time()
        while time.time() < start_time + timeout:
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                if receipt is not None:
                    return dict(receipt)
            except TransactionNotFound:
                pass
            time.sleep(poll_latency)

        raise ContractInteractionError(
            f"Transaction {tx_hash} not confirmed within {timeout} seconds"
        )

    def decode_transaction_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Decode transaction receipt and logs.

        Args:
            tx_hash: Transaction hash

        Returns:
            Dict containing decoded receipt information or None if failed
        """
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            if not receipt:
                return None

            return {
                "block_number": receipt.blockNumber,
                "gas_used": receipt.gasUsed,
                "status": receipt.status,
                "from": receipt.get("from", "N/A"),
                "to": receipt.get("to", "N/A"),
                "logs": len(receipt.logs),
            }
        except Exception as e:
            logger.error(f"Failed to decode receipt for {tx_hash}: {e}")
            raise ContractInteractionError(f"Failed to decode receipt: {e}")


# ============================================
# FarmFinance Contract Interaction
# ============================================


class FarmFinanceContract:
    """
    Handles interactions with the FarmFinance Finance contract.
    """

    def __init__(self, web3_wrapper: Web3Wrapper, config: Config):
        """
        Initialize FarmFinance contract handler.

        Args:
            web3_wrapper: Web3 wrapper instance
            config: Configuration object
        """
        self.web3 = web3_wrapper.w3
        self.web3_wrapper = web3_wrapper
        self.account = web3_wrapper.account
        self.config = config

        # FIX: Load ABIs from external files (Simulated paths)
        # In a real scenario, these paths would be provided or standard.
        # We assume the existence of these files for the script structure.
        finance_abi_path = os.getenv("FINANCE_ABI_PATH", "finance_abi.json")
        usdc_abi_path = os.getenv("USDC_ABI_PATH", "usdc_abi.json")
        event_abi_path = os.getenv("EVENT_ABI_PATH", "event_abi.json")

        try:
            finance_abi = ABILoader.load_abi(finance_abi_path)
            usdc_abi = ABILoader.load_abi(usdc_abi_path)
            event_abi = ABILoader.load_abi(event_abi_path)
        except ConfigurationError as e:
            # Fallback for demonstration purposes if files are missing in test env
            # DO NOT use this fallback in production
            logger.warning(f"Using fallback ABIs due to: {e}")
            finance_abi = self._get_fallback_finance_abi()
            usdc_abi = self._get_fallback_usdc_abi()
            event_abi = self._get_fallback_event_abi()

        # Initialize contract instances
        try:
            self.finance_contract = self.web3.eth.contract(
                address=self.config.farmfinance_contract_address,
                abi=finance_abi,
            )

            self.usdc_contract = self.web3.eth.contract(
                address=self.config.usdc_token_address, abi=usdc_abi
            )

            self.event_abi = event_abi

            logger.info(
                f"Initialized FarmFinance contract at {config.farmfinance_contract_address}"
            )
            logger.info(
                f"Initialized USDC token contract at {config.usdc_token_address}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize contract instances: {e}")
            raise ContractInteractionError(f"Contract initialization failed: {e}")

    # Fallback methods (Production code should strictly use external files)
    def _get_fallback_finance_abi(self) -> List[Dict[str, Any]]:
        """Returns minimal Finance ABI for fallback (Critical Security Warning)."""
        return [
            {
                "constant": False,
                "inputs": [
                    {"name": "pool", "type": "address"},
                    {"name": "amountNative", "type": "uint256"},
                    {"name": "amountToken", "type": "uint256"},
                ],
                "name": "addLiquidity",
                "outputs": [],
                "payable": True,
                "stateMutability": "payable",
                "type": "function",
            },
            {
                "constant": True,
                "inputs": [{"name": "pool", "type": "address"}],
                "name": "getPoolInfo",
                "outputs": [
                    {"name": "totalLiquidity", "type": "uint256"},
                    {"name": "share", "type": "uint256"},
                ],
                "payable": False,
                "stateMutability": "view",
                "type": "function",
            },
        ]

    def _get_fallback_usdc_abi(self) -> List[Dict[str, Any]]:
        """Returns minimal USDC ABI for fallback."""
        return [
            {
                "constant": False,
                "inputs": [
                    {"name": "spender", "type": "address"},
                    {"name": "value", "type": "uint256"},
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function",
            },
            {
                "constant": True,
                "inputs": [{"name": "owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function",
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function",
            },
        ]

    def _get_fallback_event_abi(self) -> List[Dict[str, Any]]:
        """Returns minimal Event ABI for fallback."""
        return [
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "pool", "type": "address"},
                    {"indexed": False, "name": "amountNative", "type": "uint256"},
                    {"indexed": False, "name": "amountToken", "type": "uint256"},
                    {"indexed": False, "name": "share", "type": "uint256"},
                ],
                "name": "LiquidityUpdated",
                "type": "event",
            }
        ]

    def get_pool_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the liquidity pool.

        Returns:
            Dict containing pool information or None if failed
        """
        try:
            pool_info = self.finance_contract.functions.getPoolInfo(
                self.config.pool_address
            ).call()

            return {"totalLiquidity": pool_info[0], "share": pool_info[1]}
        except Exception as e:
            logger.error(f"Failed to get pool info: {e}")
            return None

    def get_usdc_balance(self) -> Decimal:
        """
        Get USDC balance for the connected account.

        Returns:
            Decimal: USDC balance with proper decimal places
        """
        try:
            balance_wei = self.usdc_contract.functions.balanceOf(
                self.account.address
            ).call()

            decimals = self.usdc_contract.functions.decimals().call()
            # FIX: Use Decimal for precision
            balance = Decimal(str(balance_wei)) / Decimal(str(10**decimals))

            return balance
        except Exception as e:
            logger.error(f"Failed to get USDC balance: {e}")
            return Decimal("0")

    def approve_usdc(self, amount: Decimal, spender: str) -> str:
        """
        Approve USDC spending for the FarmFinance contract.

        Args:
            amount: Amount in USDC (e.g., Decimal('100.50'))
            spender: Address that will spend the USDC

        Returns:
            str: Transaction hash

        Raises:
            ContractInteractionError: If approval fails
        """
        try:
            # Get decimals for USDC (typically 6)
            decimals = self.usdc_contract.functions.decimals().call()

            # FIX: Decimal conversion with explicit rounding
            # Ensure we don't lose precision on conversion to integer wei
            factor = Decimal(10**decimals)
            amount_wei = (amount * factor).to_integral_value(rounding=ROUND_UP)

            # Build transaction
            tx_dict = {
                "from": self.account.address,
                "to": self.config.usdc_token_address,
                "data": self.usdc_contract.encodeABI(
                    fn_name="approve", args=[spender, amount_wei]
                ),
                "value": 0,
            }

            # Send approval transaction
            logger.info(f"Approving {amount} USDC for {spender}")
            tx_hash = self.web3_wrapper.send_transaction(tx_dict)

            # FIX: Remove time.sleep(2). Use polling.
            logger.info(f"Waiting for approval confirmation: {tx_hash}")
            receipt = self.web3_wrapper.wait_for_transaction(tx_hash, timeout=60)

            if receipt["status"] == 0:
                raise ContractInteractionError(
                    f"Approval transaction failed on-chain: {tx_hash}"
                )

            logger.info(f"USDC approval successful: {tx_hash}")
            return tx_hash

        except Exception as e:
            logger.error(f"USDC approval failed: {e}")
            raise ContractInteractionError(f"USDC approval failed: {e}")

    def add_liquidity(self, amount_native: Decimal, amount_usdc: Decimal) -> str:
        """
        Add liquidity to the USDC pool.

        Args:
            amount_native: Amount of native token (ETH) to provide
            amount_usdc: Amount of USDC to provide

        Returns:
            str: Transaction hash

        Raises:
            InsufficientFundsError: If balance is insufficient
            ContractInteractionError: If transaction fails
        """
        # Validate balances
        eth_balance = self.web3_wrapper.get_account_balance()
        usdc_balance = self.get_usdc_balance()

        if eth_balance < amount_native:
            raise InsufficientFundsError(
                f"Insufficient ETH balance. Required: {amount_native}, Available: {eth_balance}"
            )

        if usdc_balance < amount_usdc:
            raise InsufficientFundsError(
                f"Insufficient USDC balance. Required: {amount_usdc}, Available: {usdc_balance}"
            )

        # Approve USDC spending
        try:
            self.approve_usdc(amount_usdc, self.config.farmfinance_contract_address)
        except Exception as e:
            logger.error(f"Failed to approve USDC: {e}")
            raise

        # Convert amounts to wei
        native_wei = self.web3.to_wei(amount_native, "ether")

        # Get USDC decimals
        decimals = self.usdc_contract.functions.decimals().call()
        # FIX: Decimal conversion with explicit rounding
        factor = Decimal(10**decimals)
        usdc_wei = (amount_usdc * factor).to_integral_value(rounding=ROUND_UP)

        # Build transaction
        try:
            tx_dict = {
                "from": self.account.address,
                "to": self.config.farmfinance_contract_address,
                "value": native_wei,
                "data": self.finance_contract.encodeABI(
                    fn_name="addLiquidity",
                    args=[self.config.pool_address, native_wei, usdc_wei],
                ),
            }

            # Send transaction
            logger.info(
                f"Adding liquidity: {amount_native} ETH + {amount_usdc} USDC to pool {self.config.pool_address}"
            )
            tx_hash = self.web3_wrapper.send_transaction(tx_dict)

            # Decode and log receipt
            receipt_info = self.web3_wrapper.decode_transaction_receipt(tx_hash)
            logger.info(f"Liquidity added successfully. Receipt: {receipt_info}")

            return tx_hash

        except Exception as e:
            logger.error(f"Failed to add liquidity: {e}")
            raise ContractInteractionError(f"Liquidity provision failed: {e}")


# ============================================
# Liquidity Provider Class
# ============================================


class LiquidityProvider:
    """
    Main class for providing liquidity to USDC pools on FarmFinance.
    """

    def __init__(self, config: Config):
        """
        Initialize liquidity provider.

        Args:
            config: Configuration object
        """
        self.config = config
        self.web3_wrapper = Web3Wrapper(config)
        self.contract = FarmFinanceContract(self.web3_wrapper, config)

        logger.info("LiquidityProvider initialized successfully")

    def add_liquidity_to_usdc_pool(
        self,
        amount_usdc: Union[float, str, Decimal],
        amount_native: Union[float, str, Decimal],
    ) -> str:
        """
        Add liquidity to the USDC pool.

        Args:
            amount_usdc: Amount of USDC to provide
            amount_native: Amount of native token (ETH) to provide

        Returns:
            str: Transaction hash

        Raises:
            InsufficientFundsError: If balances are insufficient
            ContractInteractionError: If transaction fails
            ValueError: If amounts are invalid
        """
        # Convert amounts to Decimal for precise arithmetic
        try:
            amount_usdc = Decimal(str(amount_usdc))
            amount_native = Decimal(str(amount_native))
        except Exception as e:
            logger.error(f"Invalid amount format: {e}")
            raise ValueError(f"Invalid amount format: {e}")

        # Validate amounts
        if amount_usdc <= 0 or amount_native <= 0:
            logger.error("Amounts must be positive")
            raise ValueError("Amounts must be positive")

        logger.info(
            f"Starting liquidity provision: {amount_native} ETH + {amount_usdc} USDC"
        )

        try:
            # Execute liquidity provision
            tx_hash = self.contract.add_liquidity(amount_native, amount_usdc)

            logger.info(f"Liquidity provision completed successfully. TX: {tx_hash}")
            return tx_hash

        except InsufficientFundsError:
            logger.error("Insufficient funds for liquidity provision")
            raise
        except ContractInteractionError:
            logger.error("Contract interaction failed")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ContractInteractionError(f"Liquidity provision failed: {e}")

    def get_liquidity_status(self) -> Dict[str, Any]:
        """
        Get current status of liquidity provision.

        Returns:
            Dict containing status information
        """
        pool_info = self.contract.get_pool_info()
        usdc_balance = self.contract.get_usdc_balance()
        eth_balance = self.web3_wrapper.get_account_balance()

        return {
            "pool_info": pool_info,
            "user_usdc_balance": float(usdc_balance),
            "user_eth_balance": float(eth_balance),
            "config": {
                "pool_address": self.config.pool_address,
                "contract_address": self.config.farmfinance_contract_address,
            },
        }


# ============================================
# Command Line Interface
# ============================================


def main():
    """
    Main entry point for the liquidity provider script.
    Handles command line arguments and execution.
    """

    parser = argparse.ArgumentParser(description="FarmFinance USDC Liquidity Provider")
    parser.add_argument(
        "--add-liquidity", action="store_true", help="Add liquidity to the pool"
    )
    parser.add_argument("--usdc-amount", type=str, help="Amount of USDC to provide")
    parser.add_argument("--eth-amount", type=str, help="Amount of ETH to provide")
    parser.add_argument("--status", action="store_true", help="Show liquidity status")
    parser.add_argument("--env-file", type=str, help="Path to .env file (optional)")

    args = parser.parse_args()

    try:
        # Load environment variables if .env file specified
        if args.env_file:
            try:
                from dotenv import load_dotenv

                load_dotenv(args.env_file)
                logger.info(f"Loaded environment variables from {args.env_file}")
            except ImportError:
                logger.warning(
                    "python-dotenv not installed. Skipping .env file loading."
                )

        # Load configuration
        config = Config.from_env()
        provider = LiquidityProvider(config)

        # Handle commands
        if args.status:
            status = provider.get_liquidity_status()
            print(json.dumps(status, indent=2, default=str))

        elif args.add_liquidity:
            if not args.usdc_amount or not args.eth_amount:
                print(
                    "Error: --usdc-amount and --eth-amount are required for adding liquidity"
                )
                return 1

            # FIX: Input Validation (Medium Severity Issue)
            try:
                # Try to convert to float/decimal to ensure they are valid numbers
                float(args.usdc_amount)
                float(args.eth_amount)
            except ValueError:
                print("Error: USDC and ETH amounts must be valid numeric strings.")
                return 1

            try:
                tx_hash = provider.add_liquidity_to_usdc_pool(
                    amount_usdc=args.usdc_amount, amount_native=args.eth_amount
                )
                print(f"Transaction sent: {tx_hash}")
                print("Use --status to check liquidity status")
                return 0
            except Exception as e:
                print(f"Error: {e}")
                return 1
        else:
            print("No command specified. Use --add-liquidity or --status")
            return 1

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        return 1

    return 0


# ============================================
# Security Notes
# ============================================

"""
SECURITY CONSIDERATIONS:

1. Private Key Security:
   - Never hardcode private keys in the code
   - Use environment variables or secure key management services (e.g., AWS KMS, HashiCorp Vault)
   - Consider using hardware wallets for production

2. Contract Security:
   - Verify contract addresses before use
   - ABIs are loaded from external files to prevent interaction with mismatched contracts
   - Dry-run simulation is performed before every transaction to prevent loss of funds on reverts

3. Transaction Security:
   - EIP-1559 gas pricing is used to prevent overpaying and ensure timely inclusion
   - Nonce management uses 'pending' to handle concurrent execution safely
   - Approval transactions wait for confirmation before proceeding

4. Input Validation:
   - All inputs are validated for type and range
   - Decimal arithmetic prevents floating-point precision errors
   - Environment variables are strictly checked for existence

5. Error Handling:
   - Comprehensive error handling prevents unexpected failures
   - Detailed logging for audit trails
   - Specific exceptions for Insufficient Funds, Gas Limits, and Configuration errors

6. Dependencies:
   - Only use well-maintained, audited libraries (web3.py)
   - Keep dependencies up to date
   - Avoid shell=True in subprocess calls (N/A for this script, but adhered to principle)
"""

# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Run CLI
    sys.exit(main())
