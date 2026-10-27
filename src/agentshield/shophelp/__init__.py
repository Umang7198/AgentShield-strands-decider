"""ShopHelp: the primary demo scenario (online-store support agent with mocked tools)."""

from agentshield.shophelp.agent import SHOPHELP_FACTS, ShopHelpApp, build_shophelp
from agentshield.shophelp.store import Customer, Order, ShopHelpError, ShopHelpStore
from agentshield.shophelp.tools import bind_tools

__all__ = [
    "SHOPHELP_FACTS",
    "Customer",
    "Order",
    "ShopHelpApp",
    "ShopHelpError",
    "ShopHelpStore",
    "bind_tools",
    "build_shophelp",
]
