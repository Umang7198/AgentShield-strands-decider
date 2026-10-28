"""ShopHelp: Hearth & Cable support agent (primary demo scenario)."""

from agentshield.shophelp.agent import SHOPHELP_FACTS, ShopHelpApp, build_shophelp
from agentshield.shophelp.sql_store import SqlShopHelpStore
from agentshield.shophelp.store import (
    Customer,
    Order,
    ShopHelpCatalogue,
    ShopHelpError,
    ShopHelpStore,
    Ticket,
)
from agentshield.shophelp.tools import bind_tools
from agentshield.shophelp.world import (
    AGENT_NAME,
    STORE_NAME,
    SUPERVISOR_NAME,
    company,
)

__all__ = [
    "AGENT_NAME",
    "SHOPHELP_FACTS",
    "STORE_NAME",
    "SUPERVISOR_NAME",
    "Customer",
    "Order",
    "ShopHelpApp",
    "ShopHelpCatalogue",
    "ShopHelpError",
    "ShopHelpStore",
    "SqlShopHelpStore",
    "Ticket",
    "bind_tools",
    "build_shophelp",
    "company",
]
