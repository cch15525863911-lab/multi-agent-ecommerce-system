"""
PostgreSQL database initialization — creates tables and seeds business master data.

Usage:
    python -m database.init_db            # create tables + seed
    python -m database.init_db --drop      # recreate (dev only)
"""
from __future__ import annotations

import argparse

import structlog
from sqlalchemy import select

from database.engine import get_session, init_db
from database.models import Warehouse, WarehouseStock, Product

logger = structlog.get_logger()

SEED_PRODUCTS = [
    {"product_id": "P001", "name": "戴尔 XPS 15 笔记本电脑", "category": "数码", "price": 8999.0, "brand": "Dell"},
    {"product_id": "P003", "name": "罗技 MX Master 3S 无线鼠标", "category": "数码", "price": 599.0, "brand": "Logitech"},
    {"product_id": "P013", "name": "苹果 AirPods Pro 2", "category": "数码", "price": 1899.0, "brand": "Apple"},
    {"product_id": "P014", "name": "华为 Mate 60 Pro", "category": "手机", "price": 6999.0, "brand": "Huawei"},
    {"product_id": "P015", "name": "小米 14 Ultra", "category": "手机", "price": 6499.0, "brand": "Xiaomi"},
]

SEED_WAREHOUSES = [
    {"warehouse_id": "WH-NORTH", "region": "华北", "city": "北京", "carriers": ["SF", "JD"]},
    {"warehouse_id": "WH-SOUTH", "region": "华南", "city": "深圳", "carriers": ["SF", "YTO"]},
    {"warehouse_id": "WH-EAST", "region": "华东", "city": "上海", "carriers": ["SF", "JD", "YTO"]},
]

SEED_STOCK = [
    ("P001", "WH-NORTH", 30),
    ("P001", "WH-SOUTH", 8),
    ("P001", "WH-EAST", 120),
    ("P003", "WH-NORTH", 200),
    ("P003", "WH-EAST", 60),
    ("P013", "WH-NORTH", 15),
    ("P014", "WH-EAST", 5),
    ("P015", "WH-SOUTH", 50),
]


def seed_data() -> None:
    """Insert master data if tables are empty."""
    with get_session() as session:
        existing = session.scalar(select(Warehouse).limit(1))
        if existing:
            logger.info("init_db.seed.skip", reason="data already exists")
            return

        for p in SEED_PRODUCTS:
            session.add(
                Product(
                    product_id=p["product_id"],
                    name=p["name"],
                    category=p["category"],
                    price=p["price"],
                    brand=p["brand"],
                )
            )

        for w in SEED_WAREHOUSES:
            session.add(
                Warehouse(
                    warehouse_id=w["warehouse_id"],
                    region=w["region"],
                    city=w["city"],
                    carriers=w["carriers"],
                )
            )

        for product_id, warehouse_id, qty in SEED_STOCK:
            session.add(
                WarehouseStock(
                    product_id=product_id,
                    warehouse_id=warehouse_id,
                    physical_stock=qty,
                )
            )

    logger.info(
        "init_db.seed.done",
        products=len(SEED_PRODUCTS),
        warehouses=len(SEED_WAREHOUSES),
        stock_records=len(SEED_STOCK),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL database")
    parser.add_argument("--drop", action="store_true", help="Drop and recreate tables (dev only)")
    args = parser.parse_args()

    logger.info("init_db.start", drop=args.drop)
    init_db(drop=args.drop)
    seed_data()
    logger.info("init_db.complete")


if __name__ == "__main__":
    main()
