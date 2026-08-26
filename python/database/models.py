from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Float, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from database.base import Base


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    brand: Mapped[str] = mapped_column(String(128), default="")
    seller_id: Mapped[str] = mapped_column(String(64), default="")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    image_url: Mapped[str] = mapped_column(Text, default="")


class Warehouse(Base):
    __tablename__ = "warehouses"

    warehouse_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    region: Mapped[str] = mapped_column(String(64), nullable=False)
    city: Mapped[str] = mapped_column(String(64), nullable=False)
    carriers: Mapped[list] = mapped_column(JSON, default=list)


class WarehouseStock(Base):
    __tablename__ = "warehouse_stock"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("products.product_id"), nullable=False, index=True
    )
    warehouse_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("warehouses.warehouse_id"), nullable=False, index=True
    )
    physical_stock: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Reservation(Base):
    __tablename__ = "reservations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    reservation_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    product_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    warehouse_id: Mapped[str] = mapped_column(String(32), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")


class LogisticsRoute(Base):
    __tablename__ = "logistics_routes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    carrier: Mapped[str] = mapped_column(String(32), nullable=False)
    warehouse_id: Mapped[str] = mapped_column(String(32), nullable=False)
    destination: Mapped[str] = mapped_column(String(64), nullable=False)
    insured: Mapped[bool] = mapped_column(Boolean, default=False)
    insured_amount: Mapped[float] = mapped_column(Float, default=0.0)
    encrypted: Mapped[bool] = mapped_column(Boolean, default=False)
    eta_hours: Mapped[int] = mapped_column(Integer, default=48)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    product_id: Mapped[str] = mapped_column(String(64), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    reservation_id: Mapped[str] = mapped_column(String(32), nullable=False)
    logistics_route_id: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="created")
    total_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
