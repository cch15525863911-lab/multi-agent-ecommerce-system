"""
Knowledge Graph Store — Neo4j-backed user/product graph.

Graph Schema (Nodes + Edges):

  Nodes (labels):
    :User       {user_id, age?, gender?, city?, registered_at?}
    :Product    {product_id, name, price, description?, brand?, seller_id?, stock?, tags[]}
    :Category   {category_id, name, parent_id?}
    :Brand      {brand_id, name, country?}
    :Seller     {seller_id, name, rating?}

  Relationships (types):
    (User)-[:VIEWED     {at: timestamp, duration_sec?}]->(Product)
    (User)-[:PURCHASED  {at: timestamp, amount, order_id?}]->(Product)
    (User)-[:FAVORITED  {at: timestamp}]->(Product)
    (Product)-[:BELONGS_TO]->(Category)
    (Product)-[:BRANDED_BY]->(Brand)
    (Product)-[:SOLD_BY]->(Seller)
    (Product)-[:RELATED_TO {rel_id, relation, weight}]->(Product)

Why Neo4j here (vs Redis KV):
  - Relationships are first-class citizens. We can ask: "Which categories
    has this user bought from most often?" with a single Cypher JOIN,
    instead of denormalising everything into Redis hashes.
  - Graph algorithms (Jaccard similarity, degree centrality, multi-hop
    traversal) are already queryable for recall and GraphRAG evidence.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from config import get_settings

logger = structlog.get_logger()

HOUR = 3600
DAY = 86400
WEEK = 7 * DAY

RELATION_TYPES = ("bought_together", "complementary", "substitute")


class KGStore:
    """Thin wrapper around the Neo4j Python driver.

    Node upserts use MERGE so they stay idempotent. Behavior edges are
    append-only event logs, so VIEWED/PURCHASED edges are created per event.
    """

    def __init__(self, driver: Any | None = None):
        settings = get_settings()
        self._driver = driver
        self._database = settings.neo4j_database
        self._mock_data: dict[str, Any] = {}  # fallback when Neo4j is unavailable
        self._connected = False
        self._settings = settings

    # -------- connection mgmt --------

    async def connect(self) -> bool:
        """Open the driver. Returns True if the real driver works, False
        if we fall back to the in-memory mock mode."""
        if self._driver is not None:
            self._connected = True
            return True
        try:
            from neo4j import AsyncGraphDatabase  # lazy import so tests work without neo4j installed

            self._driver = AsyncGraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(self._settings.neo4j_user, self._settings.neo4j_password_str),
            )
            await self._driver.verify_connectivity()
            try:
                await self.ensure_schema()
            except Exception as exc:  # pragma: no cover - Neo4j permission/schema edge cases
                logger.warning("kg_store.schema_setup_failed", err=str(exc))
            self._connected = True
            logger.info(
                "kg_store.connected", uri=self._settings.neo4j_uri, db=self._database
            )
            return True
        except Exception as exc:  # pragma: no cover - depends on live Neo4j
            logger.warning("kg_store.fallback_mock", reason=str(exc))
            self._connected = False
            return False

    async def close(self) -> None:
        if self._driver is not None and self._connected:
            try:
                await self._driver.close()
            finally:
                self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def ensure_schema(self) -> None:
        """Create uniqueness constraints and indexes when they do not exist."""
        if self._driver is None:
            return
        statements = [
            """
            CREATE CONSTRAINT user_id_unique IF NOT EXISTS
            FOR (u:User) REQUIRE u.user_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT product_id_unique IF NOT EXISTS
            FOR (p:Product) REQUIRE p.product_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT category_name_unique IF NOT EXISTS
            FOR (c:Category) REQUIRE c.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT brand_name_unique IF NOT EXISTS
            FOR (b:Brand) REQUIRE b.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT seller_id_unique IF NOT EXISTS
            FOR (s:Seller) REQUIRE s.seller_id IS UNIQUE
            """,
            """
            CREATE INDEX product_category_id_index IF NOT EXISTS
            FOR (p:Product) ON (p.product_id)
            """,
        ]
        async with self._driver.session(database=self._database) as session:
            for statement in statements:
                await session.run(statement)

    # -------- node upserts --------

    async def upsert_user(self, user_id: str, **attrs: Any) -> None:
        q = """
        MERGE (u:User {user_id: $user_id})
        SET u += $attrs
        """
        await self._run(q, user_id=user_id, attrs=attrs or {})

    async def upsert_product(
        self,
        product_id: str,
        name: str,
        price: float,
        category: str | None = None,
        brand: str | None = None,
        seller_id: str | None = None,
        **attrs: Any,
    ) -> None:
        """Create a product + optionally link to category/brand/seller nodes."""
        q = """
        MERGE (p:Product {product_id: $product_id})
        SET p.name = $name, p.price = $price, p += $attrs
        WITH p
        CALL (p, $category) {
            WITH p, $category AS cat
            WHERE cat IS NOT NULL
            MERGE (c:Category {name: cat})
            MERGE (p)-[:BELONGS_TO]->(c)
        }
        WITH p
        CALL (p, $brand) {
            WITH p, $brand AS br
            WHERE br IS NOT NULL
            MERGE (b:Brand {name: br})
            MERGE (p)-[:BRANDED_BY]->(b)
        }
        WITH p
        CALL (p, $seller_id) {
            WITH p, $seller_id AS sid
            WHERE sid IS NOT NULL
            MERGE (s:Seller {seller_id: sid})
            MERGE (p)-[:SOLD_BY]->(s)
        }
        """
        await self._run(
            q,
            product_id=product_id,
            name=name,
            price=price,
            category=category,
            brand=brand,
            seller_id=seller_id,
            attrs=attrs or {},
        )

    # -------- behaviour edges --------

    async def record_view(
        self,
        user_id: str,
        product_id: str,
        at: float | None = None,
        duration_sec: int = 0,
    ) -> None:
        q = """
        MERGE (u:User {user_id: $user_id})
        MERGE (p:Product {product_id: $product_id})
        CREATE (u)-[:VIEWED {at: $at, duration_sec: $dur}]->(p)
        """
        await self._run(
            q, user_id=user_id, product_id=product_id, at=at or time.time(), dur=duration_sec
        )

    async def record_purchase(
        self,
        user_id: str,
        product_id: str,
        amount: float,
        at: float | None = None,
        order_id: str | None = None,
    ) -> None:
        q = """
        MERGE (u:User {user_id: $user_id})
        MERGE (p:Product {product_id: $product_id})
        CREATE (u)-[:PURCHASED {at: $at, amount: $amount, order_id: $oid}]->(p)
        """
        await self._run(
            q,
            user_id=user_id,
            product_id=product_id,
            at=at or time.time(),
            amount=amount,
            oid=order_id,
        )

    async def record_favorite(
        self, user_id: str, product_id: str, at: float | None = None
    ) -> None:
        q = """
        MERGE (u:User {user_id: $user_id})
        MERGE (p:Product {product_id: $product_id})
        MERGE (u)-[:FAVORITED {at: $at}]->(p)
        """
        await self._run(q, user_id=user_id, product_id=product_id, at=at or time.time())

    async def record_behavior(
        self,
        user_id: str,
        product_id: str,
        action: str,
        at: float | None = None,
        amount: float = 0.0,
        duration_sec: int = 0,
        order_id: str | None = None,
    ) -> None:
        """Unified behavior write entry: view / purchase / favorite."""
        action = action.lower()
        if action == "view":
            await self.record_view(user_id, product_id, at=at, duration_sec=duration_sec)
        elif action == "purchase":
            await self.record_purchase(
                user_id, product_id, amount=amount, at=at, order_id=order_id
            )
        elif action == "favorite":
            await self.record_favorite(user_id, product_id, at=at)
        else:
            raise ValueError(f"Unsupported behavior action: {action}")

    # -------- product relations --------

    async def upsert_product_relation(
        self,
        product_a: str,
        product_b: str,
        relation: str = "bought_together",
        weight: float = 1.0,
        **attrs: Any,
    ) -> None:
        """Create or update a directed Product -> Product relationship."""
        if relation not in RELATION_TYPES:
            raise ValueError(f"Unsupported product relation: {relation}")
        rel_id = f"{product_a}|{relation}|{product_b}"
        q = """
        MERGE (a:Product {product_id: $a_id})
        MERGE (b:Product {product_id: $b_id})
        MERGE (a)-[r:RELATED_TO {rel_id: $rel_id}]->(b)
        SET r.relation = $relation, r.weight = $weight, r += $attrs
        """
        await self._run(
            q,
            a_id=product_a,
            b_id=product_b,
            rel_id=rel_id,
            relation=relation,
            weight=weight,
            attrs=attrs or {},
        )

    async def build_co_purchase_relations(
        self, min_support: int = 2
    ) -> int:
        """Mine co-purchase pairs from PURCHASED edges and store RELATED_TO."""
        if not self._connected:
            return 0
        q = """
        MATCH (u:User)-[:PURCHASED]->(a:Product)
        MATCH (u:User)-[:PURCHASED]->(b:Product)
        WHERE a.product_id < b.product_id
        WITH a,
             b,
             count(DISTINCT u) AS support,
             a.product_id + '|bought_together|' + b.product_id AS rel_id
        WHERE support >= $min_support
        MERGE (a)-[r:RELATED_TO {rel_id: rel_id}]->(b)
        SET r.relation = 'bought_together',
            r.weight = 1.0 - 1.0 / (support + 1.0),
            r.evidence_count = support,
            r.source = 'co_purchase'
        RETURN count(r) AS created
        """
        rows = await self._run(q, min_support=min_support)
        return rows[0]["created"] if rows else 0

    async def get_related_products(
        self,
        product_id: str,
        relations: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return products directly related to a seed product."""
        if not self._connected:
            return []
        q = """
        MATCH (p:Product {product_id: $pid})-[r:RELATED_TO]-(other:Product)
        WHERE $relations IS NULL OR r.relation IN $relations
        RETURN other.product_id AS product_id,
               other.name AS name,
               other.category AS category,
               other.price AS price,
               other.brand AS brand,
               other.seller_id AS seller_id,
               other.stock AS stock,
               other.tags AS tags,
               other.description AS description,
               other.image_url AS image_url,
               r.relation AS relation,
               r.weight AS weight,
               r.evidence_count AS evidence_count
        ORDER BY coalesce(r.weight, 0) DESC
        LIMIT $limit
        """
        rows = await self._run(
            q, pid=product_id, relations=relations, limit=limit
        )
        return rows

    async def get_multi_hop_candidates(
        self,
        user_id: str,
        max_hops: int = 2,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Traverse RELATED_TO from the user's products to new candidates."""
        if not self._connected:
            return []
        max_hops = max(1, int(max_hops))
        q = f"""
        MATCH (u:User {{user_id: $uid}})-[:VIEWED|PURCHASED|FAVORITED]->(seed:Product)
        MATCH path=(seed)-[:RELATED_TO*1..{max_hops}]-(candidate:Product)
        WHERE NOT EXISTS((u)-[:VIEWED|PURCHASED|FAVORITED]->(candidate))
        WITH candidate, path
        RETURN candidate.product_id AS product_id,
               candidate.name AS name,
               candidate.category AS category,
               candidate.price AS price,
               candidate.brand AS brand,
               candidate.seller_id AS seller_id,
               candidate.stock AS stock,
               candidate.tags AS tags,
               candidate.description AS description,
               candidate.image_url AS image_url,
               length(path) AS hops,
               reduce(
                   score = 1.0,
                   rel IN relationships(path) | score * coalesce(rel.weight, 1.0)
               ) AS graph_score
        ORDER BY graph_score DESC, hops ASC
        LIMIT $limit
        """
        return await self._run(q, uid=user_id, limit=limit)

    async def get_explanation_paths(
        self,
        user_id: str,
        target_product_id: str,
        max_hops: int = 2,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Return graph paths that explain why a target may interest a user."""
        if not self._connected:
            return []
        max_hops = max(1, int(max_hops))
        q = f"""
        MATCH (u:User {{user_id: $uid}})-[:VIEWED|PURCHASED|FAVORITED]->(seed:Product)
        MATCH path=(seed)-[:RELATED_TO*1..{max_hops}]-(target:Product {{product_id: $pid}})
        RETURN seed.product_id AS seed_product_id,
               seed.name AS seed_name,
               [rel IN relationships(path) | rel.relation] AS relations,
               [rel IN relationships(path) | rel.weight] AS weights,
               length(path) AS hops
        ORDER BY hops ASC
        LIMIT $limit
        """
        return await self._run(q, uid=user_id, pid=target_product_id, limit=limit)

    async def get_similar_users(
        self, user_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Jaccard-similar users based on shared interacted products."""
        if not self._connected:
            return []
        my_products = await self._get_user_product_ids(user_id)
        if not my_products:
            return []
        my_set = set(my_products)
        rows = await self._run(
            """
            MATCH (other:User)-[:VIEWED|PURCHASED|FAVORITED]->(p:Product)
            WHERE other.user_id <> $uid
            WITH other, collect(DISTINCT p.product_id) AS products
            RETURN other.user_id AS user_id, products
            LIMIT $limit
            """,
            uid=user_id,
            limit=limit * 5,
        )
        scores = []
        for row in rows:
            other_set = set(row.get("products") or [])
            shared = len(my_set & other_set)
            union = len(my_set | other_set)
            if not shared:
                continue
            scores.append(
                {
                    "user_id": row["user_id"],
                    "jaccard": round(shared / union, 4),
                    "shared_products": shared,
                }
            )
        scores.sort(key=lambda x: (-x["jaccard"], -x["shared_products"]))
        return scores[:limit]

    async def get_degree_central_products(
        self, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return products ranked by graph interaction degree (popularity)."""
        if not self._connected:
            return []
        q = """
        MATCH (p:Product)
        WITH p, size((p)<-[:VIEWED|PURCHASED|FAVORITED]-()) AS interaction_degree
        WHERE interaction_degree > 0
        RETURN p.product_id AS product_id,
               p.name AS name,
               p.category AS category,
               p.price AS price,
               p.brand AS brand,
               p.seller_id AS seller_id,
               p.stock AS stock,
               p.tags AS tags,
               p.description AS description,
               p.image_url AS image_url,
               interaction_degree AS graph_score
        ORDER BY interaction_degree DESC
        LIMIT $limit
        """
        return await self._run(q, limit=limit)

    async def _get_user_product_ids(self, user_id: str) -> list[str]:
        if not self._connected:
            return []
        rows = await self._run(
            """
            MATCH (u:User {user_id: $uid})-[:VIEWED|PURCHASED|FAVORITED]->(p:Product)
            RETURN collect(DISTINCT p.product_id) AS products
            """,
            uid=user_id,
        )
        return list(rows[0].get("products") or []) if rows else []

    # -------- feature queries (Cypher-powered profile) --------

    async def get_profile_features(self, user_id: str) -> dict[str, Any]:
        """All features needed to build a UserProfile, in one aggregated call."""
        if not self._connected:
            return self._mock_get_profile_features(user_id)

        q = """
        MATCH (u:User {user_id: $user_id})
        OPTIONAL MATCH (u)-[v:VIEWED]->(vp:Product)
        OPTIONAL MATCH (vp)-[:BELONGS_TO]->(vc:Category)
        WITH u, collect(DISTINCT {
            at: v.at,
            product_id: vp.product_id,
            price: vp.price,
            category: vc.name
        }) AS views
        OPTIONAL MATCH (u)-[p:PURCHASED]->(pp:Product)
        OPTIONAL MATCH (pp)-[:BELONGS_TO]->(pc:Category)
        WITH u, views, collect(DISTINCT {
            at: p.at,
            product_id: pp.product_id,
            price: pp.price,
            amount: p.amount,
            category: pc.name
        }) AS purchases
        OPTIONAL MATCH (u)-[f:FAVORITED]->(fp:Product)
        OPTIONAL MATCH (fp)-[:BELONGS_TO]->(fc:Category)
        RETURN views, purchases, collect(DISTINCT {
            at: f.at,
            product_id: fp.product_id,
            price: fp.price,
            category: fc.name
        }) AS favorites
        """
        result = await self._run(q, user_id=user_id)
        if not result:
            return self._empty_features(user_id)
        row = result[0]
        return self._build_features(
            user_id,
            row["views"] or [],
            row["purchases"] or [],
            row["favorites"] or [],
        )

    async def get_preferred_categories(
        self, user_id: str, limit: int = 5
    ) -> list[tuple[str, int]]:
        """[(category_name, interaction_count)] ordered by count desc."""
        if not self._connected:
            return [("手机", 12), ("耳机", 8), ("平板", 5)][:limit]
        q = """
        MATCH (u:User {user_id: $uid})-[r:VIEWED|PURCHASED|FAVORITED]->(p:Product)-[:BELONGS_TO]->(c:Category)
        RETURN c.name AS name, count(r) AS cnt
        ORDER BY cnt DESC
        LIMIT $limit
        """
        rows = await self._run(q, uid=user_id, limit=limit)
        return [(r["name"], r["cnt"]) for r in rows]

    async def get_price_range(self, user_id: str) -> tuple[float, float]:
        """(p10_price, p90_price) of interacted products."""
        if not self._connected:
            return (50.0, 3000.0)
        q = """
        MATCH (u:User {user_id: $uid})-[r:VIEWED|PURCHASED]->(p:Product)
        WITH coalesce(r.amount, p.price) AS price
        WHERE price IS NOT NULL
        WITH percentileCont(price, 0.1) AS low, percentileCont(price, 0.9) AS high
        RETURN coalesce(low, 0) AS low, coalesce(high, 10000) AS high
        """
        rows = await self._run(q, uid=user_id)
        r = rows[0] if rows else {"low": 0.0, "high": 10000.0}
        return (float(r["low"]), float(r["high"]))

    async def get_active_hours(self, user_id: str) -> list[int]:
        """Top 3 hours of day the user is active (0-23)."""
        if not self._connected:
            return [20, 21, 22]
        q = """
        MATCH (u:User {user_id: $uid})-[r:VIEWED|PURCHASED]->()
        WHERE r.at IS NOT NULL
        WITH r.at AS at
        WITH datetime({epochSeconds: toInteger(at)}).hour AS hour, count(*) AS cnt
        ORDER BY cnt DESC
        LIMIT 3
        RETURN hour
        """
        rows = await self._run(q, uid=user_id)
        hours = sorted(r["hour"] for r in rows)
        return hours or [20, 21, 22]

    # -------- internals --------

    async def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Run a Cypher query. Returns a list of record dicts.

        Falls back to in-memory mock when the driver isn't connected.
        """
        if self._connected and self._driver is not None:
            try:
                async with self._driver.session(database=self._database) as session:
                    res = await session.run(query, params)
                    return [rec.data() for rec in await res.list()]
            except Exception as exc:  # pragma: no cover - live only
                logger.error("kg_store.cypher_error", err=str(exc))
                # degrade to mock on transient errors to keep the system up
        return self._mock_run(query, **params)

    def _mock_run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        return []

    @staticmethod
    def _empty_features(user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "view_count_1h": 0,
            "view_count_24h": 0,
            "purchase_count_7d": 0,
            "recent_views": [],
            "recent_purchases": [],
            "preferred_categories": [],
            "rfm": {"recency": 0.0, "frequency": 0.0, "monetary": 0.0},
            "price_range": (0.0, 10000.0),
            "active_hours": [],
        }

    def _build_features(
        self,
        user_id: str,
        views: list[dict],
        purchases: list[dict],
        favorites: list[dict] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        views = [v for v in views if v.get("product_id")]
        purchases = [p for p in purchases if p.get("product_id")]
        favorites = [f for f in (favorites or []) if f.get("product_id")]
        views_sorted = sorted(views, key=lambda v: v.get("at", 0))
        purchases_sorted = sorted(purchases, key=lambda p: p.get("at", 0))
        favorites_sorted = sorted(favorites, key=lambda f: f.get("at", 0))

        def _count_recent(events: list[dict], window: int) -> int:
            cutoff = now - window
            return sum(1 for e in events if e.get("at", 0) >= cutoff)

        # preferred categories from purchases + views (weighted)
        cat_count: dict[str, int] = {}
        for p in purchases:
            c = p.get("category")
            if c:
                cat_count[c] = cat_count.get(c, 0) + 3  # purchases = strong signal
        for v in views_sorted:
            c = v.get("category")
            if c:
                cat_count[c] = cat_count.get(c, 0) + 1
        for f in favorites_sorted:
            c = f.get("category")
            if c:
                cat_count[c] = cat_count.get(c, 0) + 2  # favorites = strong signal
        preferred = [c for c, _ in sorted(cat_count.items(), key=lambda x: -x[1])]

        # price range (p10 / p90 of purchase amount, fallback to view price)
        prices = [
            p.get("amount") or p.get("price", 0)
            for p in purchases
            if p.get("amount") or p.get("price")
        ] or [
            v.get("price", 0) for v in views_sorted if v.get("price")
        ]
        price_range = self._percentile_range(prices) if prices else (0.0, 10000.0)

        # active hours
        hours_count: dict[int, int] = {}
        for e in views_sorted + purchases + favorites_sorted:
            at = e.get("at")
            if at:
                h = int((at % DAY) // HOUR)
                hours_count[h] = hours_count.get(h, 0) + 1
        active_hours = [h for h, _ in sorted(hours_count.items(), key=lambda x: -x[1])[:3]]

        # RFM
        rfm = self._compute_rfm(purchases_sorted)

        return {
            "user_id": user_id,
            "view_count_1h": _count_recent(views_sorted, HOUR),
            "view_count_24h": _count_recent(views_sorted, DAY),
            "purchase_count_7d": _count_recent(purchases_sorted, WEEK),
            "recent_views": [v["product_id"] for v in views_sorted[-20:] if "product_id" in v],
            "recent_purchases": [p["product_id"] for p in purchases_sorted[-10:] if "product_id" in p],
            "recent_favorites": [f["product_id"] for f in favorites_sorted[-10:] if "product_id" in f],
            "preferred_categories": preferred,
            "rfm": rfm,
            "price_range": price_range,
            "active_hours": active_hours,
        }

    @staticmethod
    def _percentile_range(values: list[float]) -> tuple[float, float]:
        if not values:
            return (0.0, 10000.0)
        s = sorted(values)

        def _pct(p: float) -> float:
            if len(s) == 1:
                return s[0]
            idx = p * (len(s) - 1)
            lo = int(idx)
            hi = min(lo + 1, len(s) - 1)
            frac = idx - lo
            return s[lo] * (1 - frac) + s[hi] * frac

        return (round(_pct(0.1), 2), round(_pct(0.9), 2))

    @staticmethod
    def _compute_rfm(purchases: list[dict]) -> dict[str, float]:
        """Recency/Frequency/Monetary, normalised 0-1."""
        if not purchases:
            return {"recency": 0.0, "frequency": 0.0, "monetary": 0.0}
        now = time.time()
        latest = max(p.get("at", 0) for p in purchases)
        days_since = (now - latest) / DAY
        recency = max(0.0, 1.0 - days_since / 30.0)
        frequency = min(1.0, len(purchases) / 10.0)
        amounts = [p.get("amount") or p.get("price") or 100 for p in purchases]
        avg_amount = sum(amounts) / len(amounts)
        monetary = min(1.0, avg_amount / 1000.0)
        return {
            "recency": round(recency, 3),
            "frequency": round(frequency, 3),
            "monetary": round(monetary, 3),
        }

    # -------- legacy mock fallback --------

    def _mock_get_profile_features(self, user_id: str) -> dict[str, Any]:
        features = self._mock_data.get(f"features:{user_id}")
        if features:
            return features
        return self._empty_features(user_id)

    # -------- helpers for testing / mock seeding --------

    def seed_mock_profile(self, user_id: str, features: dict[str, Any]) -> None:
        """Inject features for a user when running in mock (no Neo4j) mode."""
        self._mock_data[f"features:{user_id}"] = features
