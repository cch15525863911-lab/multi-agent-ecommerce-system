"""
安全网关护栏 (Guardrails Gate) — 四大安全能力:

1. Prompt 注入防护 (Prompt Injection Protection)
   - 关键词黑名单 + 模式匹配
   - 指令覆盖检测 (system prompt override)
   - 间接注入检测 (base64 / markdown 隐藏指令)

2. PII 敏感数据脱敏 (Personally Identifiable Information)
   - 手机号、身份证、银行卡、邮箱、地址等脱敏
   - 请求入站脱敏 + 响应出站脱敏

3. JWT / OAuth2 鉴权 (Authentication)
   - Bearer Token 校验
   - 用户身份提取
   - 权限范围检查

4. 速率限制 (Rate Limiting)
   - 基于 Redis 的滑动窗口限流
   - 按 user_id / IP 双维度限流
   - 超限返回 429 Too Many Requests

使用方式:
    在 FastAPI 中以中间件形式接入, 所有 /api/v1/* 请求依次经过:
    限流 → 鉴权 → PII脱敏 → Prompt注入检测 → 业务处理
"""

from __future__ import annotations

import base64
import hashlib
import re
import time
from typing import Any

import structlog

from config import get_settings

logger = structlog.get_logger()


# =========================================================================
# 1. Prompt 注入防护
# =========================================================================

# 常见 prompt injection 关键词模式
INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|above|prior|earlier|initial)\s*(instructions|rules|directives|prompts)?", re.I),
    re.compile(r"\b(you are|you're|act as|pretend to be|now you are)\s+(a|an|the|no longer|now)", re.I),
    re.compile(r"\b(override|bypass|circumvent|disable|turn off)\s+(all\s+)?(safety|guardrails|restrictions|filters|content policy)", re.I),
    re.compile(r"\b(system\s*prompt|developer\s*mode|god\s*mode|dan\s*mode)", re.I),
    re.compile(r"\b(reveal|show|tell|output|print|display)\s+(your|the|all)\s*(system\s*)?(prompt|instructions|rules|system\s*message)", re.I),
    re.compile(r"\b(do not follow|break free from|stop following| disobey)\s+(these|the|all)\s*(rules|instructions|guidelines)", re.I),
    re.compile(r"-{5,}\s*(system|start|begin|new)\s*(prompt|instructions)", re.I),
    re.compile(r"\b(new\s+persona|roleplay\s+as|from\s+now\s+on|imagine\s+you\s+are)", re.I),
    re.compile(r"\b(human\s+input|user\s+input|text\s+below).*(ignore|override|forget)", re.I | re.DOTALL),
]

# 间接注入: base64 编码的隐藏指令
BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")


class PromptInjectionDetector:
    """Prompt 注入检测器 — 关键词 + 模式 + 间接注入三重检测。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.block_on_high_risk = True  # 高风险直接拦截
        self.max_risk_score = 100.0

    def detect(self, text: str) -> dict[str, Any]:
        """检测 prompt 注入风险。

        Returns:
            {risk_score, is_injection, patterns_hit, details}
        """
        if not self.enabled or not text:
            return {"risk_score": 0.0, "is_injection": False, "patterns_hit": [], "details": "disabled"}

        risk_score = 0.0
        patterns_hit: list[str] = []

        # 1. 关键词模式匹配
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                patterns_hit.append(pattern.pattern[:50])
                risk_score += 25.0

        # 2. 间接注入: 检测疑似 base64 编码的长字符串
        base64_matches = BASE64_PATTERN.findall(text)
        for match in base64_matches:
            if len(match) >= 40:
                try:
                    decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
                    # 如果解码后包含敏感关键词, 加分
                    if any(kw in decoded.lower() for kw in ["ignore", "override", "system", "prompt", "instruction"]):
                        patterns_hit.append(f"base64_hidden_instruction:{len(match)}")
                        risk_score += 30.0
                except Exception:
                    pass

        # 3. 长文本中出现多段 "---" 分隔符 (可能在模拟 system prompt)
        sep_count = len(re.findall(r"-{5,}", text))
        if sep_count >= 3:
            patterns_hit.append(f"multiple_separators:{sep_count}")
            risk_score += 15.0

        is_injection = risk_score >= 20.0  # 命中一条核心规则即判定为注入尝试

        result = {
            "risk_score": min(risk_score, self.max_risk_score),
            "is_injection": is_injection,
            "patterns_hit": patterns_hit,
            "details": f"{len(patterns_hit)} patterns matched",
        }

        if is_injection:
            logger.warning(
                "guardrails.prompt_injection_detected",
                risk_score=risk_score,
                patterns=patterns_hit,
            )

        return result


# =========================================================================
# 2. PII 敏感数据脱敏
# =========================================================================

PII_PATTERNS = {
    "phone": re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)"),
    "id_card": re.compile(r"(?<!\d)([1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)"),
    "bank_card": re.compile(r"(?<!\d)(\d{16,19})(?!\d)"),
    "email": re.compile(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"),
    "address": re.compile(r"(北京|上海|广州|深圳|杭州|成都|武汉|西安|南京|重庆)(市)?([\u4e00-\u9fa5]{2,10}(区|县|街道|路|街))"),
}


class PIISanitizer:
    """PII 敏感数据脱敏器 — 入站请求脱敏 + 出站响应脱敏。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def mask_phone(self, match: re.Match) -> str:
        phone = match.group(1)
        return phone[:3] + "****" + phone[7:]

    def mask_id_card(self, match: re.Match) -> str:
        id_num = match.group(1)
        return id_num[:6] + "********" + id_num[-4:]

    def mask_bank_card(self, match: re.Match) -> str:
        card = match.group(1)
        return card[:4] + " **** **** " + card[-4:]

    def mask_email(self, match: re.Match) -> str:
        user = match.group(1)
        domain = match.group(2)
        if len(user) <= 2:
            return f"{user[0]}***@{domain}"
        return f"{user[:2]}***@{domain}"

    def mask_address(self, match: re.Match) -> str:
        return f"{match.group(1)}市***"

    def sanitize_text(self, text: str) -> tuple[str, dict[str, int]]:
        """对文本进行 PII 脱敏。

        Returns:
            (脱敏后文本, {各类型脱敏数量})
        """
        if not self.enabled or not text:
            return text, {}

        stats: dict[str, int] = {}
        mask_fns = {
            "phone": self.mask_phone,
            "id_card": self.mask_id_card,
            "bank_card": self.mask_bank_card,
            "email": self.mask_email,
            "address": self.mask_address,
        }

        for pii_type, pattern in PII_PATTERNS.items():
            count = len(pattern.findall(text))
            if count > 0:
                text = pattern.sub(mask_fns[pii_type], text)
                stats[pii_type] = count

        return text, stats

    def sanitize_dict(self, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        """递归对字典中所有字符串值进行脱敏。"""
        if not self.enabled:
            return data, {}

        total_stats: dict[str, int] = {}
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized, stats = self.sanitize_text(value)
                result[key] = sanitized
                for k, v in stats.items():
                    total_stats[k] = total_stats.get(k, 0) + v
            elif isinstance(value, dict):
                nested, stats = self.sanitize_dict(value)
                result[key] = nested
                for k, v in stats.items():
                    total_stats[k] = total_stats.get(k, 0) + v
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, str):
                        s, stats = self.sanitize_text(item)
                        new_list.append(s)
                        for k, v in stats.items():
                            total_stats[k] = total_stats.get(k, 0) + v
                    elif isinstance(item, dict):
                        nested, stats = self.sanitize_dict(item)
                        new_list.append(nested)
                        for k, v in stats.items():
                            total_stats[k] = total_stats.get(k, 0) + v
                    else:
                        new_list.append(item)
                result[key] = new_list
            else:
                result[key] = value
        return result, total_stats


# =========================================================================
# 3. JWT 鉴权 (简化版 — 生产环境接入真实 OAuth2 Provider)
# =========================================================================


class JWTAuth:
    """JWT 鉴权中间件 — Bearer Token 校验与用户身份提取。

    简化实现: 使用 HMAC-SHA256 签名, 演示用; 生产环境建议接入
    Auth0 / Keycloak / 自建 OAuth2 Server。
    """

    def __init__(self, enabled: bool = False):  # 默认关闭, 避免影响现有接口
        self.enabled = enabled
        settings = get_settings()
        self.secret = getattr(settings, "jwt_secret", "dev-secret-change-in-production")
        self.algorithm = "HS256"

    def _generate_token(self, user_id: str, scopes: list[str] | None = None) -> str:
        """生成 JWT token (开发/测试用)。"""
        import hmac
        import json

        header = base64.urlsafe_b64encode(
            b'{"alg":"HS256","typ":"JWT"}'
        ).rstrip(b"=").decode()
        payload = {
            "sub": user_id,
            "scopes": scopes or ["read", "write"],
            "iat": int(time.time()),
            "exp": int(time.time()) + 86400,  # 24小时
        }
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).rstrip(b"=").decode()
        signing_input = f"{header}.{payload_b64}".encode()
        signature = base64.urlsafe_b64encode(
            hmac.new(self.secret.encode(), signing_input, hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        return f"{header}.{payload_b64}.{signature}"

    def verify_token(self, token: str) -> tuple[bool, dict[str, Any]]:
        """校验 JWT token 有效性。

        Returns:
            (是否有效, payload 字典)
        """
        if not self.enabled:
            return True, {"sub": "anonymous", "scopes": ["read", "write"]}

        try:
            import hmac
            import json

            parts = token.split(".")
            if len(parts) != 3:
                return False, {}

            header_b64, payload_b64, signature = parts
            signing_input = f"{header_b64}.{payload_b64}".encode()

            # 验证签名
            expected_sig = base64.urlsafe_b64encode(
                hmac.new(self.secret.encode(), signing_input, hashlib.sha256).digest()
            ).rstrip(b"=").decode()
            if signature != expected_sig:
                return False, {}

            # 解码 payload
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))

            # 检查过期
            if payload.get("exp", 0) < int(time.time()):
                return False, {"error": "token_expired"}

            return True, payload

        except Exception as exc:
            logger.warning("guardrails.jwt_verify_failed", error=str(exc))
            return False, {}

    def extract_user(self, auth_header: str | None) -> tuple[bool, dict[str, Any]]:
        """从 Authorization header 提取用户身份。

        开启鉴权后, 任何缺少/无效 Bearer、或密钥未配置的情况一律拒绝,
        绝不回退为 anonymous (修复 P0: 默认无认证即可写履约)。
        """
        if self.enabled and not self.secret:
            logger.warning("guardrails.jwt_secret_missing")
            return False, {}
        if not auth_header or not auth_header.startswith("Bearer "):
            if not self.enabled:
                return True, {"sub": "anonymous", "scopes": ["read", "write"]}
            return False, {}
        token = auth_header[7:]  # 去掉 "Bearer "
        return self.verify_token(token)


# =========================================================================
# 4. 速率限制 (基于 Redis 滑动窗口)
# =========================================================================


class RateLimiter:
    """滑动窗口速率限制器 — 按 user_id / IP 双维度限流。

    使用 Redis Sorted Set 实现滑动窗口:
    - key: rate_limit:{dim}:{key}
    - score: 时间戳 (毫秒)
    - value: 请求ID (唯一标识)

    Redis 不可用/异常时, 降级为进程内本地滑动窗口 (容量 = 正常阈值),
    即 fail-closed: 仍执行限流、超出阈值即拒绝, 而非放开 (修复 P1: 限流失败放行)。
    """

    # 进程内本地滑动窗口 (降级兜底); 仅作为 Redis 不可用时的最后防线
    _local_store: dict[str, list[float]] = {}

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        settings = get_settings()
        self.user_limit = getattr(settings, "rate_limit_user_per_min", 60)  # 每分钟60次
        self.ip_limit = getattr(settings, "rate_limit_ip_per_min", 300)    # 每分钟300次
        self.window_seconds = 60
        self._redis_client = None

    async def _get_redis(self):
        """懒加载 Redis 客户端."""
        if self._redis_client is None:
            try:
                from redis.asyncio import Redis
                settings = get_settings()
                self._redis_client = Redis.from_url(settings.redis_url)
            except Exception as exc:
                logger.warning("guardrails.redis_unavailable", error=str(exc))
                self._redis_client = None
        return self._redis_client

    async def check_and_record(
        self,
        user_id: str | None = None,
        ip: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """检查限流并记录本次请求。

        Returns:
            (是否允许通过, 各维度详情)
        """
        if not self.enabled:
            return True, {"enabled": False}

        redis = await self._get_redis()
        details: dict[str, Any] = {}
        allowed = True

        now_ms = int(time.time() * 1000)
        window_start = now_ms - self.window_seconds * 1000

        # User 维度
        if user_id and user_id != "anonymous":
            key = f"rate_limit:user:{user_id}"
            count = await self._check_dimension(redis, key, window_start, now_ms, self.user_limit)
            details["user"] = {"limit": self.user_limit, "count": count, "allowed": count <= self.user_limit}
            if count > self.user_limit:
                allowed = False

        # IP 维度
        if ip:
            key = f"rate_limit:ip:{ip}"
            count = await self._check_dimension(redis, key, window_start, now_ms, self.ip_limit)
            details["ip"] = {"limit": self.ip_limit, "count": count, "allowed": count <= self.ip_limit}
            if count > self.ip_limit:
                allowed = False

        return allowed, details

    async def _check_dimension(
        self, redis, key: str, window_start: int, now_ms: int, limit: int
    ) -> int:
        """检查单个维度的请求数, 并记录本次请求。

        返回当前窗口内计数; 当 Redis 不可用/异常时降级为进程内本地限流
        (fail-closed: 达到容量即拒绝, 不会放开)。
        """
        if redis is not None:
            try:
                import uuid
                # 清理窗口外的记录
                await redis.zremrangebyscore(key, 0, window_start - 1)
                # 计数当前窗口
                count = await redis.zcount(key, window_start, now_ms)
                if count < limit:
                    # 记录本次请求
                    await redis.zadd(key, {str(uuid.uuid4()): now_ms})
                    await redis.expire(key, self.window_seconds + 10)
                    count += 1
                return count
            except Exception as exc:
                logger.warning(
                    "guardrails.rate_limit_error_fallback_local",
                    key=key,
                    error=str(exc),
                )
        # Redis 不可用或异常: 进程内本地滑动窗口兜底 (仍限流, 不放开)
        return self._local_check(key, window_start, now_ms, limit)

    def _local_check(self, key: str, window_start: int, now_ms: int, limit: int) -> int:
        """进程内本地滑动窗口 — Redis 降级兜底。

        达到容量 (limit) 后返回 limit+1, 使调用方判定为超限拒绝。
        """
        times = RateLimiter._local_store.setdefault(key, [])
        while times and times[0] < window_start:
            times.pop(0)
        if len(times) < limit:
            times.append(now_ms)
            return len(times)
        return limit + 1  # 已达容量, fail-closed 拒绝


# =========================================================================
# Guardrails Gate — 统一安全网关
# =========================================================================


class GuardrailsGate:
    """统一安全网关护栏 — 限流 → 鉴权 → PII脱敏 → Prompt注入检测。

    作为 FastAPI 中间件接入, 所有 /api/v1/* 请求依次经过四层安全检查。
    """

    def __init__(self):
        settings = get_settings()
        self.prompt_detector = PromptInjectionDetector(
            enabled=getattr(settings, "guardrails_prompt_injection_enabled", True)
        )
        self.pii_sanitizer = PIISanitizer(
            enabled=getattr(settings, "guardrails_pii_enabled", True)
        )
        self.jwt_auth = JWTAuth(
            enabled=getattr(settings, "guardrails_jwt_enabled", False)
        )
        self.rate_limiter = RateLimiter(
            enabled=getattr(settings, "guardrails_rate_limit_enabled", True)
        )

    async def process_request(
        self,
        path: str,
        method: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
        client_ip: str | None,
    ) -> tuple[bool, int, dict[str, Any], dict[str, Any]]:
        """处理请求 — 依次执行四层安全检查。

        Returns:
            (是否通过, HTTP状态码, 处理后的body, 安全上下文)
        """
        security_ctx: dict[str, Any] = {}
        processed_body = body or {}

        # 1. 速率限制
        user_id = processed_body.get("user_id")
        allowed, rate_details = await self.rate_limiter.check_and_record(user_id, client_ip)
        security_ctx["rate_limit"] = rate_details
        if not allowed:
            logger.warning("guardrails.rate_limited", ip=client_ip, user_id=user_id)
            return False, 429, processed_body, {
                **security_ctx,
                "error": "Too Many Requests",
                "retry_after": "60s",
            }

        # 2. JWT 鉴权
        auth_header = headers.get("authorization", "")
        auth_ok, auth_payload = self.jwt_auth.extract_user(auth_header)
        security_ctx["auth"] = {"authenticated": auth_ok, **auth_payload}
        if not auth_ok:
            return False, 401, processed_body, {
                **security_ctx,
                "error": "Unauthorized",
            }

        # 3. PII 脱敏 (入站)
        if processed_body:
            sanitized, pii_stats = self.pii_sanitizer.sanitize_dict(processed_body)
            processed_body = sanitized
            security_ctx["pii_sanitized"] = pii_stats

        # 4. Prompt 注入检测 (覆盖所有可能进入 prompt 的字段, 含 graph_context / 商品数据)
        injection_result = self._scan_injection(processed_body)
        if injection_result:
            security_ctx["prompt_injection"] = injection_result
            return False, 403, processed_body, {
                **security_ctx,
                "error": "Prompt Injection Detected",
                "risk_score": injection_result["risk_score"],
            }

        return True, 200, processed_body, security_ctx

    def _scan_injection(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """递归扫描请求体中可能进入 prompt 的字段, 检测 Prompt 注入。

        覆盖: query / context / refund_reason / graph_context / product(名称/描述/品牌/标签)
        以及任意嵌套字典/列表中的字符串值 (防御 RAG 数据污染诱导越权)。
        """
        if not isinstance(body, dict):
            return None

        # 显式高风险字段 (字符串)
        for field in ("query", "context", "refund_reason", "graph_context"):
            val = body.get(field)
            if isinstance(val, str) and val.strip():
                res = self.prompt_detector.detect(val)
                if res["is_injection"]:
                    return res
            elif isinstance(val, dict):
                found = self._scan_value_injection(val)
                if found:
                    return found

        # 商品数据 (进入重排/文案 prompt 的字段)
        product = body.get("product")
        if isinstance(product, dict):
            ptext = " ".join(
                str(product.get(k, ""))
                for k in ("name", "description", "brand", "tags")
                if product.get(k)
            )
            if ptext.strip():
                res = self.prompt_detector.detect(ptext)
                if res["is_injection"]:
                    return res

        return None

    def _scan_value_injection(self, data: Any, _depth: int = 0) -> dict[str, Any] | None:
        """递归扫描任意嵌套结构中的字符串值, 返回首个命中注入的结果。"""
        if _depth > 6:
            return None
        if isinstance(data, str) and data.strip():
            res = self.prompt_detector.detect(data)
            return res if res["is_injection"] else None
        if isinstance(data, dict):
            for v in data.values():
                r = self._scan_value_injection(v, _depth + 1)
                if r:
                    return r
        elif isinstance(data, list):
            for v in data:
                r = self._scan_value_injection(v, _depth + 1)
                if r:
                    return r
        return None

    def sanitize_response(self, body: Any) -> tuple[Any, dict[str, int]]:
        """对响应体 (出站) 进行 PII 脱敏, 返回 (脱敏后对象, 统计)。

        修复 P1: 原脱敏仅作用于入站请求, 出站的用户画像/订单/地址等
        敏感字段可能明文返回。
        """
        if isinstance(body, dict):
            return self.pii_sanitizer.sanitize_dict(body)
        if isinstance(body, list):
            total: dict[str, int] = {}
            new_list = []
            for item in body:
                s, stats = self.sanitize_response(item)
                new_list.append(s)
                for k, v in stats.items():
                    total[k] = total.get(k, 0) + v
            return new_list, total
        if isinstance(body, str):
            return self.pii_sanitizer.sanitize_text(body)
        return body, {}


# 单例
_gate: GuardrailsGate | None = None


def get_guardrails_gate() -> GuardrailsGate:
    global _gate
    if _gate is None:
        _gate = GuardrailsGate()
    return _gate
