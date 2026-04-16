"""BaoStock 本地适配服务。"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Optional


class StockDataService:
    """封装 BaoStock 查询能力并统一返回结构。"""

    STOCK_FREQUENCIES = {"d", "w", "m", "5", "15", "30", "60"}
    INDEX_AND_VALUATION_FREQUENCIES = {"d", "w", "m"}
    ADJUST_FLAGS = {"1", "2", "3"}

    def __init__(self, bs_client: Any | None = None):
        self.bs = bs_client
        if self.bs is None:
            try:
                import baostock as bs  # type: ignore

                self.bs = bs
            except Exception:
                self.bs = None

    def get_stock_basic(self, code: str) -> dict[str, Any]:
        """查询股票基础信息。"""
        normalized_code = self._normalize_code(code, for_index=False)
        rows = self._query_stock_basic(normalized_code)
        return {
            "query_type": "basic",
            "code": normalized_code,
            "rows": rows,
            "row_count": len(rows),
            "meta": {},
        }

    def get_stock_kdata(
        self,
        code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str = "d",
        adjustflag: str = "3",
    ) -> dict[str, Any]:
        """查询股票 K 线数据。"""
        normalized_code = self._normalize_code(code, for_index=False)
        validated_frequency = self._validate_frequency(frequency, self.STOCK_FREQUENCIES)
        validated_adjustflag = self._validate_adjustflag(adjustflag)
        validated_start, validated_end = self._validate_date_range(start_date, end_date)
        rows = self._query_history_k_data_plus(
            code=normalized_code,
            fields="date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
            start_date=validated_start,
            end_date=validated_end,
            frequency=validated_frequency,
            adjustflag=validated_adjustflag,
        )
        return {
            "query_type": "kdata",
            "code": normalized_code,
            "rows": rows,
            "row_count": len(rows),
            "meta": {
                "frequency": validated_frequency,
                "start_date": validated_start,
                "end_date": validated_end,
                "adjustflag": validated_adjustflag,
            },
        }

    def get_index_data(
        self,
        code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str = "d",
    ) -> dict[str, Any]:
        """查询指数 K 线数据。"""
        normalized_code = self._normalize_code(code, for_index=True)
        validated_frequency = self._validate_frequency(frequency, self.INDEX_AND_VALUATION_FREQUENCIES)
        validated_start, validated_end = self._validate_date_range(start_date, end_date)
        rows = self._query_history_k_data_plus(
            code=normalized_code,
            fields="date,code,open,high,low,close,preclose,volume,amount,pctChg",
            start_date=validated_start,
            end_date=validated_end,
            frequency=validated_frequency,
            adjustflag=None,
        )
        return {
            "query_type": "index",
            "code": normalized_code,
            "rows": rows,
            "row_count": len(rows),
            "meta": {
                "frequency": validated_frequency,
                "start_date": validated_start,
                "end_date": validated_end,
            },
        }

    def get_valuation_data(
        self,
        code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str = "d",
    ) -> dict[str, Any]:
        """查询估值指标数据。"""
        normalized_code = self._normalize_code(code, for_index=False)
        validated_frequency = self._validate_frequency(frequency, self.INDEX_AND_VALUATION_FREQUENCIES)
        validated_start, validated_end = self._validate_date_range(start_date, end_date)
        rows = self._query_history_k_data_plus(
            code=normalized_code,
            fields="date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM",
            start_date=validated_start,
            end_date=validated_end,
            frequency=validated_frequency,
            adjustflag="3",
        )
        return {
            "query_type": "valuation",
            "code": normalized_code,
            "rows": rows,
            "row_count": len(rows),
            "meta": {
                "frequency": validated_frequency,
                "start_date": validated_start,
                "end_date": validated_end,
                "adjustflag": "3",
            },
        }

    def _ensure_client(self) -> Any:
        if self.bs is None:
            raise RuntimeError("BaoStock 依赖不可用，请先安装 baostock")
        return self.bs

    def _normalize_code(self, code: str, *, for_index: bool) -> str:
        if not code or not code.strip():
            raise ValueError("code 不能为空")

        normalized = re.sub(r"\s+", "", code).lower()
        if re.match(r"^(sh|sz)\.\d{6}$", normalized):
            return normalized

        if not re.match(r"^\d{6}$", normalized):
            raise ValueError("code 格式错误，支持 sh.600000 / sz.000001 / 600000")

        if for_index:
            if normalized.startswith("399"):
                return f"sz.{normalized}"
            return f"sh.{normalized}"

        if normalized.startswith(("600", "601", "603", "605", "688", "900")):
            return f"sh.{normalized}"
        if normalized.startswith(("000", "001", "002", "003", "200", "300", "301")):
            return f"sz.{normalized}"
        raise ValueError("无法从 6 位代码推断市场，请使用 sh./sz. 前缀")

    def _validate_date(self, value: str, field_name: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"{field_name} 必须是 YYYY-MM-DD 格式") from exc
        return value

    def _validate_date_range(self, start_date: Optional[str], end_date: Optional[str]) -> tuple[str | None, str | None]:
        validated_start = self._validate_date(start_date, "start_date") if start_date else None
        validated_end = self._validate_date(end_date, "end_date") if end_date else None
        if validated_start and validated_end and validated_start > validated_end:
            raise ValueError("start_date 不能晚于 end_date")
        return validated_start, validated_end

    def _validate_frequency(self, frequency: str, allowed: set[str]) -> str:
        candidate = (frequency or "").strip().lower()
        if candidate not in allowed:
            allowed_values = "|".join(sorted(allowed, key=lambda item: (len(item), item)))
            raise ValueError(f"frequency 不合法，允许值：{allowed_values}")
        return candidate

    def _validate_adjustflag(self, adjustflag: str) -> str:
        candidate = (adjustflag or "").strip()
        if candidate not in self.ADJUST_FLAGS:
            raise ValueError("adjustflag 不合法，允许值：1|2|3")
        return candidate

    def _query_stock_basic(self, code: str) -> list[dict[str, str]]:
        client = self._ensure_client()
        logged_in = False
        try:
            self._login(client)
            logged_in = True
            rs = client.query_stock_basic(code=code)
            return self._collect_rows(rs)
        finally:
            if logged_in:
                self._safe_logout(client)

    def _query_history_k_data_plus(
        self,
        *,
        code: str,
        fields: str,
        start_date: str | None,
        end_date: str | None,
        frequency: str,
        adjustflag: str | None,
    ) -> list[dict[str, str]]:
        client = self._ensure_client()
        logged_in = False
        try:
            self._login(client)
            logged_in = True
            kwargs: dict[str, Any] = {
                "code": code,
                "fields": fields,
                "start_date": start_date,
                "end_date": end_date,
                "frequency": frequency,
            }
            if adjustflag is not None:
                kwargs["adjustflag"] = adjustflag
            rs = client.query_history_k_data_plus(**kwargs)
            return self._collect_rows(rs)
        finally:
            if logged_in:
                self._safe_logout(client)

    def _login(self, client: Any) -> None:
        result = client.login()
        if getattr(result, "error_code", None) != "0":
            raise RuntimeError(f"BaoStock 登录失败: {getattr(result, 'error_msg', 'unknown error')}")

    def _safe_logout(self, client: Any) -> None:
        try:
            client.logout()
        except Exception:
            pass

    def _collect_rows(self, rs: Any) -> list[dict[str, str]]:
        if getattr(rs, "error_code", None) != "0":
            raise RuntimeError(f"BaoStock 查询失败: {getattr(rs, 'error_msg', 'unknown error')}")
        fields = list(getattr(rs, "fields", []) or [])
        rows: list[dict[str, str]] = []
        while rs.next():
            row_data = rs.get_row_data()
            if len(row_data) != len(fields):
                raise RuntimeError("BaoStock 返回字段与数据列不一致")
            rows.append(dict(zip(fields, row_data)))
        return rows
