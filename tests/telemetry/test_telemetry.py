from pathlib import Path
from typing import Any

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agentshield.audit import MemoryAuditLog
from agentshield.core import Outcome, ToolCall
from agentshield.decider import DeciderChecker, DeciderTimeout, FakeDeciderClient, Message
from agentshield.pipeline import Shield
from agentshield.policy import PolicyEngine, load_policy
from agentshield.telemetry import SPAN_DECIDER, SPAN_EVALUATE, SPAN_POLICY, Telemetry

POLICY = Path(__file__).resolve().parents[2] / "policies" / "shophelp.yaml"
CHAT = [Message(role="user", content="Please cancel order A1001.")]
GOOD = {"arg:*": 0.95, "consent": 0.95}


def _telemetry() -> tuple[Telemetry, InMemorySpanExporter, InMemoryMetricReader]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    telemetry = Telemetry(
        tracer=tracer_provider.get_tracer("test"),
        meter=meter_provider.get_meter("test"),
    )
    return telemetry, exporter, reader


def _shield(telemetry: Telemetry, **kw: Any) -> Shield:
    fake = FakeDeciderClient(
        kw.pop("script", GOOD), **{k: kw.pop(k) for k in list(kw) if k == "error"}
    )
    return Shield(
        PolicyEngine(load_policy(POLICY)),
        DeciderChecker(fake),
        telemetry=telemetry,
        sinks=[kw.pop("log")] if "log" in kw else (),
        **kw,
    )


def _counter(reader: InMemoryMetricReader, name: str) -> list[Any]:
    data = reader.get_metrics_data()
    points: list[Any] = []
    if data is None:
        return points
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    points.extend(metric.data.data_points)
    return points


class TestSpans:
    async def test_policy_only_has_no_decider_span(self) -> None:
        tel, exporter, _ = _telemetry()
        await _shield(tel).evaluate(
            ToolCall(name="delete_customer_data", arguments={"customer_id": "C"}), CHAT
        )
        names = [s.name for s in exporter.get_finished_spans()]
        assert SPAN_EVALUATE in names
        assert SPAN_POLICY in names
        assert SPAN_DECIDER not in names

    async def test_full_path_nests_policy_and_decider(self) -> None:
        tel, exporter, _ = _telemetry()
        await _shield(tel).evaluate(
            ToolCall(name="get_order_status", arguments={"order_id": "A1"}), CHAT
        )
        spans = {s.name: s for s in exporter.get_finished_spans()}
        parent = spans[SPAN_EVALUATE]
        policy_parent, decider_parent = spans[SPAN_POLICY].parent, spans[SPAN_DECIDER].parent
        assert policy_parent is not None and decider_parent is not None
        assert policy_parent.span_id == parent.context.span_id
        assert decider_parent.span_id == parent.context.span_id
        attrs = dict(parent.attributes or {})
        assert attrs["agentshield.outcome"] == Outcome.ALLOW.value
        assert attrs["agentshield.tool"] == "get_order_status"

    async def test_arguments_never_appear_on_spans(self) -> None:
        tel, exporter, _ = _telemetry()
        secret = "unique-secret-value-xyz"
        await _shield(tel).evaluate(
            ToolCall(name="get_order_status", arguments={"order_id": secret}), CHAT
        )
        for span in exporter.get_finished_spans():
            blob = " ".join(str(v) for v in (span.attributes or {}).values())
            assert secret not in blob
            assert "order_id" not in (span.attributes or {})

    async def test_fallback_marks_the_parent_as_error(self) -> None:
        tel, exporter, _ = _telemetry()
        await _shield(tel, error=DeciderTimeout("slow")).evaluate(
            ToolCall(name="get_order_status", arguments={"order_id": "A1"}), CHAT
        )
        parent = next(s for s in exporter.get_finished_spans() if s.name == SPAN_EVALUATE)
        assert parent.status.status_code is StatusCode.ERROR


class TestMetrics:
    async def test_counts_a_decision(self) -> None:
        tel, _, reader = _telemetry()
        await _shield(tel).evaluate(
            ToolCall(name="get_order_status", arguments={"order_id": "A1"}), CHAT
        )
        points = _counter(reader, "agentshield.decisions")
        assert sum(p.value for p in points) == 1
        attrs = dict(points[0].attributes)
        assert attrs["outcome"] == "allow"
        assert attrs["source"] == "decider"
        assert attrs["risk"] == "low"

    async def test_decider_errors_are_counted(self) -> None:
        tel, _, reader = _telemetry()
        await _shield(tel, error=DeciderTimeout("slow")).evaluate(
            ToolCall(name="cancel_order", arguments={"order_id": "A1"}), CHAT
        )
        points = _counter(reader, "agentshield.decider.errors")
        assert sum(p.value for p in points) == 1


class TestTogetherWithAudit:
    async def test_span_id_is_not_required_for_the_log(self) -> None:
        tel, _, _ = _telemetry()
        log = MemoryAuditLog()
        v = await _shield(tel, log=log).evaluate(
            ToolCall(name="get_order_status", arguments={"order_id": "A1"}), CHAT
        )
        assert log.get(v.audit_id) is not None  # type: ignore[arg-type]
