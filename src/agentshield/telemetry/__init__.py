"""OpenTelemetry traces and metrics for one gate evaluation.

Spans never carry tool arguments: those belong in the (optionally redacted) audit record, not
in a collector that many more people can read.

Call :func:`configure` once at process start to export via OTLP. Tests pass a
:class:`Telemetry` built from in-memory providers instead.
"""

from __future__ import annotations

import os
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.metrics import Meter
from opentelemetry.trace import Span, StatusCode, Tracer

from agentshield.audit.record import AuditRecord
from agentshield.core.models import DecisionSource

TRACER_NAME = "agentshield"
SPAN_EVALUATE = "agentshield.evaluate"
SPAN_POLICY = "agentshield.policy"
SPAN_DECIDER = "agentshield.decider"

_ATTR_PREFIX = "agentshield."


class Telemetry:
    def __init__(self, tracer: Tracer | None = None, meter: Meter | None = None) -> None:
        self.tracer = tracer or trace.get_tracer(TRACER_NAME)
        meter = meter or metrics.get_meter(TRACER_NAME)
        self._decisions = meter.create_counter(
            "agentshield.decisions", description="Gate decisions by outcome, source and risk."
        )
        self._evaluate_ms = meter.create_histogram(
            "agentshield.evaluate.duration", unit="ms", description="End-to-end gate latency."
        )
        self._policy_ms = meter.create_histogram(
            "agentshield.policy.duration",
            unit="ms",
            description="Deterministic policy-engine latency.",
        )
        self._decider_ms = meter.create_histogram(
            "agentshield.decider.duration", unit="ms", description="Decider round-trip latency."
        )
        self._errors = meter.create_counter(
            "agentshield.decider.errors", description="Decider failures that triggered fallback."
        )
        self._p_ok = meter.create_histogram(
            "agentshield.check.p_ok",
            description="p_ok of each Decider check (for calibration).",
        )
        self._approvals = meter.create_counter(
            "agentshield.approvals", description="Human approval resolutions."
        )
        self._approval_wait = meter.create_histogram(
            "agentshield.approval.wait", unit="ms", description="Time spent waiting for a human."
        )

    def record(self, audit: AuditRecord, span: Span | None = None) -> None:
        labels: dict[str, Any] = {
            "outcome": audit.outcome.value,
            "source": audit.source.value,
            "risk": audit.risk.value,
        }
        self._decisions.add(1, labels)
        self._evaluate_ms.record(audit.total_ms, labels)
        self._policy_ms.record(audit.policy_ms, labels)
        if audit.decider_ms is not None:
            self._decider_ms.record(audit.decider_ms, labels)
        if audit.decider_error:
            self._errors.add(1, {"source": audit.source.value})
        for check in audit.checks:
            self._p_ok.record(check.p_ok, {"kind": check.kind.value})
        if span is not None:
            annotate_span(span, audit)

    def record_approval(self, result: str, wait_ms: float) -> None:
        labels = {"result": result}
        self._approvals.add(1, labels)
        self._approval_wait.record(wait_ms, labels)


def annotate_span(span: Span, audit: AuditRecord) -> None:
    span.set_attribute(_ATTR_PREFIX + "tool", audit.tool)
    span.set_attribute(_ATTR_PREFIX + "outcome", audit.outcome.value)
    span.set_attribute(_ATTR_PREFIX + "risk", audit.risk.value)
    span.set_attribute(_ATTR_PREFIX + "source", audit.source.value)
    span.set_attribute(_ATTR_PREFIX + "session_id", audit.session_id)
    span.set_attribute(_ATTR_PREFIX + "agent_id", audit.agent_id)
    span.set_attribute(_ATTR_PREFIX + "total_ms", audit.total_ms)
    if audit.rule_id:
        span.set_attribute(_ATTR_PREFIX + "rule_id", audit.rule_id)
    if audit.limiting_check:
        span.set_attribute(_ATTR_PREFIX + "limiting_check", audit.limiting_check)
    if audit.decider_error:
        span.set_attribute(_ATTR_PREFIX + "decider_error", audit.decider_error)
        span.set_status(StatusCode.ERROR, audit.decider_error)
    if audit.source is DecisionSource.FALLBACK:
        span.set_status(StatusCode.ERROR, audit.reason[:200])


def configure(*, endpoint: str | None = None, service_name: str = "agentshield") -> None:
    """Install global Tracer and Meter providers that export over OTLP HTTP.

    ``endpoint`` defaults to ``OTEL_EXPORTER_OTLP_ENDPOINT``. If neither is set, providers
    are installed without an exporter so in-process instrumentation still works.
    """
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from agentshield import __version__

    endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    resource = Resource.create({"service.name": service_name, "service.version": __version__})
    tracer_provider = TracerProvider(resource=resource)
    meter_readers = []
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        meter_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint)))
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=meter_readers))
