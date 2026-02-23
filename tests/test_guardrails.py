"""Tests for guardrail logic."""

import pytest
from aegis.guardrails import Guardrail


def test_guardrail_pass():
    g = Guardrail(confidence_threshold=0.9)
    assert g.check(None, confidence=0.95) is True


def test_guardrail_fail():
    g = Guardrail(confidence_threshold=0.9)
    assert g.check(None, confidence=0.8) is False
