#!/usr/bin/env python3
"""Backward-compatible launcher for the installable :mod:`sltcap_plus` CLI."""

from sltcap_plus import cli


if __name__ == "__main__":
    raise SystemExit(cli())
