#!/usr/bin/env python3
"""Compatibility wrapper for source-tree and installed CLI usage."""

from fast_pcc_rmsd.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
