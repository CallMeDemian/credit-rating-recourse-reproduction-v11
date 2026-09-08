"""Reviewer-facing thesis output preparation.

The package deliberately separates three authorities:

* the thesis DOCX defines the inventory, captions, sections and printed claims;
* the selected run defines empirical source data;
* source/config files define non-empirical design statements.

Printed thesis values and legacy paper assets are comparison references only.
"""

from .docx_inventory import extract_thesis_inventory

__all__ = ["extract_thesis_inventory"]
