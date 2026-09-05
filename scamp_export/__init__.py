"""
scamp_export
============
Export curated, phase-split Pythomics datasets into the legacy "luc"-format
text files that SCAMP (`scamp.m`) ingests. See README.md in this folder for
format details and the manual MATLAB acceptance procedure.

Public API:
    scamp_writer.write_dam_file
    scamp_writer.bin_to_30min
    scamp_writer.military_time
    scamp_exporter.export_dataset_to_scamp
"""

from . import scamp_exporter, scamp_writer

__all__ = ["scamp_writer", "scamp_exporter"]
