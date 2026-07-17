#!/usr/bin/env python3
"""Shims for upstream helpers used by the ported DUET agent code.

Replaces ``utils.distributed`` and ``utils.logger`` from HM3DAutoVLN with
the minimal subset needed by agent_base.py / agent_obj.py.
"""
import sys
import math
import torch.distributed as dist


def is_default_gpu(opts=None) -> bool:
    """Returns True if this process is rank 0 (or distributed is not used)."""
    try:
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0
    except Exception:
        return True


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """Terminal progress bar — verbatim from HM3DAutoVLN/utils/logger.py."""
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
