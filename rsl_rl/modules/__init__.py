# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy and critic modules used by FERPO."""
from .actor_q import ActorQ
from .actor_v import ActorV

__all__ = ["ActorQ", "ActorV"]
