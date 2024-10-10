# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import sys

# Ensure the correct path to the dinov2 directory
dinov2_path = "/data/home/bowen/projects/dinov2_trajectory/dinov2/dinov2"
if os.path.isdir(dinov2_path):
    sys.path.append(dinov2_path)
else:
    raise FileNotFoundError(f"The directory {dinov2_path} does not exist")

# Print sys.path for debugging
print("sys.path:", sys.path)

# Check if __init__.py exists in the dinov2 directory
if not os.path.isfile(os.path.join(dinov2_path, "__init__.py")):
    raise FileNotFoundError(f"__init__.py not found in {dinov2_path}")

# Attempt to import modules and print debug information
try:
    from dinov2.logging import setup_logging
    from dinov2.train import get_args_parser as get_train_args_parser
    from dinov2.run.submit import get_args_parser, submit_jobs
    print("Imports successful")
except ImportError as e:
    print(f"ImportError: {e}")
    raise

logger = logging.getLogger("dinov2")


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        from dinov2.train import main as train_main

        self._setup_args()
        train_main(self.args)

    def checkpoint(self):
        import submitit

        logger.info(f"Requeuing {self.args}")
        empty = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty)