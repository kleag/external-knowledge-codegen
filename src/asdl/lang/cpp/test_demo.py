# coding=utf-8

from .demo import *

import argparse
import colorama
import os
import re
import subprocess
import sys
from typing import List

import cppastor
import cpplang
from asdl.lang.cpp.cpp_transition_system import *
from asdl.hypothesis import *

def test_test_dir():

    fail_on_error = False
    check_hypothesis = True
    skip_checks = False
    exclusions = []
    # exclusions.extend(load_exclusions(args.exclusions, debug=True))
    nb_ok = 0
    nb_ko = 0
    filepaths = [
        "test.cpp"
    ]
    test_num = 0

    files = collect_files("test")
    files.remove("test/exclusions.txt")
    print(f"files: {files}", file=sys.stderr)
    total = len(files)
    for filepath in files:
        test_num += 1
        if filepath not in exclusions:
            test_result = check_filepath(filepath,
                                        check_hypothesis=check_hypothesis,
                                        skip_checks=skip_checks,
                                        fail_on_error=fail_on_error,
                                        member=False,
                                        number=test_num,
                                        total=total)
            assert(test_result is not None and test_result)
