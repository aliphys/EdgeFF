#!/usr/bin/env python3
"""Tests for the consolidated script wrappers."""

import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from scripts import train, evaluate, analysis, sweep


class TestScriptWrappers(unittest.TestCase):
    def test_train_import(self):
        self.assertTrue(callable(train.main))

    def test_evaluate_import(self):
        self.assertTrue(callable(evaluate.main))

    def test_analysis_import(self):
        self.assertTrue(callable(analysis.main))

    def test_sweep_import(self):
        self.assertTrue(callable(sweep.main))

    def _assert_help_exits(self, module, argv):
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        sys_stdout = io.StringIO()
        sys_stderr = io.StringIO()
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr
        saved_argv = sys.argv
        try:
            sys.argv = argv
            with self.assertRaises(SystemExit) as cm:
                module.main()
            self.assertEqual(cm.exception.code, 0)
            output = sys_stdout.getvalue().lower() + sys_stderr.getvalue().lower()
            self.assertIn('usage', output)
        finally:
            sys.argv = saved_argv
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr

    def test_train_help(self):
        self._assert_help_exits(train, ['train.py', '--help'])

    def test_evaluate_inference_help(self):
        self._assert_help_exits(evaluate, ['evaluate.py', 'inference', '--help'])

    def test_evaluate_energy_help(self):
        self._assert_help_exits(evaluate, ['evaluate.py', 'energy', '--help'])

    def test_evaluate_variance_help(self):
        self._assert_help_exits(evaluate, ['evaluate.py', 'variance', '--help'])

    def test_analysis_help(self):
        self._assert_help_exits(analysis, ['analysis.py', '--help'])

    def test_sweep_help(self):
        self._assert_help_exits(sweep, ['sweep.py', '--help'])


if __name__ == '__main__':
    unittest.main()
