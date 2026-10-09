#!/usr/bin/env python3
"""Run isolated CPU unit tests without executing the Habitat trainer registry.

Only the package initializer is bypassed; every tested module is real source.
This is not an environment-import or real navigation acceptance test.
"""
import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
package = types.ModuleType('vlnce_baselines')
package.__path__ = [str(root / 'vlnce_baselines')]
sys.modules['vlnce_baselines'] = package
import pytest
raise SystemExit(pytest.main(sys.argv[1:] or ['-q', 'tests/test_progressive_core.py']))
