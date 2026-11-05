import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('guarded_process',Path(__file__).resolve().parents[1]/'scripts/guarded_process.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class WatchdogTests(unittest.TestCase):
    def test_normal_paging_with_headroom_and_resource_exhaustion_are_distinguished(self):
        gib=2**30
        reason=lambda available,growth:module.memory_stop_reason(available*gib,growth*gib,18,.25)
        self.assertIsNone(reason(40,.4))
        self.assertEqual(reason(25,.4),'swap_growth')
        self.assertEqual(reason(40,4.1),'excessive_swap_growth')
        self.assertEqual(reason(17.9,0),'host_memory_floor')
