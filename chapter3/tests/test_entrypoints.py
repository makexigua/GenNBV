"""Migration checks: CLI discovery and legacy entry points need no GPU."""
from pathlib import Path
import subprocess
import sys
import unittest

from chapter3.paths import REPO_ROOT


class EntryPointTests(unittest.TestCase):
    def test_repository_root_and_lazy_imports(self):
        self.assertEqual(REPO_ROOT, Path(__file__).resolve().parents[2])
        self.assertTrue((REPO_ROOT / 'gennbv/network/hybrid_encoder.py').is_file())
        result = subprocess.run([sys.executable, '-c',
                                 'import sys, chapter3, chapter3.cli, chapter3.perception; '
                                 'assert "isaacgym" not in sys.modules; assert "torch" not in sys.modules'],
                                cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_command_help(self):
        for command in (None, 'eval', 'ablate', 'pair', 'inspect', 'verify-lines', 'verify-image', 'detect-legacy'):
            with self.subTest(command=command):
                args = [sys.executable, '-m', 'chapter3'] + ([command] if command else []) + ['--help']
                result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_published_legacy_commands(self):
        for script in ('eval_pretrained.py', 'verify_image_encoder.py', 'line_detection.py'):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, script, '--help'], cwd=REPO_ROOT,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
