import shutil
import tempfile
import unittest
from pathlib import Path

from dat_grpo.checkpoints import archive_adapter


class SnapshotTests(unittest.TestCase):
    def test_adapter_survives_resume_checkpoint_rotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-200"
            checkpoint.mkdir()
            (checkpoint / "adapter_model.safetensors").write_bytes(b"saved-weights")
            (checkpoint / "adapter_config.json").write_text('{}')
            (checkpoint / "optimizer.pt").write_bytes(b"optimizer-state")
            snapshot = archive_adapter(checkpoint, root / "snapshots", 200)
            shutil.rmtree(checkpoint)
            self.assertEqual((snapshot / "adapter_model.safetensors").read_bytes(), b"saved-weights")
            self.assertFalse((snapshot / "optimizer.pt").exists())
            self.assertEqual(archive_adapter(checkpoint, root / "snapshots", 200), snapshot)

    def test_incomplete_checkpoint_does_not_publish_a_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-200"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text('{}')
            with self.assertRaises(ValueError):
                archive_adapter(checkpoint, root / "snapshots", 200)
            self.assertFalse((root / "snapshots/step-200").exists())

class FullSnapshotTests(unittest.TestCase):
    def test_full_shards_survive_rotation_without_retaining_optimizer(self):
        import json
        from dat_grpo.checkpoints import archive_full_model
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint = root/'checkpoint-250'; checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            (checkpoint/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'model-00001.safetensors','b':'model-00002.safetensors'}}))
            (checkpoint/'model-00001.safetensors').write_bytes(b'first')
            (checkpoint/'model-00002.safetensors').write_bytes(b'second')
            (checkpoint/'optimizer.pt').write_bytes(b'optimizer')
            destination = archive_full_model(checkpoint,root/'snapshots',250)
            shutil.rmtree(checkpoint)
            self.assertEqual((destination/'model-00001.safetensors').read_bytes(),b'first')
            self.assertEqual((destination/'model-00002.safetensors').read_bytes(),b'second')
            self.assertFalse((destination/'optimizer.pt').exists())

    def test_final_export_reuses_shards_and_survives_checkpoint_rotation(self):
        from dat_grpo.checkpoints import publish_full_final
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);checkpoint=root/'checkpoint-1';checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            (checkpoint/'model.safetensors').write_bytes(b'full-weights')
            (checkpoint/'optimizer.pt').write_bytes(b'state')
            destination=publish_full_final(checkpoint,root/'model-final',1)
            self.assertEqual((destination/'model.safetensors').stat().st_ino,(checkpoint/'model.safetensors').stat().st_ino)
            shutil.rmtree(checkpoint)
            self.assertEqual((destination/'model.safetensors').read_bytes(),b'full-weights')
            self.assertFalse((destination/'optimizer.pt').exists())

    def test_explicit_early_steps_and_regular_archive_schedule(self):
        from dat_grpo.checkpoints import full_snapshot_due
        settings={'archive_steps':250,'archive_at_steps':[50,75,100,150,200,250]}
        for step in [50,75,100,150,200,250,500,750,1000,1250]:
            self.assertTrue(full_snapshot_due(step,settings),step)
        for step in [0,25,125,175,225,275]:
            self.assertFalse(full_snapshot_due(step,settings),step)

    def test_live_preserver_waits_for_complete_state_and_survives_rotation(self):
        import json
        from dat_grpo.checkpoints import preserve_available_early_snapshots,archive_full_model
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for step in [100,250]:
                checkpoint=root/'checkpoints'/f'checkpoint-{step}'
                checkpoint.mkdir(parents=True)
                (checkpoint/'config.json').write_text('{}')
                (checkpoint/'model.safetensors').write_bytes(b'weights')
                (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step':step}))
                for name in ['optimizer.pt','scheduler.pt']:
                    (checkpoint/name).write_bytes(b'state')
            self.assertEqual(preserve_available_early_snapshots(root,[100,250],250),[])
            for step in [100,250]:
                (root/'checkpoints'/f'checkpoint-{step}'/'rng_state.pth').write_bytes(b'rng')
            # The live trainer owns regular archives; the helper adds only extras.
            self.assertEqual(preserve_available_early_snapshots(root,[100,250],250),[100])
            snapshot=root/'model-snapshots/step-100/model.safetensors'
            source=root/'checkpoints/checkpoint-100/model.safetensors'
            self.assertEqual(snapshot.stat().st_ino,source.stat().st_ino)
            shutil.rmtree(source.parent)
            self.assertEqual(preserve_available_early_snapshots(root,[100,250],250),[100])
            archive_full_model(root/'checkpoints/checkpoint-250',root/'model-snapshots',250)
            self.assertEqual(preserve_available_early_snapshots(root,[100,250],250),[100,250])
