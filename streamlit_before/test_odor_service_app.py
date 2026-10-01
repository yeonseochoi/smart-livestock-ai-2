import tempfile
import unittest
from pathlib import Path
from unittest import mock

from odor_service import app
from odor_service.app import SAMPLE, compass_ko, load_replay, replay_files


class DevelopAppTest(unittest.TestCase):
    def test_sample_loading(self):
        replay = load_replay(SAMPLE)
        self.assertEqual(replay["night_date"], "2022-06-30")
        self.assertEqual(replay["afternoon"]["season_rank"], 8)
        self.assertEqual(replay["standby"]["chosen_point_id"], "AWS-YONGJI")
        self.assertEqual(replay["events"][0]["field"]["cards"][0]["score"], 71.2)

    def test_falls_back_to_sample_without_replay_files(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(app, "REPLAY_DIR", Path(d)):
            self.assertIn(SAMPLE, replay_files())

    def test_uses_replay_files_when_present(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(app, "REPLAY_DIR", Path(d)):
            (Path(d) / "2022-06-30.json").write_text("{}", encoding="utf-8")
            (Path(d) / "index.json").write_text("[]", encoding="utf-8")
            names = [p.name for p in replay_files()]
            self.assertEqual(names, ["2022-06-30.json"])

    def test_compass(self):
        self.assertEqual(compass_ko(0), "북")
        self.assertEqual(compass_ko(90), "동")
        self.assertEqual(compass_ko(190), "남")


if __name__ == "__main__":
    unittest.main()
