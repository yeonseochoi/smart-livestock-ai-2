"""과거 하루 다시 보기 산출물의 계약 구조 검사."""
import json
import unittest
from pathlib import Path

from develop.event_link.replay import OUT, make_replays


SAMPLE = Path(__file__).parent / "fixtures/develop/replay_sample.json"


class ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index, _ = make_replays()
        cls.sample = json.loads(SAMPLE.read_text(encoding="utf-8"))

    def test_structure_and_top3(self):
        self.assertTrue(self.index)
        for entry in self.index:
            replay = json.loads((OUT / f"{entry['night_date']}.json").read_text(encoding="utf-8"))
            self.assertEqual(set(replay), set(self.sample) - {"_note"})
            self.assertEqual(set(replay["afternoon"]), set(self.sample["afternoon"]))
            self.assertEqual(set(replay["standby"]), set(self.sample["standby"]))
            for point in replay["standby"]["points"]:
                self.assertEqual(set(point), set(self.sample["standby"]["points"][0]))
            for event in replay["events"]:
                reference = self.sample["events"][0]
                self.assertEqual(set(event), set(reference))
                self.assertIn(event["level"], ("확인", "출동"))
                self.assertEqual(len(event["top3"]), 3)
                self.assertEqual([r["rank"] for r in event["top3"]], [1, 2, 3])
                self.assertEqual(set(event["direction_overlap"]), set(reference["direction_overlap"]))
                self.assertEqual(set(event["field"]), set(reference["field"]))
                for card in event["field"]["cards"]:
                    self.assertIn("disclaimer", card)
                    self.assertEqual(set(reference["field"]["cards"][0]) - {"mode", "confidence"}, set(card) - {"mode", "confidence"})


if __name__ == "__main__":
    unittest.main()
