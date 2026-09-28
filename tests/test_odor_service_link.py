"""과거 하루 다시 보기 산출물의 계약 구조 검사."""
import json
import unittest
from pathlib import Path

import pandas as pd

from odor_service.event_link.replay import OUT, attach_weather_to_complaints, make_replays, weather_at


SAMPLE = Path(__file__).parent / "fixtures/odor_service/replay_sample.json"


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
                self.assertEqual(set(event), set(reference) | {"weather"})
                self.assertIn(event["level"], ("확인", "출동"))
                self.assertGreaterEqual(len(event["first30_complaints"]), 5)
                self.assertEqual(event["direction_overlap"]["n_complaints"], len(event["first30_complaints"]))
                self.assertTrue(all(row["intensity"] is not None for row in event["first30_complaints"]))
                observed_at = event["weather"]["observed_at"]
                if observed_at:
                    self.assertLessEqual(pd.Timestamp(observed_at), pd.Timestamp(event["event_hour"]))
                self.assertEqual(len(event["top3"]), 3)
                self.assertEqual([r["rank"] for r in event["top3"]], [1, 2, 3])
                self.assertEqual(set(event["direction_overlap"]), set(reference["direction_overlap"]))
                self.assertEqual(set(event["field"]), set(reference["field"]))
                for card in event["field"]["cards"]:
                    self.assertIn("disclaimer", card)
                    required_card_keys = set(reference["field"]["cards"][0]) - {"mode", "confidence"}
                    self.assertTrue(required_card_keys <= set(card))
                    self.assertTrue({"selection_summary", "evidence", "next_action", "merged_count", "merged_names"} <= set(card))

    def test_field_cards_use_complaint_overlap_and_history(self):
        cards = []
        for entry in self.index:
            replay = json.loads((OUT / f"{entry['night_date']}.json").read_text(encoding="utf-8"))
            for event in replay["events"]:
                cards.extend(event["field"]["cards"])
        self.assertTrue(any("민원 위치" in " ".join(card.get("evidence", [])) for card in cards))
        self.assertTrue(any(card["components"]["과거 반복"] > 0 for card in cards))

    def test_candidates_are_anchored_to_complaint_locations(self):
        """후보는 민원 위치 4 km 안·풍상에서만 나오고, 교차 확인이 있으면 단일 지점 후보를 섞지 않는다."""
        for entry in self.index:
            replay = json.loads((OUT / f"{entry['night_date']}.json").read_text(encoding="utf-8"))
            for event in replay["events"]:
                field = event["field"]
                self.assertEqual(field["method"], "complaint_union")
                cards = field["cards"]
                self.assertLessEqual(len(cards), 5)
                if field["mode"] is None:
                    self.assertEqual(cards, [])
                    continue
                self.assertLessEqual(field["locations"], len(event["first30_complaints"]))
                tiers = {card["tier"] for card in cards}
                if field["selection"] == "교차 확인":
                    self.assertEqual(tiers, {"교차 확인"})
                for card in cards:
                    self.assertGreaterEqual(card["support"]["count"], 1)
                    self.assertEqual(card["support"]["total"], field["locations"])
                    self.assertLessEqual(card["complaint_km"], 4.0 + 1e-6)
                    self.assertGreater(card["components"]["다중 측정 일치"], 0)
                self.assertEqual(sorted(card["visit_order"] for card in cards), list(range(1, len(cards) + 1)))
                self.assertEqual([card["rank"] for card in cards], list(range(1, len(cards) + 1)))


class ReplayHistoryInputTest(unittest.TestCase):
    def test_optional_weather_is_exact_hour_and_preserves_missing(self):
        frame = pd.DataFrame({"datetime": pd.to_datetime(["2025-07-01 22:00", "2025-07-01 23:00"]),
                              "humidity": [71.0, 99.0], "rainfall_hour": [0.0, 10.0]})
        result = weather_at(frame, pd.Timestamp("2025-07-01 22:59"))
        self.assertEqual(result["humidity"], 71)
        self.assertEqual(result["rainfall"], 0)
        self.assertIsNone(weather_at(frame.drop(columns="humidity"), pd.Timestamp("2025-07-01 22:59"))["humidity"])
        self.assertIsNone(weather_at(frame, pd.Timestamp("2025-07-02"))["rainfall"])

    def test_attaches_hourly_weather_without_changing_complaint_count(self):
        complaints = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-07-01 22:10", "2025-07-01 22:50"]),
            "latitude": [35.9, 35.91],
            "longitude": [127.0, 127.01],
        })
        weather = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-07-01 22:00"]),
            "wind_direction": [320.0],
            "wind_speed": [1.2],
        })
        result = attach_weather_to_complaints(complaints, weather)
        self.assertEqual(len(result), len(complaints))
        self.assertEqual(result["wind_direction"].tolist(), [320.0, 320.0])
        self.assertEqual(result["wind_speed"].tolist(), [1.2, 1.2])


if __name__ == "__main__":
    unittest.main()
