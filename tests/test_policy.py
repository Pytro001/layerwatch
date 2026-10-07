from layerwatch.agent.policy import CONTINUE, RE_IMAGE, RECOAT, STOP, Agent


def m(det=(), ok=True, reasons=(), iou=0.9, expected_px=2000):
    ev = {"lap_ratio": 1.0, "saturated_frac": 0.0, "anisotropy": 0.9}
    return {"layer_index": 1, "detections": list(det),
            "evidence": {"ok": ok, "reasons": list(reasons), "melt": ev, "recoat": ev},
            "slice": {"iou": iou, "match_shapes": 0.01, "expected_px": expected_px, "observed_px": expected_px}}


def streak(length=300, inside=40):
    return {"type": "streak", "bbox": [50, 200, 50 + length, 204], "length_px": length, "inside_part_px": inside}


def test_clean_layer_continues():
    assert Agent().decide(m()).action == CONTINUE


def test_bad_evidence_asks_for_reimage_never_guesses():
    d = Agent().decide(m(det=[streak()], ok=False, reasons=["glare: 1.2% saturated"]))
    assert d.action == RE_IMAGE and "glare" in d.reason


def test_bad_evidence_after_two_reimages_stops():
    assert Agent().decide(m(ok=False, reasons=["blurred"]), attempt=2).action == STOP


def test_long_streak_across_part_recoats_then_stops_if_it_survives():
    a = Agent()
    d1 = a.decide(m(det=[streak()]), recoats=0)
    assert d1.action == RECOAT and d1.evidence["longest_px"] == 300
    d2 = a.decide(m(det=[streak()]), recoats=1)
    assert d2.action == STOP and "blade" in d2.reason


def test_short_streak_outside_part_is_only_noted():
    d = Agent().decide(m(det=[streak(length=80, inside=0)]))
    assert d.action == CONTINUE and "streak" in d.reason


def test_streak_trend_stops():
    a = Agent()
    for _ in range(2):
        r = m(det=[streak()])
        a.commit(r, a.decide(r, recoats=0))
    assert a.decide(m(det=[streak()]), recoats=0).action == STOP


def test_short_feed_recoats_then_stops_after_max():
    sf = {"type": "short_feed", "bbox": [300, 100, 450, 250], "area_px": 9000}
    a = Agent()
    assert a.decide(m(det=[sf]), recoats=0).action == RECOAT
    assert a.decide(m(det=[sf]), recoats=2).action == STOP


def test_super_elevation_large_arc_stops_small_arc_watches():
    big = {"type": "super_elevation", "bbox": [0, 0, 80, 10], "arc_len_px": 80}
    small = {"type": "super_elevation", "bbox": [0, 0, 30, 6], "arc_len_px": 30}
    assert Agent().decide(m(det=[big])).action == STOP
    d = Agent().decide(m(det=[small]))
    assert d.action == CONTINUE and d.watch


def test_super_elevation_two_layers_in_a_row_stops():
    small = {"type": "super_elevation", "bbox": [0, 0, 30, 6], "arc_len_px": 30}
    a = Agent()
    r = m(det=[small])
    a.commit(r, a.decide(r))
    assert a.decide(m(det=[small])).action == STOP


def test_slice_mismatch_stops_only_on_measurable_slices():
    assert Agent().decide(m(iou=0.3)).action == STOP
    assert Agent().decide(m(iou=0.3, expected_px=100)).action == CONTINUE


def test_spatter_never_stops_on_its_own():
    sp = {"type": "spatter", "bbox": [0, 0, 20, 20], "count": 40}
    d = Agent().decide(m(det=[sp]))
    assert d.action == CONTINUE and "CT" in d.reason


def test_every_decision_carries_numbers():
    d = Agent().decide(m(det=[streak()]))
    assert d.evidence and all(isinstance(v, (int, float)) for v in d.evidence.values())
