from adaptive_chunking.parsing import _assign_title_end_offsets


def test_nested_titles_end_at_next_shallower_title():
    titles = [
        {"start": 0, "level": 1},
        {"start": 10, "level": 2},
        {"start": 20, "level": 1},
    ]
    _assign_title_end_offsets(titles, 30)
    assert titles[0]["end"] == 20  # next level <= 1 is at offset 20
    assert titles[1]["end"] == 20  # next level <= 2 is the level-1 at 20
    assert titles[2]["end"] == 30  # nothing after -> document end


def test_deeper_title_does_not_close_shallower():
    titles = [
        {"start": 0, "level": 1},
        {"start": 5, "level": 2},
        {"start": 8, "level": 3},
    ]
    _assign_title_end_offsets(titles, 50)
    # the level-1 title spans to the document end (no later level <= 1)
    assert titles[0]["end"] == 50
    assert titles[1]["end"] == 50
    assert titles[2]["end"] == 50


def test_empty_titles_is_noop():
    titles = []
    _assign_title_end_offsets(titles, 10)
    assert titles == []
