"""Controller events for a solo real-robot episode loop, independent of XR hardware."""

from teleop.utils.dimenso_episode_buttons import EpisodeButtons


def test_quest_face_buttons_emit_once_and_do_not_quit_after_engage_release():
    buttons = EpisodeButtons(hold_ticks=3)
    assert buttons.sample(x=True, a=True, y=False, b=False) == []
    assert buttons.sample(x=False, a=True, y=False, b=False) == []
    assert buttons.sample(x=False, a=False, y=False, b=False) == []
    assert buttons.sample(x=False, a=False, y=False, b=True) == []
    assert buttons.sample(x=False, a=False, y=False, b=True) == []
    assert buttons.sample(x=False, a=False, y=False, b=False) == ["toggle"]
    assert buttons.sample(x=False, a=False, y=False, b=True) == []
    assert buttons.sample(x=False, a=False, y=False, b=False) == ["toggle"]
    assert [e["kind"] for e in buttons.events] == ["toggle", "toggle"]


def test_y_hold_discards_once_and_a_pauses_until_x_plus_a_resumes():
    buttons = EpisodeButtons(hold_ticks=3)
    buttons.sample(x=False, a=False, y=False, b=False)
    assert buttons.sample(x=False, a=False, y=True, b=False) == []
    assert buttons.sample(x=False, a=False, y=True, b=False) == []
    assert buttons.sample(x=False, a=False, y=True, b=False) == ["discard"]
    assert buttons.paused is True
    assert buttons.sample(x=False, a=False, y=True, b=False) == []
    buttons.sample(x=False, a=False, y=False, b=False)
    for _ in range(3):
        buttons.sample(x=True, a=True, y=False, b=False)
    buttons.sample(x=False, a=False, y=False, b=False)
    assert buttons.sample(x=False, a=True, y=False, b=False) == ["pause"]
    assert buttons.paused is True
    assert buttons.sample(x=False, a=True, y=False, b=True) == []
    buttons.sample(x=False, a=False, y=False, b=False)
    assert buttons.sample(x=True, a=True, y=False, b=False) == []
    assert buttons.sample(x=True, a=True, y=False, b=False) == []
    assert buttons.sample(x=True, a=True, y=False, b=False) == ["resume"]
    assert buttons.paused is False
    assert [e["seq"] for e in buttons.events] == [1, 2, 3, 4]


def test_b_hold_ends_without_saving_an_extra_take():
    buttons = EpisodeButtons(hold_ticks=3)
    for _ in range(14):
        assert buttons.sample(x=False, a=False, y=False, b=True) == []
    assert buttons.sample(x=False, a=False, y=False, b=True) == ["end"]
    assert buttons.sample(x=False, a=False, y=False, b=False) == []


def test_b_save_pauses_following_immediately_but_b_start_does_not():
    buttons = EpisodeButtons(hold_ticks=3)
    buttons.sample(x=False, a=False, y=False, b=True, recording=False)
    assert buttons.sample(x=False, a=False, y=False, b=False, recording=False) == ["toggle"]
    assert buttons.paused is False
    buttons.sample(x=False, a=False, y=False, b=True, recording=True)
    assert buttons.sample(x=False, a=False, y=False, b=False, recording=True) == ["toggle"]
    assert buttons.paused is True
