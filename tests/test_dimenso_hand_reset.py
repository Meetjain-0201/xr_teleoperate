from teleop.utils.dimenso_hand_reset import release_brainco_at_home


def test_release_only_when_brainco_quest_arms_are_home_between_takes():
    args = {"end_effector": "brainco", "input_mode": "controller"}
    assert not release_brainco_at_home(phase="recording", home_ready=True, **args)
    assert not release_brainco_at_home(phase="awaiting_reset", home_ready=False, **args)
    assert release_brainco_at_home(phase="awaiting_reset", home_ready=True, **args)
    assert release_brainco_at_home(phase="complete", home_ready=True, **args)
    assert not release_brainco_at_home(phase="awaiting_reset", home_ready=True,
                                      end_effector="inspire_dfx", input_mode="controller")
