from tools.validate_application import summarize


def row(available_correct, occupied_correct):
    return dict(true_bays=100, matched_bays=100, predicted_bays=100,
                occupied_truth=50, available_truth=50,
                occupied_correct=occupied_correct, available_correct=available_correct,
                extra_bays=0, false_available=0)


def test_validation_counts_abstentions_as_failures():
    counts, metrics, passed = summarize([row(20, 50)] * 3)
    assert metrics['joint_bay_and_occupancy_accuracy'] == .7
    assert metrics['available_recall'] == .4
    assert not passed


def test_validation_does_not_hide_a_failed_class_in_average():
    _, metrics, passed = summarize([row(35, 50)] * 3)
    assert metrics['joint_bay_and_occupancy_accuracy'] == .85
    assert not passed


def test_validation_accepts_both_classes_above_target():
    _, metrics, passed = summarize([row(45, 48)] * 3)
    assert metrics['photo_cluster_bootstrap_95_interval'] == [.93, .93]
    assert passed
