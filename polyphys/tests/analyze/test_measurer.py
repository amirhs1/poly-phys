import pytest
import numpy as np
from numpy.testing import assert_array_equal

from polyphys.analyze.measurer import (
    apply_pbc_orthogonal,
    pair_distance,
    transverse_size,
    max_distance,
    fsd,
    simple_stats,
    sem,
    spherical_segment,
    sphere_sphere_intersection,
    create_bin_edge_and_hist,
    fixedsize_bins,
    radial_cyl_histogram,
    axial_histogram,
    radial_histogram,
    azimuth_cyl_histogram,
    planar_cartesian_histogram
)


@pytest.mark.parametrize(
    "pbc,expected_lengths,expected_inv",
    [
        ({0: 10.0, 2: 20.0}, [10.0, 0.0, 20.0], [0.1, 0.0, 0.05]),
    ],
)
def test_apply_pbc_orthogonal(pbc, expected_lengths, expected_inv):
    lengths = np.zeros(3)
    inv = np.zeros(3)
    updated, updated_inv = apply_pbc_orthogonal(lengths, inv, pbc)
    assert_array_equal(updated, expected_lengths)
    assert_array_equal(updated_inv, expected_inv)


def test_apply_pbc_zero_length_raises():
    with pytest.raises(ZeroDivisionError):
        apply_pbc_orthogonal(np.zeros(3), np.zeros(3), {1: 0.0})


def test_apply_pbc_negative_length_raises():
    with pytest.raises(ValueError):
        apply_pbc_orthogonal(np.zeros(3), np.zeros(3), {1: -5.0})


def test_pair_distance_without_pbc():
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 3.0]])
    result = pair_distance(pos)
    assert_array_equal(result, np.array([3.0, 0.0, 0.0]))


def test_pair_distance_invalid_shape_raises():
    with pytest.raises(ValueError):
        pair_distance(np.array([[1.0, 2.0, 3.0]]))  # only one position


def test_transverse_size():
    pos = np.array([[1, 0, 0], [1, 2, 2]])
    result = transverse_size(pos, axis=0)
    assert np.isclose(result, 2.82842712)


def test_transverse_size_invalid_axis():
    pos = np.array([[1, 0, 0], [1, 2, 2]])
    with pytest.raises(IndexError):
        transverse_size(pos, axis=3)


def test_max_distance():
    pos = np.array([[0, 0, 0], [4, 2, 6]])
    result = max_distance(pos)
    assert_array_equal(result, np.array([4.0, 2.0, 6.0]))


def test_fsd():
    pos = np.array([[1, 2, 3], [4, 8, 6]])
    result = fsd(pos, axis=1)
    assert result == 6.0


def test_fsd_invalid_axis():
    pos = np.array([[1, 2, 3]])
    with pytest.raises(IndexError):
        fsd(pos, axis=5)


def test_simple_stats_valid():
    data = np.array([1.0, 2.0, 3.0])
    result = simple_stats("energy", data)
    assert result["energy_mean"] == 2.0
    assert result["energy_var"] == 1.0
    assert result["energy_sem"] == pytest.approx(0.57735, rel=1e-5)


def test_simple_stats_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        simple_stats("test", np.array([]))


def test_sem_valid():
    result = sem(np.array([1.0, 2.0, 3.0]))
    assert result == pytest.approx(0.57735, rel=1e-5)


def test_sem_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        sem(np.array([]))


def test_spherical_segment_basic():
    assert spherical_segment(3, 1, 2) == \
        pytest.approx(20.94395102393195, rel=1e-5)


def test_spherical_segment_full_sphere():
    assert spherical_segment(3, -3, 3) == \
        pytest.approx(113.09733552923255, rel=1e-5)


def test_spherical_segment_negative_radius():
    with pytest.raises(ValueError, match="must be positive"):
        spherical_segment(-3, 1, 2)


def test_sphere_sphere_intersection_partial_overlap():
    assert sphere_sphere_intersection(3, 4, 2) == \
        pytest.approx(94.90227807719167, rel=1e-5)


def test_sphere_sphere_intersection_no_overlap():
    assert sphere_sphere_intersection(3, 4, 10) == 0.0


def test_sphere_sphere_intersection_fully_contained():
    assert sphere_sphere_intersection(3, 4, 1) == \
        pytest.approx(113.0973, rel=1e-5)


def test_sphere_sphere_intersection_zero_radius():
    assert sphere_sphere_intersection(3, 0, 1) == 0.0
    assert sphere_sphere_intersection(0, 4, 1) == 0.0


def test_create_bin_edge_and_hist():
    edges, hist = create_bin_edge_and_hist(1.0, 0.0, 5.0)
    assert_array_equal(edges, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert_array_equal(hist, np.zeros(5, dtype=np.int16))


def test_create_bin_edge_and_hist_invalid_range():
    with pytest.raises(ValueError):
        create_bin_edge_and_hist(1.0, 5.0, 0.0)


def test_radial_histogram():
    pos = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    edges = np.array([0, 1, 2, 3])
    result = radial_histogram(pos, edges, (0, 3))
    assert_array_equal(result, np.array([0, 2, 1]))


def test_radial_histogram_invalid_input():
    with pytest.raises(ValueError):
        radial_histogram(np.array([1, 2, 3]), np.array([0, 1]), (0, 1))


@pytest.mark.parametrize(
    "bin_type,expected_n_bins",
    [("ordinary", 5), ("nonnegative", 5), ("periodic", 5)],
)
def test_fixedsize_bins(bin_type, expected_n_bins):
    result = fixedsize_bins(1.0, 0.0, 5.0, bin_type=bin_type)
    assert result["n_bins"] == expected_n_bins
    assert len(result["bin_edges"]) == expected_n_bins + 1


def test_radial_cyl_histogram_valid():
    positions = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3], [0, 3, 4]])
    edges = np.array([0, 1, 2, 3, 4])
    bin_range = (0, 4)

    # Test dim=1 -> use x and z
    hist = radial_cyl_histogram(positions, edges, bin_range, dim=1)
    assert hist.tolist() == [1, 1, 0, 2]


def test_radial_cyl_histogram_invalid_dim():
    pos = np.random.rand(3, 3)
    with pytest.raises(ValueError, match="must be one of"):
        radial_cyl_histogram(pos, np.array([0, 1]), (0, 1), dim=5)


def test_radial_cyl_histogram_invalid_ndim():
    pos = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="must be a 2D array"):
        radial_cyl_histogram(pos, np.array([0, 1]), (0, 1), dim=1)


def test_axial_histogram_valid():
    positions = np.array([
        [1, 2, 3],
        [2, 2, 3],
        [3, 2, 3],
        [4, 2, 3]
    ])
    edges = np.array([0, 1, 2, 3, 4, 5])
    hist = axial_histogram(positions, edges, (0, 5), dim=0)
    assert hist.tolist() == [0, 1, 1, 1, 1]


def test_axial_histogram_invalid_dim():
    pos = np.random.rand(3, 3)
    with pytest.raises(ValueError, match="must be one of"):
        axial_histogram(pos, np.array([0, 1]), (0, 1), dim=4)


def test_axial_histogram_invalid_ndim():
    pos = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="must be a 2D array"):
        axial_histogram(pos, np.array([0, 1]), (0, 1), dim=0)


def test_azimuth_cyl_histogram():
    pos = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])  # x-y plane
    edges = np.linspace(-np.pi, np.pi, 5)
    result = azimuth_cyl_histogram(pos, edges, (-np.pi, np.pi), dim=2)
    assert result.sum() == 3


def test_planar_cartesian_histogram_normal_case():
    positions = np.array([
        [1, 1, 0],  # falls into bin (0,0)
        [2, 2, 0],  # falls into bin (1,1)
        [3, 3, 0]   # falls into bin (1,1)
    ])
    edges = [np.array([0, 2, 4]), np.array([0, 2, 4])]
    bin_ranges = [(0, 4), (0, 4)]

    # Project onto plane perpendicular to z-axis (dim=2), i.e. use x and y
    hist = planar_cartesian_histogram(positions, edges, bin_ranges, dim=2)

    expected = np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ])
    np.testing.assert_array_equal(hist, expected)
