from kanop.simulation import simulate_gbm_paths


def test_simulation_shape_and_initial_value():
    paths, times = simulate_gbm_paths(
        s0=4.0,
        maturity_years=50 / 252,
        r=0.0,
        sigma=0.2,
        n_steps=50,
        n_paths=100,
        seed=1,
    )
    assert paths.shape == (100, 51)
    assert times.shape == (51,)
    assert (paths[:, 0] == 4.0).all()
