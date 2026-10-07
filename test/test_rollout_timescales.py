import numpy as np

from data_analysis.correlations.compare_rollout_timescales import (
    lag_statistics, persistent_crossing, reference_controls, summarize_population)


def test_fft_structure_matches_direct_pairs_and_acf_convention():
    x=np.random.default_rng(18).normal(size=(3,97,2)).cumsum(axis=1)
    x+=np.array([1e5,-2e5])
    acf,structure=lag_statistics(x,30)
    centered=x-x.mean(axis=1,keepdims=True)
    for lag in (0,1,2,15,30):
        expected=np.zeros((3,2)) if lag==0 else np.mean((x[:,lag:]-x[:,:-lag])**2,axis=1)
        np.testing.assert_allclose(structure[:,lag],expected,atol=1e-10)
        numerator=(centered*centered).sum(axis=1) if lag==0 else (centered[:,lag:]*centered[:,:-lag]).sum(axis=1)
        np.testing.assert_allclose(acf[:,lag],numerator/(centered*centered).sum(axis=1),atol=1e-12)


def test_periodogram_closure_and_slow_large_motion_small_increments():
    time=np.arange(4096)
    slow=2*np.sin(2*np.pi*time/512)
    fast=np.sin(2*np.pi*time/64)
    result=summarize_population(np.stack((slow,fast),axis=-1)[None],1000,100,2048)
    np.testing.assert_allclose(result['periodogram_variance'],[2,.5],atol=1e-12)
    assert result['position_rms'][0]>result['position_rms'][1]
    assert result['velocity_rms'][0]<result['velocity_rms'][1]


def test_persistent_crossing_requires_complete_run():
    assert persistent_crossing([1,.001,.5,.001,.001],consecutive=2)==3
    assert persistent_crossing([1,.001,.5,.001],consecutive=2) is None


def test_trim_respects_all_coordinates_and_intervening_stencil():
    increments=np.array([[1,1],[2,1],[3,100],[4,1],[5,1],[6,1],[7,1]],dtype=float)
    reference=np.concatenate((np.zeros((1,2)),np.cumsum(increments,axis=0)))[None]
    spec=dict(state_std=np.ones(2),cutoff=10.)
    config={'model':{'history_steps':1},'training':{'multistep_weight':.1,'multistep_horizons':[2]}}
    controls,arrays=reference_controls(reference,spec,config,[0],[1,2])
    assert controls['edge_retained_fraction']==6/7
    # H=2 windows with t=0,1,2 cross bad edge; only t=3,4 remain.
    np.testing.assert_allclose(controls['training_window_current_increment_rms'],[np.sqrt((4**2+5**2)/2)])
    np.testing.assert_array_equal(arrays['trim_velocity_pair_counts'],[4,2])
