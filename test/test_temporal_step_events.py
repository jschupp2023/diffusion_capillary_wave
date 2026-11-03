"""Small known-truth checks for the exploratory temporal-event diagnostic."""
import unittest
import numpy as np
from data_analysis.analyze_temporal_step_events import local_fit_evidence, innovation_evidence


class TemporalEventTests(unittest.TestCase):
    def test_step_on_nonzero_physical_trend(self):
        q=1.9969696
        t=np.arange(200)
        clean=.1*t
        observed=clean.copy()
        observed[100:]+=5*q
        candidates=np.zeros(200,dtype=int)
        candidates[100]=5
        regional=np.repeat(observed[:,None],4,axis=1)
        support,_=local_fit_evidence(regional,candidates,q)
        self.assertTrue(np.all(support[100]==4))
        self.assertTrue(np.all(innovation_evidence(regional,candidates,q)[100]==4))

    def test_smooth_trend_rejects_spurious_step_hypothesis(self):
        q=1.9969696
        regional=np.repeat((.1*np.arange(200))[:,None],4,axis=1)
        candidates=np.zeros(200,dtype=int)
        candidates[100]=5
        support,_=local_fit_evidence(regional,candidates,q)
        self.assertFalse(np.any(support))
        self.assertFalse(np.any(innovation_evidence(regional,candidates,q)))

    def test_one_region_step_is_not_global_consensus(self):
        q=1.9969696
        regional=np.repeat((.1*np.arange(200))[:,None],4,axis=1)
        regional[100:,0]+=5*q
        candidates=np.zeros(200,dtype=int)
        candidates[100]=5
        support,_=local_fit_evidence(regional,candidates,q)
        self.assertTrue(np.all(support[100]==1))


if __name__=='__main__':unittest.main()
