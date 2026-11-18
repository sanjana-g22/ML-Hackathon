import numpy as np
import pandas as pd
import pytest
from src.data_loader import TripData
from src.member1_io import load_trip
from src.network import build_network, build_graph, segment_projection
from src.features import make_features, FEATURE_COLUMNS
from src.labels import align_reference
from src.member1_pipeline import prepare_trip


def trip_fixture(route=(0, 1, 2, 0)):
    nodes = pd.DataFrame({'longitude':[77.,77.001,77.002], 'latitude':[13.,13.,13.]})
    arcs = pd.DataFrame({'from_node_id':[0,1,2,0], 'to_node_id':[1,2,0,1]})
    track = pd.DataFrame({'longitude':[77.0002,77.0008,77.0018,77.001,77.0002,77.0008],
                          'latitude':[13.]*6,'timestamp':np.arange(6,dtype=float)})
    return TripData('synthetic','train',track,nodes,arcs,pd.DataFrame({'arc_id':list(route)}))


def test_metric_projection_and_parallel_edges():
    t=trip_fixture(); n=build_network(t)
    assert 100 < n.lengths[0] < 120
    assert len(n.arc_ids)==4
    g=build_graph(n)
    assert np.isclose(g[0,1],n.lengths[0])
    distance,ratio=segment_projection(np.array([.5,1.]),np.array([0.,0.]),np.array([1.,0.]))
    assert np.isclose(distance,1) and np.isclose(ratio,.5)


def test_features_do_not_depend_on_route():
    t=trip_fixture(); n=build_network(t)
    a,_,_=make_features(t,n,50,10)
    t.route=pd.DataFrame({'arc_id':[3,2,1]})
    b,_,_=make_features(t,n,50,10)
    pd.testing.assert_frame_equal(a,b)
    assert not {'label','trip_id','arc_id','route_position'} & set(FEATURE_COLUMNS)


def test_alignment_order_repeated_arcs():
    t=trip_fixture(); n=build_network(t)
    _,xy,valid=make_features(t,n)
    labels=align_reference(t,n,xy,valid)
    assert np.all(np.diff(labels.route_position)>=0)
    assert labels.reference_arc_id.isin(t.route.arc_id).all()
    assert labels.route_position.iloc[-1] == 3


def test_missing_candidates_preserved():
    t=trip_fixture()
    t.track.loc[2,'latitude']+=.01
    c,o,_=prepare_trip(t,50,10)
    p=c[['split','trip_id','obs_idx','arc_id']].copy(); p['prob']=c.label*.98+.01
    assert o.loc[2,'n_candidates']==0
    assert len(o)==6 and not o.loc[2,'candidate_hit']


def test_nonfinite_and_duplicate_times():
    t=trip_fixture(); t.track.loc[1,'timestamp']=0; t.track.loc[2,'longitude']=np.nan
    c,o,_=prepare_trip(t,50,10)
    assert not np.isinf(c[FEATURE_COLUMNS].to_numpy()).any()
    assert not o.loc[2,'valid_coordinate']
    assert not o.loc[2,'reference_available']
    assert c.loc[c.obs_idx==1,'time_valid'].eq(0).all()


def test_default_loader_never_reads_reference(tmp_path):
    d=tmp_path/'train'/'trip';d.mkdir(parents=True)
    (d/'trip.track').write_text('77 13 0\n77.001 13 1\n')
    (d/'trip.nodes').write_text('77 13\n77.001 13\n')
    (d/'trip.arcs').write_text('0 1\n')
    (d/'trip.route').write_text('THIS MUST NOT BE PARSED')
    assert load_trip('trip','train',str(tmp_path)).route is None
    with pytest.raises(ValueError):
        load_trip('trip','train',str(tmp_path),load_reference=True)


def test_no_reference_inference_and_empty_candidates():
    t=trip_fixture();t.route=None;t.track.latitude=0.
    c,_,valid=make_features(t,build_network(t),1,1)
    assert c.empty and valid.all()
    assert set(FEATURE_COLUMNS)<=set(c.columns)



def check_exports(data):
    from pathlib import Path
    import json
    root = Path(data)
    config = json.loads((root / 'feature_config.json').read_text())
    assert config['features'] == FEATURE_COLUMNS, 'Feature order mismatch'
    c = pd.read_parquet(root / 'candidates.parquet')
    o = pd.read_parquet(root / 'observations.parquet')
    keys = ['split','trip_id','obs_idx']
    assert not c.duplicated(keys+['arc_id']).any(), 'Duplicate candidate keys'
    assert not o.duplicated(keys).any(), 'Duplicate observations'
    assert set(o.split) == {'train','student_test'}, 'Unexpected partition'
    assert o[o.split=='train'].trip_id.nunique() == 80, 'Expected all 80 training trips'
    assert o[o.split=='student_test'].trip_id.nunique() == 10, 'Expected all 10 student-test trips'
    assert not np.isinf(c[FEATURE_COLUMNS].to_numpy()).any(), 'Infinite features'
    assert c.label.isin([-1,0,1]).all(), 'Invalid labels'
    counts = c.groupby(keys).size().rename('actual')
    joined = o.join(counts, on=keys)
    assert (joined.actual.fillna(0) == joined.n_candidates).all(), 'Candidate count mismatch'
    positives = c[c.label==1].groupby(keys).size()
    assert positives.max() <= 1, 'More than one positive per observation'
    lookup = o.set_index(keys)
    refs = c.set_index(keys).index.map(lookup.reference_arc_id)
    known = c.set_index(keys).index.map(lookup.reference_available).to_numpy(bool)
    expected = np.where(known, c.arc_id.to_numpy()==np.asarray(refs), -1)
    np.testing.assert_array_equal(c.label.to_numpy(), expected)
    folds = pd.read_csv(root/'folds.csv',dtype={'trip_id':str})
    assert not folds.trip_id.duplicated().any(), 'Trip assigned to multiple folds'
    assert set(folds.trip_id) == set(c[c.split=='train'].trip_id), 'Fold manifest mismatch'
    assert set(folds.fold) == set(range(5)), 'Expected five folds'
    for (split,tid), part in o.groupby(['split','trip_id']):
        positions = part.loc[part.reference_available,'route_position'].to_numpy()
        assert (np.diff(positions) >= 0).all(), f'Non-monotone alignment: {split}/{tid}'
    print(f'PASS: 80 training + 10 student-test trips; {len(c):,} candidates; 14 features; labels and five trip folds verified.')
    print(o.groupby('split').agg(observations=('obs_idx','size'), candidate_miss_rate=('candidate_hit',lambda x:1-x.mean())).to_string())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='results/member1/data')
    check_exports(parser.parse_args().data)
