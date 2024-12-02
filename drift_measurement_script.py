#!/usr/bin/python

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy
import yaml
import time
from tqdm import tqdm

from erpe.analysis import *
from erpe.experiment_design import *
from erpe.models import *
from erpe.qcal_util import *
from erpe.lqr import *

import plotly.io as pio
pio.renderers.default = 'jupyterlab'


import qcal as qc

from qcal.units import *
from qcal.utils import load_from_pickle
from qcal.backend.qubic.qpu import QubicQPU
from qcal.backend.qubic.utils import qubic_sequence

from qcal.benchmarking.readout import ReadoutFidelity
from qcal.calibration.readout import ReadoutCalibration

from qcal.interface.pygsti.circuits import load_circuits
from qcal.interface.pygsti.transpiler import Transpiler
from qcal.interface.pygsti.datasets import generate_pygsti_dataset
import qcal.settings as settings

from qcal.characterization.coherence import T1, T2

def run_ramsey_rpe(qid, config, depths, num_shots_per_circuit, classifier):
    edesign_ramsey = EDesign_Ramsey(depths, [qid])
    ds_0_ramsey = make_dataset(config, edesign_ramsey, num_shots_per_circuit, classifier)
    analysis_ramsey0 = Analysis_Ramsey(ds_0_ramsey, edesign_ramsey)
    return ds_0_ramsey, analysis_ramsey0

def run_xgate_rpe(qid, config, depths, num_shots_per_circuit, classifier):
    edesign_xgate = EDesign_Xgate(depths, [qid])
    ds_0_xgate = make_dataset(config, edesign_xgate, num_shots_per_circuit, classifier)
    analysis_xgate0 = Analysis_Xgate(ds_0_xgate, edesign_xgate)
    return ds_0_xgate, analysis_xgate0

def run_cz_rpe(qids, config, depths, num_shots_per_circuit, classifier):
    edesign_cz = EDesign_CZ(depths, qids)
    ds_0_cz = make_dataset(config, edesign_cz, num_shots_per_circuit, classifier)
    analysis_cz0 = Analysis_CZ(ds_0_cz, edesign_cz)
    return ds_0_cz, analysis_cz0

def run_t1(qid, config):
    qubits = (int(qid[1]),)
    print(qubits)
    n_elements = 50
    char1 = T1(
        QubicQPU,
        config,
        qubits,
        t_max=500*us,
        classifier=classifier,
        n_elements=n_elements,
        n_circs_per_seq=n_elements,
        reload_freq=False,
        reload_env=False,
        zero_between_reload=False
    )
    char1.run()
    res = char1._char_values[int(qid[1])]
    return res

def run_t2(qid, config):
    qubits = (int(qid[1]),)
    print(qubits)
    n_elements = 50
    char = T2(
        QubicQPU,
        config,
        qubits,
        t_max=300*us,
        echo=False,
        detuning=25*kHz,
        classifier=classifier,
        n_elements=n_elements,
        n_circs_per_seq=n_elements,
        reload_freq=False,
        reload_env=False,
        zero_between_reload=False
    )
    char.run()
    res = char._char_values[int(qid[1])]
    return res

basedir = '/home/jpmarceaux/experiment/'
settings.Settings.config_path = basedir + 'config/'
settings.Settings.data_path = basedir + 'data/'
settings.Settings.save_data = True
akel_config = qc.Config(basedir + 'config/configs/X6Y3/config.yaml')
classifier = load_from_pickle(basedir + 'config/configs/X6Y3/ClassificationManager.pkl')

# long-term data collection
qids_1qb = ['Q0', 'Q1', 'Q2']
qids_CZ = [('Q0', 'Q1'), ('Q1', 'Q2')]
depths_ramsey = [2**i for i in range(12)]
depths_xgate = [2**i for i in range(9)]
depths_cz = [2**i for i in range(6)]
num_shots_per_circuit = 1000

# load a pandas df of the data
df_1qb = pd.read_csv('/home/jpmarceaux/longtime_stability/drift_characterization_experiments/weekend_nov8/single_qb_data.csv')
df_2qb = pd.read_csv('/home/jpmarceaux/longtime_stability/drift_characterization_experiments/weekend_nov8/two_qb_data.csv')


ramsey_estimates = {}
xgate_estimates = {}
t1s = {}
t2s = {}

timestamps_1qb = {}

for qid in qids_1qb:
    timestamp = time.time()
    ds_ramsey, analysis_ramsey = run_ramsey_rpe(qid, akel_config, depths_ramsey, 1000, classifier)
    pygsti.io.write_dataset(f'/home/jpmarceaux/longtime_stability/drift_characterization_experiments/weekend_nov8/{qid}_ramsey_{timestamp}.txt', ds_ramsey)
    # pickle the anlaysis 
    with open(f'/home/jpmarceaux/longtime_stability/drift_characterization_experiments/weekend_nov8/{qid}_ramsey_{timestamp}.pkl', 'wb') as f:
        pickle.dump(analysis_ramsey, f)
    # add results to dataframe
    ramsey_estimates[qid] = analysis_ramsey.estimates['idle']

    ds_xgate, analysis_xgate = run_xgate_rpe(qid, akel_config, depths_xgate, 1000, classifier)
    pygsti.io.write_dataset(f'/home/jpmarceaux/longtime_stability/drift_characterization_experiments/weekend_nov8/{qid}_xgate_{timestamp}.txt', ds_xgate)
    # pickle the anlaysis 
    with open(f'/home/jpmarceaux/longtime_stability/drift_characterization_experiments/weekend_nov8/{qid}_xgate_{timestamp}.pkl', 'wb') as f:
        pickle.dump(analysis_xgate, f)

    # rune T1 and T2
    t1 = run_t1(qid, akel_config)
    t1s[qid] = t1
    t2 = run_t2(qid, akel_config)
    t2s[qid] = t2
    xgate_estimates[qid] = [analysis_xgate.estimates['X overrot'], analysis_xgate.estimates['X axis']]
    timestamps_1qb[qid] = timestamp

# add the results to the dataframe 
data = [{   
        'qid' : q, 
        'timestamp': timestamps_1qb[q],
        'ramsey_estimate' : ramsey_estimates[q],
        'overrot' : xgate_estimates[q][0],
        'axis' : xgate_estimates[q][1],
        'T1' : t1s[q],
        'T2' : t2s[q]
        } for q in qids_1qb]
# add the data to the dataframe
df_1qb = df_1qb._append(data, ignore_index=True)
df_1qb.to_csv('/home/jpmarceaux/longtime_stability/drift_characterization_experiments/weekend_nov8/single_qb_data.csv', index=False)

cz_estimates = {}
# run cz on the qubits
for qids in qids_CZ:
    timestamp = time.time()
    ds_cz, analysis_cz = run_cz_rpe(qids, akel_config, depths_cz, 1000, classifier)
    pygsti.io.write_dataset(f'/home/jpmarceaux/longtime_stability/drift_characterization_experiments/weekend_nov8/{qids[0]}_{qids[1]}_cz_{timestamp}.txt', ds_cz)
    # pickle the anlaysis 
    with open(f'/home/jpmarceaux/longtime_stability/drift_characterization_experiments/weekend_nov8/{qids[0]}_{qids[1]}_cz_{timestamp}.pkl', 'wb') as f:
        pickle.dump(analysis_cz, f)
    cz_estimates[qids] = [analysis_cz.estimates['ZZ'], analysis_cz.estimates['IZ'], analysis_cz.estimates['ZI']]
# add the results to the dataframe
data = [{   
        'timestamp': timestamp,
        'qids' : q, 
        'ZZ' : cz_estimates[q][0],
        'IZ' : cz_estimates[q][1],
        'ZI' : cz_estimates[q][2]
        } for q in qids_CZ]
# add the data to the dataframe
df_2qb = df_2qb._append(data, ignore_index=True)
df_2qb.to_csv('/home/jpmarceaux/longtime_stability/drift_characterization_experiments/weekend_nov8/two_qb_data.csv', index=False)




