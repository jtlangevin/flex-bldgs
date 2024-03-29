#!/usr/bin/env python3
import pymc3 as pm
import theano as tt
from theano.sandbox.rng_mrg import MRG_RandomStream
import numpy as np
import pandas as pd
# from numpy.polynomial.polynomial import polyfit
from scipy.special import softmax
from scipy import stats
from os import getcwd, path, remove
from argparse import ArgumentParser
import pickle
import arviz as az
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import json
from pymc3.exceptions import SamplingError
from matplotlib.ticker import FormatStrFormatter
tt.config.compute_value = "ignore"


class UsefulFilesVars(object):
    """Summarize key model input, estimation, and output information.

    Attributes:
        coefs (tuple): Path to CSV file with regression coefficients
        coef_names_dtypes (list): Variable names/formats for coefficients CSV
        mod_dict (dict): Dict with information on input data, variables, and
            output plotting file names for each model type.
        predict_out (JSON): Output JSON with strategy recommendation pcts.
    """

    def __init__(self, bldg_type_vint, mod_init, mod_est, mod_assess):
        """Initialize class attributes."""

        # Initialize all data input variables as None
        dmd_bl_dat, stored_dmd_bl, dmd_therm_dat, dmd_ntherm_dat, tmp_dat, \
            dmd_tmp_dat, stored_tmp, stored_dmd_therm, stored_dmd_ntherm, \
            stored_lt, pc_dmd_dat, stored_pc_dmd = (None for n in range(12))

        # Set data input and output files for all models
        # Medium office, >=2004 vintage
        if bldg_type_vint == "mediumofficenew":
            # Handle data inputs differently for model initialization vs.
            # model re-estimation and prediction (the former uses different
            # CSVs for each building type, while the latter will only draw
            # from one CSV)
            if mod_init is True or (mod_assess is True and mod_est is False):
                dmd_bl_dat = ("data", "MO_B.csv")
                dmd_therm_dat = ("data", "MO_Thermal_Demand_new.csv")
                dmd_ntherm_dat = ("data", "MO_Nonthermal_Demand_new.csv")
                tmp_dat = ("data", "MO_Temperature_new.csv")
                pc_dmd_dat = ("data", "MO_PC_Demand_new.csv")
            elif mod_est is True:
                dmd_bl_dat = ("data", "test_update_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_update.csv") for n in range(4))
            else:
                dmd_bl_dat = ("data", "test_predict_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_predict.csv") for n in range(4))
            # Set stored model data files
            stored_dmd_bl = ("model_stored", "dmd_mo_b.pkl")
            stored_dmd_therm = ("model_stored", "dmd_therm_mo_n.pkl")
            stored_dmd_ntherm = ("model_stored", "dmd_ntherm_mo_n.pkl")
            stored_tmp = ("model_stored", "tmp_mo_n.pkl")
            stored_pc_dmd = ("model_stored", "pc_dmd_mo_n.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_mo_n.csv")
        # Medium office, <2004 vintage
        elif bldg_type_vint == "mediumofficeold":
            if mod_init is True or (mod_assess is True and mod_est is False):
                dmd_bl_dat = ("data", "MO_B.csv")
                dmd_therm_dat = ("data", "MO_Thermal_Demand_old.csv")
                dmd_ntherm_dat = ("data", "MO_Nonthermal_Demand_old.csv")
                tmp_dat = ("data", "MO_Temperature_old.csv")
                pc_dmd_dat = ("data", "MO_PC_Demand_old.csv")
            elif mod_est is True:
                dmd_bl_dat = ("data", "test_update_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_update.csv") for n in range(4))
            else:
                dmd_bl_dat = ("data", "test_predict_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_predict.csv") for n in range(4))
            # Set stored model data files
            stored_dmd_bl = ("model_stored", "dmd_mo_b.pkl")
            stored_dmd_therm = ("model_stored", "dmd_therm_mo_o.pkl")
            stored_dmd_ntherm = ("model_stored", "dmd_ntherm_mo_o.pkl")
            stored_tmp = ("model_stored", "tmp_mo_o.pkl")
            stored_pc_dmd = ("model_stored", "pc_dmd_mo_o.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_mo_o.csv")
        # Retail, >=2004 vintage
        elif bldg_type_vint == "stdaloneretailnew":
            if mod_init is True or (mod_assess is True and mod_est is False):
                dmd_bl_dat = ("data", "SR_B.csv")
                dmd_therm_dat = ("data", "SR_Thermal_Demand_new.csv")
                dmd_ntherm_dat = ("data", "SR_Nonthermal_Demand_new.csv")
                tmp_dat = ("data", "SR_Temperature_new.csv")
                pc_dmd_dat = ("data", "SR_PC_Demand_new.csv")
            elif mod_est is True:
                dmd_bl_dat = ("data", "test_update_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_update.csv") for n in range(4))
            else:
                dmd_bl_dat = ("data", "test_predict_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_predict.csv") for n in range(4))
            # Set stored model data files
            stored_dmd_bl = ("model_stored", "dmd_saret_b.pkl")
            stored_dmd_therm = ("model_stored", "dmd_therm_saret_n.pkl")
            stored_dmd_ntherm = ("model_stored", "dmd_ntherm_saret_n.pkl")
            stored_tmp = ("model_stored", "tmp_saret_n.pkl")
            stored_pc_dmd = ("model_stored", "pc_dmd_saret_n.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_saret_n.csv")
        # Retail, <2004 vintage
        elif bldg_type_vint == "stdaloneretailold":
            if mod_init is True or (mod_assess is True and mod_est is False):
                dmd_bl_dat = ("data", "SR_B.csv")
                dmd_therm_dat = ("data", "SR_Thermal_Demand_old.csv")
                dmd_ntherm_dat = ("data", "SR_Nonthermal_Demand_old.csv")
                tmp_dat = ("data", "SR_Temperature_old.csv")
                pc_dmd_dat = ("data", "SR_PC_Demand_old.csv")
            elif mod_est is True:
                dmd_bl_dat = ("data", "test_update_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_update.csv") for n in range(4))
            else:
                dmd_bl_dat = ("data", "test_predict_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_predict.csv") for n in range(4))
            # Set stored model data files
            stored_dmd_bl = ("model_stored", "dmd_saret_b.pkl")
            stored_dmd_therm = ("model_stored", "dmd_therm_saret_o.pkl")
            stored_dmd_ntherm = ("model_stored", "dmd_ntherm_saret_o.pkl")
            stored_tmp = ("model_stored", "tmp_saret_o.pkl")
            stored_pc_dmd = ("model_stored", "pc_dmd_saret_o.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_saret_o.csv")
        # Large office, >=2004 vintage
        elif bldg_type_vint == "largeofficenew":
            if mod_init is True or (mod_assess is True and mod_est is False):
                dmd_bl_dat = ("data", "MO_B.csv")  # ***** UPDATE *****
                dmd_therm_dat = ("data", "LO_Thermal_Demand_new.csv")
                dmd_ntherm_dat = ("data", "LO_Nonthermal_Demand_new.csv")
                tmp_dat = ("data", "LO_Temperature_new.csv")
                pc_dmd_dat = ("data", "LO_PC_Demand_new.csv")
            elif mod_est is True:
                dmd_bl_dat = ("data", "test_update_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_update.csv") for n in range(4))
            else:
                dmd_bl_dat = ("data", "test_predict_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_predict.csv") for n in range(4))
            # Set stored model data files
            stored_dmd_bl = ("model_stored", "dmd_lo_b.pkl")
            stored_dmd_therm = ("model_stored", "dmd_therm_lo_n.pkl")
            stored_dmd_ntherm = ("model_stored", "dmd_ntherm_lo_n.pkl")
            stored_tmp = ("model_stored", "tmp_lo_n.pkl")
            stored_pc_dmd = ("model_stored", "pc_dmd_lo_n.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_lo_n.csv")
        # Large office, <2004 vintage
        elif bldg_type_vint == "largeofficeold":
            if mod_init is True or (mod_assess is True and mod_est is False):
                dmd_bl_dat = ("data", "MO_B.csv")  # ***** UPDATE *****
                dmd_therm_dat = ("data", "LO_Thermal_Demand_old.csv")
                dmd_ntherm_dat = ("data", "LO_Nonthermal_Demand_old.csv")
                tmp_dat = ("data", "LO_Temperature_old.csv")
                pc_dmd_dat = ("data", "LO_PC_Demand_old.csv")
            elif mod_est is True:
                dmd_bl_dat = ("data", "test_update_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_update.csv") for n in range(4))
            else:
                dmd_bl_dat = ("data", "test_predict_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_predict.csv") for n in range(4))
            # Set stored model data files
            stored_dmd_bl = ("model_stored", "dmd_lo_b.pkl")
            stored_dmd_therm = ("model_stored", "dmd_therm_lo_o.pkl")
            stored_dmd_ntherm = ("model_stored", "dmd_ntherm_lo_o.pkl")
            stored_tmp = ("model_stored", "tmp_lo_o.pkl")
            stored_pc_dmd = ("model_stored", "pc_dmd_lo_o.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_lo_o.csv")
        # Large office (all-electric), >=2004 vintage
        elif bldg_type_vint == "largeofficenew_elec":
            if mod_init is True or (mod_assess is True and mod_est is False):
                dmd_bl_dat = ("data", "MO_B.csv")  # ***** UPDATE *****
                dmd_therm_dat = ("data", "LO_allElec_Thermal_Demand_new.csv")
                dmd_ntherm_dat = (
                    "data", "LO_allElec_Nonthermal_Demand_new.csv")
                tmp_dat = ("data", "LO_allElec_Temperature_new.csv")
                pc_dmd_dat = ("data", "LO_allElec_PC_Demand_new.csv")
            elif mod_est is True:
                dmd_bl_dat = ("data", "test_update_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_update.csv") for n in range(4))
            else:
                dmd_bl_dat = ("data", "test_predict_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_predict.csv") for n in range(4))
            # Set stored model data files
            stored_dmd_bl = ("model_stored", "dmd_lo_allelec_b.pkl")
            stored_dmd_therm = ("model_stored", "dmd_therm_lo_allelec_n.pkl")
            stored_dmd_ntherm = ("model_stored", "dmd_ntherm_lo_allelec_n.pkl")
            stored_tmp = ("model_stored", "tmp_lo_allelec_n.pkl")
            stored_pc_dmd = ("model_stored", "pc_dmd_lo_allelec_n.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_lo_elec_n.csv")
        # Large office (all-electric), <2004 vintage
        elif bldg_type_vint == "largeofficeold_elec":
            if mod_init is True or (mod_assess is True and mod_est is False):
                dmd_bl_dat = ("data", "MO_B.csv")  # ***** UPDATE *****
                dmd_therm_dat = ("data", "LO_allElec_Thermal_Demand_old.csv")
                dmd_ntherm_dat = (
                    "data", "LO_allElec_Nonthermal_Demand_old.csv")
                tmp_dat = ("data", "LO_allElec_Temperature_old.csv")
                pc_dmd_dat = ("data", "LO_allElec_PC_Demand_old.csv")
            elif mod_est is True:
                dmd_bl_dat = ("data", "test_update_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_update.csv") for n in range(4))
            else:
                dmd_bl_dat = ("data", "test_predict_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_predict.csv") for n in range(4))
            # Set stored model data files
            stored_dmd_bl = ("model_stored", "dmd_lo_allelec_b.pkl")
            stored_dmd_therm = ("model_stored", "dmd_therm_lo_allelec_o.pkl")
            stored_dmd_ntherm = ("model_stored", "dmd_ntherm_lo_allelec_o.pkl")
            stored_tmp = ("model_stored", "tmp_lo_allelec_o.pkl")
            stored_pc_dmd = ("model_stored", "pc_dmd_lo_allelec_o.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_lo_elec_o.csv")
        # Big box retail, 2004 vintage
        elif bldg_type_vint == "bigboxretail":
            if mod_init is True or (mod_assess is True and mod_est is False):
                dmd_bl_dat = ("data", "SR_B.csv")  # ***** UPDATE *****
                dmd_therm_dat = ("data", "BBR_Thermal_Demand_new.csv")
                dmd_ntherm_dat = ("data", "BBR_Nonthermal_Demand_new.csv")
                tmp_dat = ("data", "BBR_Temperature_new.csv")
                pc_dmd_dat = ("data", "BBR_PC_Demand_new.csv")
            elif mod_est is True:
                dmd_bl_dat = ("data", "test_update_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_update.csv") for n in range(4))
            else:
                dmd_bl_dat = ("data", "test_predict_bl.csv")
                dmd_therm_dat, dmd_ntherm_dat, tmp_dat, pc_dmd_dat = (
                    ("data", "test_predict.csv") for n in range(4))
            # Set stored model data files
            stored_dmd_bl = ("model_stored", "dmd_bbr_b.pkl")
            stored_dmd_therm = ("model_stored", "dmd_therm_bbr_n.pkl")
            stored_dmd_ntherm = ("model_stored", "dmd_ntherm_bbr_n.pkl")
            stored_tmp = ("model_stored", "tmp_bbr_n.pkl")
            stored_pc_dmd = ("model_stored", "pc_dmd_bbr_n.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_bbr_n.csv")

        # Set DCE input/output file names
        dce_dat = ("data", "dce_dat.csv")
        stored_dce = ("model_stored", "dce.pkl")

        # Set data input file column names and data types for model
        # initialization; these are different by model type (though the same
        # for the temperature and demand models, which draw from the same CSV)
        if mod_init is True or (mod_assess is True and mod_est is False):
            dmd_bl_names_dtypes = [
                ('id', 'vintage', 'day_typ', 'hour_number', 'climate',
                 'dmd_sf', 't_out', 'rh_out', 'occ_frac',
                 'v1980', 'v2004', 'v2010', 'v19802004',
                 'cz_2A', 'cz_2B', 'cz_3A', 'cz_3B', 'cz_3C',
                 'cz_4A', 'cz_4B', 'cz_4C', 'cz_5A', 'cz_5B',
                 'cz_6A', 'cz_6B', 'cz_7A'),
                (['<i4'] * 4 + ['<U25'] + ['<f8'] * 4 + ['<i4'] * 17)]
            dmd_tmp_names_dtypes, dmd_ntmp_names_dtypes = ([
                ('vintage', 'climate', 'hour', 't_out', 'rh_out', 'occ_frac',
                 'dmd_delt_sf', 't_in_delt', 'rh_in_delt', 'lt_pwr_delt_pct',
                 'mels_delt_pct', 'tsp_delt', 'tsp_delt_lag',
                 'hrs_since_dr_st', 'hrs_since_dr_end', 'pc_tmp_inc',
                 'pc_length', 'hrs_since_pc_st', 'hrs_since_pc_end'),
                (['<i4'] + ['<U25'] + ['<i4'] + ['<f8'] * 16)] for
                n in range(2))
            self.coef_names_dtypes = [
                ('demand_bl', 'demand_therm', 'demand_ntherm', 'temperature',
                 'demand_precool'), (['<f8'] * 5)]
        # Set data input file column names and data types for model
        # re-estimation; these will be the same across models
        elif mod_est is True:
            dmd_bl_names_dtypes = [
                ('id', 'vintage', 'day_typ', 'day_num', 'hour_number',
                 'climate', 'dmd_sf', 't_out', 'rh_out', 'occ_frac',
                 'v1980', 'v2004', 'v2010', 'v19802004',
                 'cz_2A', 'cz_2B', 'cz_3A', 'cz_3B', 'cz_3C',
                 'cz_4A', 'cz_4B', 'cz_4C', 'cz_5A', 'cz_5B',
                 'cz_6A', 'cz_6B', 'cz_7A'),
                (['<i4'] * 5 + ['<U25'] + ['<f8'] * 4 + ['<i4'] * 17)]
            dmd_tmp_names_dtypes, dmd_ntmp_names_dtypes = ([
                ('vintage', 'climate', 'hour', 't_out', 'rh_out', 'occ_frac',
                 'dmd_delt_sf', 't_in_delt', 'rh_in_delt', 'lt_pwr_delt_pct',
                 'mels_delt_pct', 'tsp_delt', 'tsp_delt_lag',
                 'hrs_since_dr_st', 'hrs_since_dr_end', 'pc_tmp_inc',
                 'pc_length', 'hrs_since_pc_st', 'hrs_since_pc_end'),
                (['<i4'] + ['<U25'] + ['<i4'] + ['<f8'] * 16)] for
                n in range(2))
            self.coef_names_dtypes = [
                ('demand_bl', 'demand_therm', 'demand_ntherm', 'temperature',
                 'demand_precool'), (['<f8'] * 5)]
        # Set data input file column names and data types for model
        # prediction; these will be the same across models
        else:
            dmd_bl_names_dtypes = [
                ('Hr', 'vintage', 'hour_number', 'climate',
                 't_out', 'rh_out', 'occ_frac',
                 'v1980', 'v2004', 'v2010', 'v19802004',
                 'cz_2A', 'cz_2B', 'cz_3A', 'cz_3B', 'cz_3C',
                 'cz_4A', 'cz_4B', 'cz_4C', 'cz_5A', 'cz_5B',
                 'cz_6A', 'cz_6B', 'cz_7A'),
                (['<i4'] + ['<f8'] + ['<i4'] + ['<U25'] + ['<f8'] * 3 +
                 ['<i4'] * 17)]
            dmd_tmp_names_dtypes, dmd_ntmp_names_dtypes = ([(
                    'Name', 'Hr', 't_out', 'rh_out', 'lt_nat',
                    'base_lt_frac', 'occ_frac', 'delt_price_kwh',
                    'hrs_since_dr_st',
                    'hrs_since_dr_end', 'hrs_since_pc_st',
                    'hrs_since_pc_end', 'tsp_delt', 'lt_pwr_delt_pct',
                    'ven_delt_pct', 'mels_delt_pct', 'tsp_delt_lag',
                    'lt_pwr_delt_pct_lag', 'ven_delt_pct_lag',
                    'mels_delt_pct_lag', 'pc_tmp_inc', 'pc_length',
                    'lt_pwr_delt'),
                    (['<U50'] + ['<f8'] * 22)] for n in range(2))
            self.coef_names_dtypes = [
                ('demand_bl', 'demand_therm', 'demand_ntherm', 'temperature',
                 'demand_precool'), (['<f8'] * 5)]

        # Set DCE model column names and data types
        # dce_names_dtypes = [(
        #     'economy', 'pc_tmp', 'tmp', 'lgt', 'daylt', 'choice', 'plug'),
        #     (['<f8'] * 7)]
        dce_names_dtypes = [(
            'economy', 'pc_tmp_high', 'pc_tmp_low', 'tmp', 'lgt', 'daylt',
            'choice', 'plug'), (['<f8'] * 8)]

        # For each model type, store information on input/output data file
        # names, input/output data file formats, model variables, and
        # diagnostic assessment figure file names
        self.mod_dict = {
            "demand_bl": {
                "io_data": [dmd_bl_dat, stored_dmd_bl],
                "io_data_names": dmd_bl_names_dtypes,
                "var_names": ['dmd_bl_params', 'dmd_bl_sd',
                              'Baseline Demand (W/sf)'],
                "fig_names": [
                    "traceplots_dmd_bl.png", "postplots_dmd_bl.png",
                    "ppcheck_dmd_bl.png", "scatter_dmd_bl.png",
                    "update_dmd_bl.png"]
            },
            "demand_therm": {
                "io_data": [dmd_therm_dat, stored_dmd_therm],
                "io_data_names": dmd_tmp_names_dtypes,
                "var_names": [
                    'dmd_therm_params', 'dmd_therm_sd',
                    'Demand Change (W/sf)'],
                "fig_names": [
                    "traceplots_dmd_therm.png", "postplots_dmd_therm.png",
                    "ppcheck_dmd_therm.png", "scatter_dmd_therm.png",
                    "update_dmd_therm.png"]
            },
            "demand_ntherm": {
                "io_data": [dmd_ntherm_dat, stored_dmd_ntherm],
                "io_data_names": dmd_ntmp_names_dtypes,
                "var_names": [
                    'dmd_ntherm_params', 'dmd_ntherm_sd',
                    'Demand Change (W/sf)'],
                "fig_names": [
                    "traceplots_dmd_ntherm.png", "postplots_dmd_ntherm.png",
                    "ppcheck_dmd_ntherm.png", "scatter_dmd_ntherm.png",
                    "update_dmd_ntherm.png"]
            },
            "temperature": {
                "io_data": [tmp_dat, stored_tmp],
                "io_data_names": dmd_tmp_names_dtypes,
                "var_names": [
                    'ta_params', 'ta_sd',
                    'Temperature Change (ºF)'],
                "fig_names": [
                    "traceplots_tmp.png", "postplots_tmp.png",
                    "ppcheck_tmp.png", "scatter_tmp.png",
                    "update_tmp.png"]
            },
            "demand_precool": {
                "io_data": [pc_dmd_dat, stored_pc_dmd],
                "io_data_names": dmd_tmp_names_dtypes,
                "var_names": [
                    'dmd_pc_params', 'dmd_pc_sd',
                    'Demand Change (W/sf)'],
                "fig_names": [
                    "traceplots_dmd_pc.png", "postplots_dmd_pc.png",
                    "ppcheck_dmd_pc.png", "scatter_dmd_pc.png",
                    "update_dmd_pc.png"]
            },
            "choice": {
                "io_data": [dce_dat, stored_dce],
                "io_data_names": dce_names_dtypes,
                "var_names": ['dce_params', 'dce_sd', 'dce'],
                "fig_names": ["traceplots_dce.png", "postplots_dce.png"]
            }

        }
        self.predict_out = ("data", "recommendations.json")


class ModelDataLoad(object):
    """Load the data files needed to initialize, estimate, or run models.

    Attributes:
        dmd_therm (numpy ndarray): Input data for thermal demand
            model initialization.
        dmd_ntherm (numpy ndarray): Input data for non-thermal demand
            model initialization.
        tmp (numpy ndarray): Input data for temperature model initialization.
        pc_dmd (numpy ndarray): Input data for pre-cooling demand model init.
        coefs (tuple): Path to CSV file with regression coefficients for
            use in model re-estimation.
        strategy (numpy ndarray): Strategies covered by prediction input data
        strategy_orig (numpy ndarray): Strategies covered by prediction input
            data before screening for invalid input conditions.
        hr (numpy ndarray): Hours covered by DR model prediction input data.
        hr_orig (numpy ndarray): Covered hours before screening DR prediction
            data for invalid input conditions.
        hr_bl (numpy.ndarray): Hours covered by the baseline model prediction
            data.
        tmp_delt (numpy ndarray): Data used to determine whether
            a measure changes the thermostat set point in a current hour.
        tmp_delt_prev (numpy ndarray): Data used to determine whether
            a measure changed the thermostat set point in a previous hour.
        lgt_delt (numpy ndarray): Data used to determine whether
            a measure changes the lighting setting in a current hour.
        plug_delt (numpy ndarray): Plug load adjustment fractions by DR
            strategy for use in model prediction.
        pc_mag (numpy.array): Precooling magnitude assoc. with each strategy.

        oaf_delt (numpy ndarray): Outdoor air adjustment fractions by DR
            strategy for use in model prediction.
        price_delt (numpy ndarray): $/kWh incentive by DR strategy for use
            in model prediction.

    """

    def __init__(self, handyfilesvars, mod_init, mod_assess,
                 mod_est, update_days):
        """Initialize class attributes."""

        # Initialize OAF delta, plug load delta and price delta as None
        self.coefs, self.oaf_delt, self.plug_delt, self.lgt_delt, \
            self.price_delt = (None for n in range(5))
        # Data read-in for model initialization is specific to each type
        # of model (though demand/temperature share the same input data);
        if mod_init is True or (mod_assess is True and mod_est is False):
            # Read in data for initializing baseline demand model
            self.dmd_bl = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict[
                            "demand_bl"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "demand_bl"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "demand_bl"]["io_data_names"][1])
            # Read in data for initializing thermal demand model
            self.dmd_therm = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict[
                            "demand_therm"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "demand_therm"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "demand_therm"]["io_data_names"][1])
            # Read in data for initializing non-thermal demand model
            self.dmd_ntherm = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict[
                            "demand_ntherm"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "demand_ntherm"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "demand_ntherm"]["io_data_names"][1])
            # Read in data for initializing temperature model
            self.tmp = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict[
                            "temperature"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "temperature"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "temperature"]["io_data_names"][1])
            # Read in data for initializing demand pre-cool model
            self.pc_dmd = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict[
                            "demand_precool"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "demand_precool"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "demand_precool"]["io_data_names"][1])
            # Read in data for initializing choice model
            self.choice = np.genfromtxt(
                path.join(base_dir, *handyfilesvars.mod_dict[
                    "choice"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "choice"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "choice"]["io_data_names"][1])
            # Stop routine if files were not properly read in
            if any([len(x) == 0 for x in [
                    self.dmd_bl, self.dmd_therm, self.dmd_ntherm, self.tmp,
                    self.pc_dmd, self.choice]]):
                raise ValueError("Failure to read input file(s)")
            # Read in reference frequentist coefs to compare Bayesian estimates
            # against
            self.coefs = np.genfromtxt(
                path.join(base_dir, *handyfilesvars.coefs),
                skip_header=True, delimiter=',',
                names=handyfilesvars.coef_names_dtypes[0],
                dtype=handyfilesvars.coef_names_dtypes[1])
        # Data read-in for model re-estimation/prediction is common across
        # model types
        else:
            # Read in data for baseline demand model
            dmd_bl = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict[
                            "demand_bl"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "demand_bl"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "demand_bl"]["io_data_names"][1])
            # Read in data for other DR model types
            common_data = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict[
                            "demand_therm"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "demand_therm"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "demand_therm"]["io_data_names"][1])

            # Restrict prediction input file to appropriate prediction or
            # model estimation update data (if applicable)
            if mod_est is False:
                # Set inputs to baseline demand model
                # from prediction/estimation input files
                self.dmd_bl = dmd_bl
                self.hr_bl = dmd_bl['Hr']
                # Screen precooling or temperature-related strategies
                # (indicated by a set point change or lagged set point change
                # that is non-zero) for OAT <= 70
                screened_strats = np.unique(
                    common_data[np.where((
                        (common_data['tsp_delt'] != 0) |
                        (common_data['tsp_delt_lag'] != 0)) &
                        (common_data['t_out'] <= 70))]['Name'])
                # Post-screening data to use for predictions
                common_data_screened = common_data[
                    np.isin(common_data['Name'], screened_strats, invert=True)]
                # Pull out key variables from screened data as class attributes
                self.tmp_delt = common_data_screened['tsp_delt']
                self.tmp_delt_prev = common_data_screened['tsp_delt_lag']
                self.lgt_delt = common_data_screened['lt_pwr_delt_pct']
                self.plug_delt = common_data_screened['mels_delt_pct']
                self.pc_mag = common_data_screened['pc_tmp_inc']
                self.hr = common_data_screened['Hr']
                self.hr_orig = common_data['Hr']
                self.strategy = common_data_screened['Name']
                self.strategy_orig = common_data['Name']
                self.price_delt = common_data_screened['delt_price_kwh']
                # Set inputs to demand, temperature, lighting, and
                # pre-cooling models from screened prediction input files
                self.dmd_therm, self.dmd_ntherm, self.tmp, \
                    self.pc_dmd = (common_data_screened for n in range(4))
            elif mod_est is True:
                # Set inputs to demand, temperature, co2, lighting, and
                # pre-cooling models from prediction/estimation input files
                # Note: DR period data are flagged by rows where the set point
                # temperature change is greater than or equal to zero
                self.dmd_bl = dmd_bl
                # DR thermal demand data; set point change is positive or lag
                # set point change is negative (first hour of rebound)
                self.dmd_therm = common_data[
                    np.where((common_data['tsp_delt'] > 0) |
                             (common_data['tsp_delt_lag'] < 0) &
                             (common_data['t_out'] > 70))]
                # DR temp. data; set point change is positive
                self.tmp = common_data[np.where(
                    (common_data['tsp_delt'] > 0) &
                    (common_data['t_out'] > 70))]
                # DR non-thermal data; set point change and lagged set point
                # change are both zero
                self.dmd_ntherm = common_data[
                    np.where((common_data['tsp_delt'] == 0) &
                             (common_data['tsp_delt_lag'] == 0) & (
                             (common_data['lt_pwr_delt_pct'] != 0) |
                             (common_data['mels_delt_pct'] != 0)))]
                # Precooling data; set point change is negative
                self.pc_dmd = common_data[
                    np.where(
                        (common_data['tsp_delt'] < 0) &
                        (common_data['t_out'] > 70) &
                        (common_data['occ_frac'] > 0))]
                # Read in reference frequentist coefs. to compare Bayesian
                # estimates against
                self.coefs = np.genfromtxt(
                    path.join(base_dir, *handyfilesvars.coefs),
                    skip_header=True, delimiter=',',
                    names=handyfilesvars.coef_names_dtypes[0],
                    dtype=handyfilesvars.coef_names_dtypes[1])
                # Initialize attributes related to prediction mode to zero
                self.tmp_delt, self.tmp_delt_prev, self.lgt_delt, \
                    self.plug_delt, self.pc_mag, self.hr, \
                    self.hr_orig, self.strategy, self.strategy_orig, \
                    self.price_delt, self.hr_bl = (None for n in range(11))
            # When not initializing or assessing a model, choice model
            # attribute is irrelevant
            self.choice = None


class ModelIO(object):
    """Initialize input/output variables/data structure for each model

    Attributes:
        train_inds (numpy ndarray): Indices to use in restricting data for
            use in model training.
        test_inds (numpy ndarray): Indices to use in restricting data for
            use in model testing.
        X_all (numpy ndarray): Model input observations.
        Y_all (numpy ndarray): Model output observations.
    """

    def __init__(
            self, handyfilesvars, mod_init, mod_est, mod_assess, mod, data,
            bldg_type_vint):
        """Initialize class attributes."""

        # If model is being initialized and model assessment is requested,
        # set the portion of the data that should be used across models for
        # training vs. testing each model
        if (mod_init is True and mod_assess is True) or mod_assess is True:
            train_pct = 0.7
        else:
            train_pct = None

        if mod == "demand_bl":
            # If model is being initialized and model assessment is requested,
            # or previously initialized model is being assessed,
            # set training/testing indices to use on the demand_bl. data
            if (mod_init is True and mod_assess is True) or mod_assess is True:
                # Set training indices
                self.train_inds = np.random.randint(
                    0, len(data.dmd_bl),
                    size=int(len(data.dmd_bl) * train_pct))
                # Set testing indices
                self.test_inds = [
                    x for x in range(len(data.dmd_bl)) if
                    x not in self.train_inds]

            # Initialize variables for baseline demand model

            # Whole building occupancy fraction
            occ_frac = data.dmd_bl['occ_frac']
            # Outdoor air temperature
            temp_out = data.dmd_bl['t_out']
            # Outdoor relative humidity
            rh_out = data.dmd_bl['rh_out']
            # Number of hour
            hour_number = data.dmd_bl['hour_number']
            # # Climate zones
            # climate_zone = data.dmd_bl['climate']
            # # Vintages
            # vintage = data.dmd_bl['vintage']
            # Set a vector of ones for intercept estimation
            intercept = np.ones(len(occ_frac))
            # Categorical variables for Climate zones
            # cz_2A = data.dmd_bl['cz_2A']
            cz_2B = data.dmd_bl['cz_2B']
            cz_3A = data.dmd_bl['cz_3A']
            cz_3B = data.dmd_bl['cz_3B']
            cz_3C = data.dmd_bl['cz_3C']
            cz_4A = data.dmd_bl['cz_4A']
            cz_4B = data.dmd_bl['cz_4B']
            cz_4C = data.dmd_bl['cz_4C']
            cz_5A = data.dmd_bl['cz_5A']
            cz_5B = data.dmd_bl['cz_5B']
            cz_6A = data.dmd_bl['cz_6A']
            cz_6B = data.dmd_bl['cz_6B']
            cz_7A = data.dmd_bl['cz_7A']
            # Categorical variables for Vintages
            # v_1980 = data.dmd_bl['v1980']
            v_2004 = data.dmd_bl['v2004']
            v_2010 = data.dmd_bl['v2010']
            v_19802004 = data.dmd_bl['v19802004']
            # Outdoor temperature, outdoor relative humidity
            tmp_out_rh_out = temp_out * rh_out

            # Set model input (X) variables
            self.X_all = np.stack([
                intercept, temp_out, rh_out,
                v_19802004, v_2004, v_2010,
                cz_2B, cz_3A, cz_3B, cz_3C, cz_4A, cz_4B,
                cz_4C, cz_5A, cz_5B, cz_6A, cz_6B, cz_7A,
                occ_frac, hour_number, tmp_out_rh_out], axis=1)
            # Set model output (Y) variable for estimation cases
            if mod_init is True or mod_est is True or mod_assess is True:
                self.Y_all = data.dmd_bl['dmd_sf']
        elif mod == "demand_therm":
            # If model is being initialized and model assessment is requested,
            # or previously initialized model is being assessed,
            # set training/testing indices to use on the demand/temp. data
            if (mod_init is True and mod_assess is True) or mod_assess is True:
                # Set training indices
                # self.train_inds = np.random.randint(
                #     0, len(data.dmd_therm),
                #     size=int(len(data.dmd_therm) * train_pct))
                # # Set testing indices
                # self.test_inds = [
                #     x for x in range(len(data.dmd_therm)) if
                #     x not in self.train_inds]
                self.train_inds, self.test_inds = list(
                    range(len(data.dmd_therm)) for n in range(2))

            # Initialize variables for temperature and demand models

            # Whole building occupancy fraction
            occ_frac = data.dmd_therm['occ_frac']
            # Outdoor air temperature
            temp_out = data.dmd_therm['t_out']
            # Outdoor relative humidity
            rh_out = data.dmd_therm['rh_out']
            # Temperature set point difference
            tmp_delta = data.dmd_therm['tsp_delt']
            # Temperature set point difference lag
            tmp_delta_lag = data.dmd_therm['tsp_delt_lag']
            # Hours since DR event started (adjustment to normal op. condition)
            dr_start = data.dmd_therm['hrs_since_dr_st']
            # Hours since DR event ended (adjustment to normal op. condition)
            dr_end = data.dmd_therm['hrs_since_dr_end']
            # Lighting fraction reduction
            lt_delta = data.dmd_therm['lt_pwr_delt_pct']
            # Plug load power fraction reduction
            plug_delta = data.dmd_therm['mels_delt_pct']
            # Set a vector of ones for intercept estimation
            intercept = np.ones(len(tmp_delta))
            # Initialize interactive terms
            # Cooling SP/OAT interaction
            tmp_delt_tmp_out = tmp_delta * temp_out
            # Cooling SP/occupancy interaction
            tmp_delt_occ_frac = tmp_delta * occ_frac
            # Cooling SP/occupancy interaction
            tmp_delt_dr_start = tmp_delta * dr_start
            # Set model input (X) variables

            # In this model, the MELs input is dropped for retail
            if bldg_type_vint not in [
                    "stdaloneretailnew", "stdaloneretailold", "bigboxretail"]:
                self.X_all = np.stack([
                    intercept, temp_out, rh_out, occ_frac, tmp_delta, lt_delta,
                    plug_delta, dr_start, dr_end, tmp_delta_lag,
                    tmp_delt_tmp_out, tmp_delt_occ_frac, tmp_delt_dr_start],
                    axis=1)
            else:
                self.X_all = np.stack([
                    intercept, temp_out, rh_out, occ_frac, tmp_delta, lt_delta,
                    dr_start, dr_end, tmp_delta_lag, tmp_delt_tmp_out,
                    tmp_delt_occ_frac, tmp_delt_dr_start],
                    axis=1)
            # Set model output (Y) variable for estimation cases
            if mod_init is True or mod_est is True or mod_assess is True:
                self.Y_all = data.dmd_therm['dmd_delt_sf']
        elif mod == "demand_ntherm":
            # If model is being initialized and model assessment is requested,
            # or previously initialized model is being assessed,
            # set training/testing indices to use on the demand/temp. data
            if (mod_init is True and mod_assess is True) or mod_assess is True:
                # Set training indices
                # self.train_inds = np.random.randint(
                #     0, len(data.dmd_ntherm),
                #     size=int(len(data.dmd_ntherm) * train_pct))
                # # Set testing indices
                # self.test_inds = [
                #     x for x in range(len(data.dmd_ntherm)) if
                #     x not in self.train_inds]
                self.train_inds, self.test_inds = list(
                    range(len(data.dmd_ntherm)) for n in range(2))

            # Initialize variables for non-thermal demand model

            # Lighting fraction reduction
            lt_delta = data.dmd_ntherm['lt_pwr_delt_pct']
            # Plug load power fraction reduction
            plug_delta = data.dmd_ntherm['mels_delt_pct']
            # Set a vector of ones for intercept estimation
            intercept = np.ones(len(lt_delta))
            # Set model input (X) variables

            # In this model, the MELs input is dropped for retail
            if bldg_type_vint not in [
                    "stdaloneretailnew", "stdaloneretailold", "bigboxretail"]:
                self.X_all = np.stack([
                    intercept, lt_delta, plug_delta], axis=1)
            else:
                self.X_all = np.stack([
                    intercept, lt_delta], axis=1)
            # Set model output (Y) variable for estimation cases
            if mod_init is True or mod_est is True or mod_assess is True:
                self.Y_all = data.dmd_ntherm['dmd_delt_sf']
        elif mod == "temperature":
            # If model is being initialized and model assessment is requested,
            # or previously initialized model is being assessed,
            # set training/testing indices to use on the demand/temp. data
            if (mod_init is True and mod_assess is True) or mod_assess is True:
                # Set training indices
                # self.train_inds = np.random.randint(
                #     0, len(data.tmp),
                #     size=int(len(data.tmp) * train_pct))
                # # Set testing indices
                # self.test_inds = [
                #     x for x in range(len(data.tmp)) if
                #     x not in self.train_inds]
                self.train_inds, self.test_inds = list(
                    range(len(data.tmp)) for n in range(2))

            # Initialize variables for temperature and demand models

            # Whole building occupancy fraction
            occ_frac = data.tmp['occ_frac']
            # Outdoor air temperature
            temp_out = data.tmp['t_out']
            # Outdoor relative humidity
            rh_out = data.tmp['rh_out']
            # Temperature set point difference
            tmp_delta = data.tmp['tsp_delt']
            # Temperature set point difference lag
            tmp_delta_lag = data.tmp['tsp_delt_lag']
            # Hours since DR event started (adjustment to normal op. condition)
            dr_start = data.tmp['hrs_since_dr_st']
            # Hours since pre-cooling ended (if applicable)
            pcool_duration = data.tmp['pc_length']
            # Magnitude of pre-cooling temperature offset
            pcool_magnitude = data.tmp['pc_tmp_inc']
            # Set a vector of ones for intercept estimation
            intercept = np.ones(len(tmp_delta))
            # Initialize interactive terms
            # Pre-cool duration, and magnitude
            pcool_interact = pcool_duration * pcool_magnitude
            # Cooling SP/OAT interaction
            tmp_delt_tmp_out = tmp_delta * temp_out
            # Cooling SP/occupancy interaction
            tmp_delt_occ_frac = tmp_delta * occ_frac
            # Cooling SP/occupancy interaction
            tmp_delt_dr_start = tmp_delta * dr_start

            # Set model input (X) variables
            self.X_all = np.stack([
                intercept, temp_out, rh_out, occ_frac, tmp_delta,
                dr_start, tmp_delta_lag, pcool_magnitude, pcool_duration,
                tmp_delt_tmp_out, tmp_delt_occ_frac, tmp_delt_dr_start,
                pcool_interact], axis=1)
            # Set model output (Y) variable for estimation cases
            if mod_init is True or mod_est is True or mod_assess is True:
                self.Y_all = data.tmp['t_in_delt']
        elif mod == "demand_precool":
            # If model is being initialized and model assessment is requested,
            # or previously initialized model is being assessed,
            # set training/testing indices to use on the demand/temp. data
            if (mod_init is True and mod_assess is True) or mod_assess is True:
                # Set training indices
                # self.train_inds = np.random.randint(
                #     0, len(data.pc_dmd),
                #     size=int(len(data.pc_dmd) * train_pct))
                # # Set testing indices
                # self.test_inds = [
                #     x for x in range(len(data.pc_dmd)) if
                #     x not in self.train_inds]
                self.train_inds, self.test_inds = list(
                    range(len(data.pc_dmd)) for n in range(2))

            # Initialize variables for temp/demand pre-cooling models

            # Whole building occupancy fraction
            occ_frac = data.pc_dmd['occ_frac']
            # Outdoor air temperature
            temp_out = data.pc_dmd['t_out']
            # Outdoor relative humidity
            rh_out = data.pc_dmd['rh_out']
            # Temperature set point difference
            tmp_delta = data.pc_dmd['tsp_delt']
            # Hours since pre-cooling started
            pc_start = data.pc_dmd['hrs_since_pc_st']
            # Cooling SP/OAT interaction
            tmp_delt_tmp_out = tmp_delta * temp_out
            # Cooling SP/occupancy interaction
            tmp_delt_occ_frac = tmp_delta * occ_frac
            # Cooling SP/occupancy interaction
            tmp_delt_pc_start = tmp_delta * pc_start
            # Set a vector of ones for intercept estimation
            intercept = np.ones(len(tmp_delta))

            # Set model input (X) variables
            self.X_all = np.stack([
                intercept, temp_out, rh_out, occ_frac, tmp_delta, pc_start,
                tmp_delt_tmp_out, tmp_delt_occ_frac, tmp_delt_pc_start],
                axis=1)
            # Set model output (Y) variable for estimation cases
            if mod_init is True or mod_est is True or mod_assess is True:
                self.Y_all = data.pc_dmd['dmd_delt_sf']
        # Initialize input/output variables for choice regression
        elif mod == "choice":
            # Set train/test data to all data in this case
            self.train_inds, self.test_inds = ([
                x for x in range(len(data.choice))] for n in range(2))
            # Economic benefit
            economy = data.choice['economy']
            # # Pre-cooling temp. decrease
            # pc_tmp = data.choice['pc_tmp']
            # Pre-cooling temp. decrease (<=2 deg F event temp. increase)
            pc_tmp_l = data.choice['pc_tmp_low']
            # Pre-cooling temp. decrease (>2  deg F event temp. increase)
            pc_tmp_h = data.choice['pc_tmp_high']
            # Temp. increase
            tmp = data.choice['tmp']
            # Lighting decrease
            lgt = data.choice['lgt']
            # Daylighting
            dl = data.choice['daylt']
            # Daylighting multiplied by lgt
            lgt_dl = lgt * dl
            # Plug load decrease
            plug = data.choice['plug']
            # Set model input (X) variables; note, NO INTERCEPT
            # self.X_all = np.stack([
            #     economy, pc_tmp, tmp, lgt, lgt_dl, plug], axis=1)
            self.X_all = np.stack([
                economy, pc_tmp_l, pc_tmp_h, tmp, lgt, lgt_dl, plug], axis=1)
            # Set model output (Y) variable for estimation cases
            self.Y_all = data.choice['choice']


class ModelIOTrain():
    """Pull subset of data observations for use in model training.

    Attributes:
        X (numpy ndarray): Input data subset to use for model training.
        Y (numpy ndarray): Output data subset to use for model training.
    """

    def __init__(self, io_dat, mod_init, mod_assess):
        """Initialize class attributes."""

        # Only pull testing data subset for model initialization and/or
        # assessment of previously initialized model; otherwise
        # use all the available data for testing
        if (mod_init is True and mod_assess is True) or mod_assess is True:
            self.X = io_dat.X_all[io_dat.train_inds]
            self.Y = io_dat.Y_all[io_dat.train_inds]
        else:
            self.X = io_dat.X_all
            self.Y = io_dat.Y_all


class ModelIOTest():
    """Pull subset of data observations for use in model testing.

    Attributes:
        X (numpy ndarray): Input data subset to use for model testing.
        Y (numpy ndarray): Output data subset to use for model testing.
    """

    def __init__(self, io_dat, mod_init, mod_assess):
        """Initialize class attributes."""

        # Only pull testing data subset for model initialization and/or
        # assessment of previously initialized model; otherwise
        # use all the available data for testing
        if mod_init is True or mod_assess is True:
            self.X = io_dat.X_all[io_dat.test_inds]
            self.Y = io_dat.Y_all[io_dat.test_inds]
        else:
            self.X = io_dat.X_all
            self.Y = io_dat.Y_all


class ModelIOPredict():
    """Pull subset of data observations for use in model prediction.

    Attributes:
        X (numpy ndarray): Input data subset to use for model prediction.
    """

    def __init__(self, io_dat, hr_inds):
        """Initialize class attributes."""

        # Restrict data to the current hour in the prediction time horizon
        self.X = io_dat.X_all[hr_inds]


def main(base_dir):
    """Implement Bayesian network and plot resultant parameter estimates."""

    # Ensure the user is not trying to initialize and re-estimate the model
    # at the same time
    if opts.mod_init is True and opts.mod_est is True:
        raise ValueError(
            "Model initialization and re-estimation flags cannot be chosen "
            "at the same time; choose one or the other")

    # Initialize building type, square footage, and demand threshold if
    # applicable
    bldg_type_vint = opts.bldg_type
    # Require building square footage data if recommendations are generated
    if opts and opts.bldg_sf is not None:
        sf = opts.bldg_sf
    elif all([
        x is not True for x in [
            opts.mod_init, opts.mod_assess, opts.mod_est, opts.base_pred]]):
        raise ValueError(
            "Square footage is required for making predictions – specify "
            "square footage via the `--bldg_sf [insert square footage]` cmd "
            "line option and rerun")

    dmd_thres = opts.dmd_thres
    # Pull in building daylit sf percentage if given; if not, assume a
    # default daylit sf percentage of 30%
    if opts and opts.daylt_pct is not None:
        dl_pct = opts.daylt_pct
    else:
        dl_pct = 30

    # Instantiate useful input file and variable data object
    handyfilesvars = UsefulFilesVars(
        bldg_type_vint, opts.mod_init, opts.mod_est, opts.mod_assess)

    # Set numpy and theano RNG seeds for consistency across runs
    np.random.seed(123)
    th_rng = MRG_RandomStream()
    th_rng.seed(123)

    # Proceed with model inference only if user flags doing so; otherwise,
    # use previously estimated models to conduct assessment or make predictions
    if opts.mod_init is True:

        print("Loading input data...", end="", flush=True)
        # Read-in input data
        dat = ModelDataLoad(handyfilesvars, opts.mod_init, opts.mod_assess,
                            opts.mod_est, update_days=None)
        print("Complete.")

        # Loop through all model types (temperature, demand, co2, lighting)
        for mod in ["demand_therm", "demand_ntherm", "temperature",
                    "demand_precool"]:
            print("Initializing " + mod + " sub-model variables...",
                  end="", flush=True)
            # Initialize variable inputs and outputs for the given model type
            iog = ModelIO(handyfilesvars, opts.mod_init, opts.mod_est,
                          opts.mod_assess, mod, dat, bldg_type_vint)
            # Restrict model data to training subset (pertains to model
            # initialization only)
            iot = ModelIOTrain(iog, opts.mod_init, opts.mod_assess)
            print("Complete.")

            # Perform model inference
            with pm.Model() as var_mod:
                print("Setting " + mod + " sub-model priors and likelihood...",
                      end="", flush=True)
                # Handle choice model estimation (logistic regression)
                # differently than other models (linear regression)
                if mod != "choice":
                    # Set parameter priors (betas, error)
                    params = pm.Normal(
                        handyfilesvars.mod_dict[mod]["var_names"][0], 0, 10,
                        shape=(iot.X.shape[1]))
                    sd = pm.HalfNormal(
                        handyfilesvars.mod_dict[mod]["var_names"][1], 20)
                    # Likelihood of outcome estimator
                    est = pm.math.dot(iot.X, params)
                    # Likelihood of outcome
                    var = pm.Normal(
                        handyfilesvars.mod_dict[mod]["var_names"][2],
                        mu=est, sd=sd, observed=iot.Y)
                # Choice model estimation based on:
                # http://barnesanalytics.com/bayesian-
                # logistic-regression-in-python-using-pymc3
                else:
                    # Set parameter priors (betas, error); set informative
                    # prior (based on DCE future winter experiments) on plug
                    # load service loss coefficient, and vague priors on all
                    # other coefficients
                    params_1 = pm.Normal(
                        handyfilesvars.mod_dict[mod]["var_names"][0], 0, 10,
                        shape=(iot.X.shape[1] - 1))
                    # Plug load prior set to roughly that of lighting current
                    # summer coefficient given zero daylighting
                    params_2 = pm.Normal('plg', -3, 0.5, shape=1)
                    params = tt.tensor.concatenate([
                        params_1, params_2], axis=0)
                    # Likelihood of outcome estimator; apply sigmoid function
                    # for binary logistic regression
                    est = pm.math.sigmoid(pm.math.dot(iot.X, params))
                    # Likelihood of outcome; draw from Bernoulli distribution
                    # for binary logistic regression
                    var = pm.Bernoulli(
                        handyfilesvars.mod_dict[mod]["var_names"][2], p=est,
                        observed=iot.Y)
                print("Complete.")
                # Draw posterior samples
                trace = pm.sample(
                    chains=2, cores=1, init="advi",
                    return_inferencedata=False)

            # Store model, trace, and predictor variables
            with open(path.join(base_dir, *handyfilesvars.mod_dict[mod][
                    "io_data"][1]), 'wb') as co_s:
                print("Writing out " + mod + " sub-model...",
                      end="", flush=True)
                pickle.dump({'trace': trace, 'model': var_mod}, co_s)

            # If model assessment is desired, generate diagnostic plots
            if opts.mod_assess is True:
                print("Starting " + mod + " sub-model assessment...", end="",
                      flush=True)
                # Set reference coefficient values, estimated using a
                # frequentist regression approach; N/A for choice model
                if mod != "choice":
                    refs = list(
                        dat.coefs[mod][np.where(np.isfinite(dat.coefs[mod]))])
                else:
                    refs = None
                run_mod_assessment(handyfilesvars, trace, mod, iog,
                                   refs, bldg_type_vint, opts, update_n=None)
                print("Complete.")

    elif opts.mod_est is True:

        # Load updating data
        print("Loading input data...", end="", flush=True)
        dat = ModelDataLoad(handyfilesvars, opts.mod_init, opts.mod_assess,
                            opts.mod_est, update_days=None)
        print("Complete.")

        # Initialize traces
        traces = ""

        # Initialize blank traces lists for the first update to each mod type
        if not traces:
            # traces = {"demand_bl": [], "demand_therm": [], "demand_ntherm": [],
            #           "temperature": [], "demand_precool": []}
            traces = {"demand_therm": [], "demand_ntherm": [],
                      "temperature": [], "demand_precool": []}
        # Determine model types to update (thermal demand and temperature if no
        # precooling is indicated by the input data (TSP<0), otherwise add
        # precooling demand and temperature models if there are set point
        # adjustments, otherwise only update the non-thermal demand model)

        # Initialize list of models to update
        mod_update_list = []
        # If data exist to update thermal demand and temp. models, add to list
        if len(dat.dmd_therm) > 0:
            mod_update_list.extend(["demand_therm", "temperature"])
        # Ditto non-thermal demand model
        if len(dat.dmd_ntherm) > 0:
            mod_update_list.append("demand_ntherm")
        # Ditto precooling model
        if len(dat.pc_dmd) > 0:
            mod_update_list.append("demand_precool")

        # Loop through model updates
        for mod in mod_update_list:
            try:
                traces[mod] = gen_updates(handyfilesvars, opts, mod,
                                          traces[mod], dat, bldg_type_vint)
            # Handle case where update cannot be estimated (e.g., bad initial
            # energy, returns Value Error)
            except (ValueError, SamplingError):
                pass
            # After the update, generate some diagnostic plots showing
            # how parameter estimates changed for thermal-related models

            # Pull in frequentist estimates as initial reference
            refs = list(dat.coefs[mod][np.where(np.isfinite(dat.coefs[mod]))])
            plot_updating(
                handyfilesvars, handyfilesvars.mod_dict[mod]["var_names"][0],
                traces[mod], mod, bldg_type_vint, refs)

    elif opts.mod_assess is True:

        print("Loading input data...", end="", flush=True)
        # Read-in input data
        dat = ModelDataLoad(handyfilesvars, opts.mod_init, opts.mod_assess,
                            opts.mod_est, update_days=None)
        print("Complete.")

        # Loop through all model types (temperature, demand, co2, lighting)
        for mod in ["demand_therm", "demand_ntherm", "temperature",
                    "demand_precool"]:
            print("Loading " + mod + " sub-model...", end="", flush=True)
            # Reload trace
            with open(path.join(base_dir, *handyfilesvars.mod_dict[mod][
                    "io_data"][1]), 'rb') as store:
                trace = pickle.load(store)['trace']
            print("Complete.")
            print("Starting " + mod + " sub-model assessment...", end="",
                  flush=True)
            # Initialize variable inputs and outputs for the given model type
            iog = ModelIO(handyfilesvars, opts.mod_init, opts.mod_est,
                          opts.mod_assess, mod, dat, bldg_type_vint)

            # Set reference coefficient values, estimated using a frequentist
            # regression approach; N/A for choice model
            if mod != "choice":
                refs = list(
                    dat.coefs[mod][np.where(np.isfinite(dat.coefs[mod]))])
            else:
                refs = None
            run_mod_assessment(handyfilesvars, trace, mod, iog, refs,
                               bldg_type_vint, opts, update_n=None)
            print("Complete.")

    elif opts.base_pred is True:

        # Notify user of input data read
        print("Loading input data...")
        # Read-in input data for scenario
        dat = ModelDataLoad(
            handyfilesvars, opts.mod_init, opts.mod_assess,
            opts.mod_est, update_days=None)
        # Set number of samples to draw. for predictions
        n_samples = 1000
        # Initialize posterior predictive data dict
        pp_dict = {
            key: [] for key in handyfilesvars.mod_dict.keys()}
        # Make hourly baseline predictions
        print('Making baseline predictions...')
        # Predict the baseline demand and output
        with open(path.join(base_dir, *handyfilesvars.mod_dict['demand_bl'][
                "io_data"][1]), 'rb') as store:
            trace = pickle.load(store)['trace']
            pp_dict['demand_bl'] = run_mod_prediction_bl(
                    handyfilesvars, trace, 'demand_bl', dat, n_samples)
        predict_bl = pp_dict["demand_bl"]
        # Write predicted baseline demand value out to csv file
        predict_csv = path.join(base_dir, "data", "base_predict_byhr.csv")
        # THIS MAY NOT BE NECESSARY
        if path.exists(predict_csv):
            remove(predict_csv)
        # Write header string for output file with full hourly base predictions
        header_str = ""
        for x in range(predict_bl['dmd_bl'].shape[1]):
            header_str = header_str + ("Hour " + str(x+1) + ",")
        # Save full hourly baseline predictions (all samples) to CSV
        np.savetxt(predict_csv, predict_bl['dmd_bl'], delimiter=",",
                   header=header_str, comments='')
        # Convert full hourly baseline predictions to pandas DF
        predict_out_bl = pd.DataFrame(predict_bl['dmd_bl'])
        # Calculate summary statistics (mean and SD of full samples of base
        # demand predictions for each hour)
        mean = []
        std = []
        for i in range(len(predict_out_bl.columns)):
            Hr = predict_out_bl.iloc[:, [i]].values
            mean.append(np.mean(Hr))
            std.append(np.std(Hr))
        df_bl = pd.DataFrame(
            {'Hour': dat.hr_bl, 'Mean': mean, 'Standard Deviation': std})
        # Write summary stats to CSV
        predict_csv_sum = path.join(
            base_dir, "data", "base_predict_summary.csv")
        df_bl.to_csv(predict_csv_sum, sep=',', index=False)
        # Notify user of completed execution
        print("Complete (see './data/base_predict_summary.csv' and "
              "'./data/base_predict_byhr.csv' files.)")

    else:
        # Generate predictions for a next-day DR event with conditions and
        # candidate strategies described by an updated input file
        # (test_predict.csv, which is loaded into handyfilesvars)
        predict_out = gen_recs(
            handyfilesvars, sf, dmd_thres, bldg_type_vint, dl_pct)

        # Write summary dict with predictions out to JSON file
        with open(path.join(
                base_dir, *handyfilesvars.predict_out), "w") as jso:
            json.dump(predict_out, jso, indent=2)


def gen_updates(
        handyfilesvars, opts, mod, traces, dat, bldg_type_vint):

    print("Initializing " + mod + " sub-model variables...",
          end="", flush=True)
    # Initialize variable inputs/outputs for the given model type
    iog = ModelIO(handyfilesvars, opts.mod_init, opts.mod_est,
                  opts.mod_assess, mod, dat, bldg_type_vint)
    # Finalize variable inputs/outputs for the given model type
    iot = ModelIOTrain(iog, opts.mod_init, opts.mod_assess)
    print("Complete.")

    # Perform model inference
    with pm.Model() as var_mod:
        # For the first update, reload existing trace from model
        # initialization run
        if len(traces) == 0:
            print("Loading " + mod + " sub-model...", end="",
                  flush=True)
            with open(
                path.join(base_dir, *handyfilesvars.mod_dict[mod][
                    "io_data"][1]), 'rb') as store:
                trace = pickle.load(store)['trace']
                traces = [trace]
            print("Complete.")
        print("Setting " + mod +
              " sub-model priors and likelihood...",
              end="", flush=True)
        # Pull out the latest trace for use as prior information
        trace_prior = traces[-1]
        # Pull beta trace and set shorthand name
        t_params = trace_prior[
            handyfilesvars.mod_dict[mod]["var_names"][0]]
        # Determine means and standard deviation of normal
        # distributions for each beta parameter in the trace
        params_mean = t_params.mean(axis=0)
        params_sd = t_params.std(axis=0)
        # Set beta priors based on beta posteriors from last
        # iteration
        params = pm.Normal(
            handyfilesvars.mod_dict[mod]["var_names"][0],
            params_mean, params_sd,
            shape=(iot.X.shape[1]))
        # Set a prior for the model error term using a kernel
        # density estimation on the posterior error trace (
        # as the posterior for this term has an unknown
        # distribution)
        sd = from_posterior(
            handyfilesvars.mod_dict[mod]["var_names"][1], trace_prior[
                handyfilesvars.mod_dict[mod]["var_names"][1]])
        # Likelihood of outcome estimator
        est = pm.math.dot(iot.X, params)
        # Likelihood of outcome
        var = pm.Normal(
            handyfilesvars.mod_dict[mod]["var_names"][2],
            mu=est, sd=sd, observed=iot.Y)
        print("Complete.")
        # Draw posterior samples to yield a new trace
        trace = pm.sample(chains=2, cores=1, init="advi",
                          return_inferencedata=False)
        # Append current updates' traces to the traces from
        # all previous updates
        traces.append(trace)

    # Store model, trace, and predictor variables
    with open(path.join(base_dir, *handyfilesvars.mod_dict[mod][
            "io_data"][1]), 'wb') as co_s:
        print("Writing out " + mod + " sub-model...",
              end="", flush=True)
        pickle.dump({'trace': trace, 'model': var_mod}, co_s)

    # If model assessment is desired, generate diagnostic plots
    if opts.mod_assess is True:
        print("Starting " + mod + " sub-model assessment...", end="",
              flush=True)
        # Set reference coefficient values, estimated using a
        # frequentist regression approach
        if mod != "choice":
            refs = list(
                dat.coefs[mod][np.where(np.isfinite(dat.coefs[mod]))])
        else:
            refs = None
        run_mod_assessment(handyfilesvars, trace, mod, iog, refs,
                           bldg_type_vint, opts, str(len(traces) - 1))
        print("Complete.")

    return traces


def gen_recs(handyfilesvars, sf, dmd_thres, bldg_type_vint, dl_pct):

    # Notify user of input data read
    print("Loading input data...")
    # Read-in input data for scenario
    dat = ModelDataLoad(
        handyfilesvars, opts.mod_init, opts.mod_assess,
        opts.mod_est, update_days=None)
    # Set number of pre-cool hours to predict across
    hrs_pc = np.unique(dat.hr[np.where(dat.hr < 0)])
    # Set number of DR hours to predict across
    hrs_dr = np.unique(dat.hr[np.where(dat.hr > 0)])
    # Find names of candidate pre-cooling period strategies
    names_pc = dat.strategy[np.where(dat.hr == -1)]
    # Find names of candidate DR strategies (without baseline option)
    names_o = dat.strategy[np.where(dat.hr == 1)]
    names_o_orig = dat.strategy_orig[np.where(dat.hr_orig == 1)]
    # Add baseline (do nothing) option to the set of names to write out if
    # desired by the user; attach a default tag to baseline if no other
    # strategy names are tagged as the default
    if opts.null_strategy is True:
        default_flag = np.where(np.char.find(names_o, "(D)") != -1)
        if len(default_flag[0]) != 0:
            names = np.append(names_o, "Baseline - Do Nothing")
            names_orig = np.append(names_o_orig, "Baseline - Do Nothing")
        else:
            names = np.append(names_o, "Baseline - Do Nothing (D)")
            names_orig = np.append(names_o_orig, "Baseline - Do Nothing")
    else:
        names = names_o
        names_orig = names_o_orig
    # Find indices for the pre-cooling measures within the broader measure set
    names_pc_inds = []
    for pcn in names_pc:
        names_pc_inds.append(np.where(names == pcn)[0][0])
    # Set number of DR strategies to predict across
    n_choices = len(names)
    # Set number of samples to draw. for predictions
    n_samples = 1000
    # Initialize posterior predictive data dict
    pp_dict = {
        key: [] for key in handyfilesvars.mod_dict.keys()}
    # Initialize dict for storing demand/cost/service predictions for each
    # hour in the analysis (including precooling hours as applicable)
    # ds_dict_prep = {
    #     "demand": [], "demand precool": [], "cost": [], "cost precool": [],
    #     "temperature": [], "temperature precool": [],
    #     "lighting": [], "plug loads": []}
    ds_dict_prep = {
        "demand": [], "demand precool": [], "cost": [], "cost precool": [],
        "temperature": [], "temperature precool (low)": [],
        "temperature precool (high)": [], "lighting": [], "plug loads": []}
    # Initialize a numpy array that stores the count of the number of
    # times each candidate DR strategy is selected across simulated hours
    counts = np.zeros(n_choices)
    # Initialize a total count to use in normalizing number of selections
    # by strategy such that it is reported out as a % of simulated hours
    counts_denom = 0

    # Use latest set of coefficients from DCE future scenario results
    # (Order: economic benefit, temperature precool (<-=2 deg F temp. increase
    # during DR), temperature precool (>2 deg F temp. increase during DR),
    # temperature increase during DR, lighting decrease during DR, plug load
    # decrease during DR)

    # Reload DCE coefficient traces
    with open(path.join(base_dir, *handyfilesvars.mod_dict["choice"][
            "io_data"][1]), 'rb') as store:
        trace = pickle.load(store)['trace']
    # Stitch together parameter traces for plug load coefficient with the
    # other coefficient traces
    betas_choice_in = np.transpose(tt.tensor.concatenate([
        trace['dce_params'], trace['plg']], axis=1).eval())
    # Check to ensure consistency between the coefficient trace lengths and
    # the desired number of samples for predictions; if different, reformulate
    # the coefficient samples such that they match the desired sample N
    if betas_choice_in.shape[1] != n_samples:
        # Initialize final betas_choice format with zeros
        betas_choice = np.zeros((len(betas_choice_in), n_samples))
        # Loop through coefficient samples and reformulate as draws from
        # normal distribution with mean/sd determined from the original sample
        for ind, coef in enumerate(betas_choice_in):
            mean = np.mean(coef)
            sd = np.std(coef)
            betas_choice[ind] = np.random.normal(mean, sd, n_samples)
    else:
        betas_choice = betas_choice_in

    # Loop through all hours considered for the pre-cooling period
    for hr in hrs_pc:
        print("Making predictions for pre-cool hour " + str(hr))
        # Set data index that is unique to the current hour
        inds = np.where(dat.hr == hr)
        # Determine which precooling measures are active in the current hour
        pc_active_flag = []
        for pcn in names_pc:
            inds_pca = np.where((dat.hr == hr) & (dat.strategy == pcn))
            # Inactive pre-cooling measures will have TSP of zero
            if dat.tmp_delt[inds_pca] == 0:
                pc_active_flag.append(0)
            else:
                pc_active_flag.append(1)
        # Load and repopulate the demand precooling model
        for mod in ["demand_precool"]:
            # Reload trace
            with open(path.join(base_dir, *handyfilesvars.mod_dict[mod][
                    "io_data"][1]), 'rb') as store:
                trace = pickle.load(store)['trace']
            pp_dict[mod] = run_mod_prediction(
                handyfilesvars, trace, mod, dat, n_samples, inds,
                bldg_type_vint)
        # Force demand data for pre-cooling measures that aren't
        # active in the current hour to zero
        for n in range(n_samples):
            for pcm in range(len(names_pc)):
                if pc_active_flag[pcm] == 0:
                    pp_dict["demand_precool"][
                        'Demand Change (W/sf)'][n][pcm] = 0
        # Multiply change in pre-cooling demand/sf by sf and price delta to
        # get total cost difference for the operator during the pre-cool period
        # ; reflect DCE units of $100; convert demand from W/sf to kWh/sf
        cost_delt = (
            (pp_dict["demand_precool"]['Demand Change (W/sf)'] / 1000) * sf *
            dat.price_delt[inds])
        # Pull changes for each strategy during the pre-cool period directly
        # from the prediction input file
        pc_mags = np.tile(dat.pc_mag[inds], (n_samples, 1))
        # Store hourly predictions of changes in pre-cooling demand/temperature
        # and economic benefit for later write-out to recommendations.json
        # Predicted change in demand (precooling hour)
        ds_dict_prep["demand precool"].append(
            pp_dict["demand_precool"]['Demand Change (W/sf)'])
        # Predicted change in temperature (precooling hour); NOTE invert the
        # sign of the predictions to match what is expected by the DCE equation
        # ds_dict_prep["temperature precool"].append(pc_mags)
        ds_dict_prep["temperature precool (high)"].append(pc_mags)
        ds_dict_prep["temperature precool (low)"].append(pc_mags)
        # Predicted change in economic benefit (precooling hour)
        ds_dict_prep["cost precool"].append(cost_delt)

    # Loop through all hours considered for the event (event plus rebound)
    for hr in hrs_dr:
        print("Making predictions for DR hour " + str(hr))
        # Set data index that is unique to the current hour
        inds = np.where(dat.hr == hr)
        # Determine which measures affect thermostat set points in the current
        # hour, and lighting settings in the current hour
        tmp_delt_flag = []
        dmd_dat = np.tile(dat.plug_delt[inds], (n_samples, 1))
        for mn in names_o:
            inds_tmp = np.where((dat.hr == hr) & (dat.strategy == mn))
            # Measures that do not affect tsp will have the change in
            # set point and lag in set point change vars set to zero
            if (dat.tmp_delt[inds_tmp] == 0) & \
               (dat.tmp_delt_prev[inds_tmp] == 0):
                tmp_delt_flag.append(0)
            else:
                tmp_delt_flag.append(1)

        for mod in ["demand_ntherm", "demand_therm", "temperature"]:
            # Reload trace
            with open(path.join(base_dir, *handyfilesvars.mod_dict[mod][
                    "io_data"][1]), 'rb') as store:
                trace = pickle.load(store)['trace']
            pp_dict[mod] = run_mod_prediction(
                handyfilesvars, trace, mod, dat, n_samples, inds,
                bldg_type_vint)
        # Force predicted temperature change for measures that don't affect
        # thermostat set points to zero
        for n in range(n_samples):
            for tm in range(len(names_o)):
                if tmp_delt_flag[tm] == 0:
                    pp_dict["temperature"][
                        'Temperature Change (ºF)'][n][tm] = 0
                    dmd_dat[n][tm] = pp_dict["demand_ntherm"][
                        'Demand Change (W/sf)'][n][tm]
                else:
                    dmd_dat[n][tm] = pp_dict["demand_therm"][
                        'Demand Change (W/sf)'][n][tm]
        # Multiply change in demand/sf by sf and price delta to get total
        # cost difference for the operator; reflect DCE units of $100;
        # convert demand from W/sf to kWh/sf
        cost_delt = (
            (dmd_dat / 1000) * sf * dat.price_delt[inds])
        # Extend oaf delta values for each choice across all samples
        # oaf_delt = np.tile(dat.oaf_delt[inds], (n_samples, 1))
        # Extend plug load delta values for each choice across all samples
        plug_delt = np.tile(dat.plug_delt[inds], (n_samples, 1))
        # Extend lighting delta values for each choice across all samples
        lt_delt = np.tile(dat.lgt_delt[inds], (n_samples, 1))
        # Store hourly predictions of changes in demand, cost, and services
        # Predicted change in demand
        ds_dict_prep["demand"].append(dmd_dat)
        # Predicted change in economic benefit
        ds_dict_prep["cost"].append(cost_delt)
        # Predicted change in temperature (event hour)
        ds_dict_prep["temperature"].append(pp_dict["temperature"][
            "Temperature Change (ºF)"])
        # Predicted change in lighting
        ds_dict_prep["lighting"].append(lt_delt)
        # Predicted change in plug loads
        ds_dict_prep["plug loads"].append(plug_delt)

    # Initialize a dict to use in storing final demand/cost/service predictions
    # (e.g., the predictions across all hours in the event that are fed into
    # the discrete choice model), as well as final choice probabilities; all
    # values are initialized to zero; note that baseline choice option will
    # not be updated (is left with all zero values for utility calc later)
    # ds_dict_fin = {
    #     key: np.zeros((n_samples, n_choices)) for key in [
    #         "demand", "demand precool", "cost", "cost precool", "temperature",
    #         "temperature precool", "lighting", "plug loads"]
    # }
    ds_dict_fin = {
        key: np.zeros((n_samples, n_choices)) for key in [
            "demand", "demand precool", "cost", "cost precool", "temperature",
            "temperature precool (high)", "temperature precool (low)",
            "lighting", "plug loads"]
    }
    # Loop through all variable keys in the final predictions dict and update
    # the final predictions data
    for key in sorted(ds_dict_fin.keys()):
        # For economic benefit predictions, sum all the predicted changes in
        # costs across all hours in the analysis (stored in the prep dict);
        # add pre-cooling economic losses to overall economic benefit; handle
        # case where there are no pre-cooling measures or costs
        if key == "cost" or (
                key == "cost precool" and len(ds_dict_prep[key]) != 0):
            # Loop through all hours of cost data stored in the prep dicts and
            # add to the economic benefit/loss variables
            for ind, elem in enumerate(ds_dict_prep[key]):
                # For total economic benefit, sum in-event benefits across all
                # hours of the event + rebounud
                if key == "cost":
                    # Loop through all samples
                    for ind_n in range(n_samples):
                        # Loop through all measures per sample
                        for ind_m in range(len(ds_dict_prep[key][0][0])):
                            # If first hour, initialize the change in cost data
                            if ind == 0:
                                ds_dict_fin[key][ind_n][ind_m] = \
                                    elem[ind_n][ind_m]
                            # If after the first hour, add to the change in
                            # cost data
                            else:
                                ds_dict_fin[key][ind_n][ind_m] = \
                                    ds_dict_fin[key][ind_n][ind_m] + \
                                    elem[ind_n][ind_m]
                # For pre-cooling period economic loss, sum pre-cooling
                # economic losses across all hours of the pre-cooling period;
                # add these losses to the total economic benefit variable as
                # well
                else:
                    # Loop through all samples
                    for ind_n in range(n_samples):
                        # Loop through all pre-cooling strategies
                        for ind_pc in range(len(names_pc)):
                            # Update both total economic benefit ('cost') and
                            # pre-cooling economic loss variables in the final
                            # dict
                            for cost_key in ["cost", "cost precool"]:
                                ds_dict_fin[cost_key][ind_n][
                                    names_pc_inds[ind_pc]] += \
                                    elem[ind_n][ind_pc]
        # For predictions that do not concern economic benefits/losses or
        # choice probabilities (e.g., change in demand, temperature, lighting,
        # outdoor air, and plug loads), pull the max median predicted change
        # in these variables from the prep dict, across all simulated hours;
        # handle cases where there are no pre-cooling measures or service
        # changes to update
        # elif key != "choice probabilities" and (key not in [
        #     "temperature precool", "demand precool", "cost precool"] or (
        #         key in ["temperature precool", "demand precool"] and len(
        #             ds_dict_prep[key]) != 0)):
        elif key != "choice probabilities" and (key not in [
            "temperature precool (high)", "temperature precool (low)",
            "demand precool", "cost precool"] or (
                key in ["temperature precool (high)",
                        "temperature precool (low)", "demand precool"] and len(
                    ds_dict_prep[key]) != 0)):
            # Initialize list for storing the median predicted change in
            # variable for the given hour represented by the the prep dict
            medians = []
            # Loop through all hours of data and measures stored in the prep
            # dict and append median values of the data for each hour to a list
            for elem in ds_dict_prep[key]:
                medians.append(np.median(elem, axis=0))
            # Transpose the resultant list to ease further operations on it
            medians = np.transpose(medians)
            # Find the hour in which the predicted median value of the given
            # variable is at its maximum for a given measure (or, in the case
            # of precooling demand, at its minimum for a given measure).
            if key != "demand precool":
                max_min_median = [np.where(x == max(x))[0][0] for x in medians]
            else:
                max_min_median = [np.where(x == min(x))[0][0] for x in medians]
            # Set the variable in the final dict to reflect these max or min
            # hour value sets for each measure
            # Loop through all samples
            for ind_n in range(n_samples):
                # Loop through all measures per sample
                for ind_m in range(len(ds_dict_prep[key][0][0])):
                    # Set the final dict variable differently according to the
                    # variable itself; precooling  temperature and demand
                    # variables will reflect only the subset of the total set
                    # of measures that feature pre-cooling
                    # if key not in ["temperature precool",
                    #                "demand precool"]:
                    if key not in ["temperature precool (high)",
                                   "temperature precool (low)",
                                   "demand precool"]:
                        ds_dict_fin[key][ind_n][ind_m] = ds_dict_prep[key][
                            max_min_median[ind_m]][ind_n][ind_m]
                    else:
                        ds_dict_fin[key][ind_n][names_pc_inds[ind_m]] = \
                            ds_dict_prep[key][max_min_median[ind_m]][
                            ind_n][ind_m]
        # For choice probability data, do nothing here
        else:
            pass

    # Set low pre-cooling input list values to zero when predicted temperature
    # change during the DR period is >2; set high pre-cooling input list values
    # to zero when predicted temperature change during the DR period is <=2
    for ind_n in range(n_samples):
        for ind_m in range(len(names)):
            if ds_dict_fin["temperature"][ind_n][ind_m] <= 2:
                ds_dict_fin["temperature precool (high)"][ind_n][ind_m] = 0
            else:
                ds_dict_fin["temperature precool (low)"][ind_n][ind_m] = 0

    # Stack all model inputs into a single array for use in the DCE function
    # x_choice = np.stack([
    #     ds_dict_fin["cost"], ds_dict_fin["temperature precool"],
    #     ds_dict_fin["temperature"], ds_dict_fin["lighting"],
    #     ds_dict_fin["lighting"] * dl_pct, ds_dict_fin["plug loads"]])
    x_choice = np.stack([
        ds_dict_fin["cost"], ds_dict_fin["temperature precool (low)"],
        ds_dict_fin["temperature precool (high)"],
        ds_dict_fin["temperature"], ds_dict_fin["lighting"],
        ds_dict_fin["lighting"] * dl_pct, ds_dict_fin["plug loads"]])
    # Multiply model inputs by DCE betas to yield choice logits
    choice_logits = np.transpose(
        np.sum([np.transpose(x_choice[i]) * betas_choice[i] for
                i in range(len(x_choice))], axis=0))
    # Force utilities to large negative value for options that don't meet
    # the demand reduction threshold, when given; (this ensures that these
    # strategies will never be selected when they don't meet the threshold)
    if dmd_thres is not None:
        for ind_n in range(n_samples):
            for ind_m in range(len(names)):
                if ((ds_dict_fin["demand"][ind_n][ind_m] * sf) / 1000) < \
                        dmd_thres:
                    choice_logits[ind_n][ind_m] = -999
    # Softmax transformation of logits into choice probabilities
    choice_probs = softmax(choice_logits, axis=1)
    # Add choice probabilities to the final variable data dict to write out
    ds_dict_fin["choice probabilities"] = choice_probs
    # Simulate choices across all samples given inputs and betas
    choice_out = [
        np.random.choice(n_choices, 1000, p=x) for x in choice_probs]
    # Report frequency with which each choice occurs for the scenario
    unique, counts_hr = np.unique(choice_out, return_counts=True)
    # Add to count of frequency with which each DR strategy is
    # selected
    counts[unique] += counts_hr
    # Add to total number of simulated hours
    counts_denom += np.sum(counts_hr)
    # Develop a final list of simulated counts by strategy
    counts_orig = np.zeros(len(names_orig))
    # Counts are zeros for strategies in original list that were screened
    for ind, n in enumerate(names_orig):
        # Find index of post-screening measure data to add
        ind_restr = np.where(names == n)[0]
        # If data are found update counts for pre-screen list of measures
        if len(ind_restr) != 0:
            counts_orig[ind] = counts[ind_restr[0]]
    # Store summary of the percentage of simulated hours that each
    # DR strategy was predicted to be selected in a dict; also store the final
    # set of input data for the DCE model in this dict, and the predicted
    # choice probability outputs from the DCE
    predict_out = {
        "notes": (
            "'predictions' key values represent the percentage of simulations "
            "in which each candidate DR strategy is chosen for current event; "
            "'input output data' key values report the full set of input data "
            "used to generate predicted choice probabilities, as well as the "
            "predicted choice probability outputs themselves"),
        "predictions": {
            x: round(((y / counts_denom) * 100), 5) for
            x, y in zip(names_orig, counts_orig)},
        "input output data": {key: {
            names[x]: list(np.transpose(ds_dict_fin[key])[x]) for
            x in range(len(names))} for key in ds_dict_fin.keys()}
        }

    return predict_out


def run_mod_prediction(
        handyfilesvars, trace, mod, dat, n_samples, inds, bldg_type_vint):

    # Initialize variable inputs and outputs for the given model type
    iop_all = ModelIO(handyfilesvars, opts.mod_init, opts.mod_est,
                      opts.mod_assess, mod, dat, bldg_type_vint)
    # Restrict variable inputs and outputs to current prediction hour
    iop = ModelIOPredict(iop_all, inds)
    with pm.Model() as var_mod:
        # Set parameter priors (betas, error)
        params = pm.Normal(
            handyfilesvars.mod_dict[mod][
                "var_names"][0], 0, 10, shape=(iop.X.shape[1]))
        sd = pm.HalfNormal(
            handyfilesvars.mod_dict[mod]["var_names"][1], 20)
        # Likelihood of outcome estimator
        est = pm.math.dot(iop.X, params)
        # Likelihood of outcome
        var = pm.Normal(
            handyfilesvars.mod_dict[mod]["var_names"][2],
            mu=est, sd=sd, observed=np.zeros(iop.X.shape[0]))
        # Sample predictions for trace
        ppc = pm.sample_posterior_predictive(
            trace, samples=n_samples)

    return ppc


def run_mod_prediction_bl(handyfilesvars, trace, mod, dat, n_samples):
    # Initialize variable inputs and outputs for the given model type
    iop = ModelIO(handyfilesvars, opts.mod_init, opts.mod_est,
                  opts.mod_assess, mod, dat)
    with pm.Model() as var_mod:
        # Set parameter priors (betas, error)
        params = pm.Normal(
            handyfilesvars.mod_dict[mod][
                "var_names"][0], 0, 10, shape=(iop.X_all.shape[1]))
        sd = pm.HalfNormal(
            handyfilesvars.mod_dict[mod]["var_names"][1], 20)
        # Likelihood of outcome estimator
        est = pm.math.dot(iop.X_all, params)
        # Likelihood of outcome
        var = pm.Normal(
            handyfilesvars.mod_dict[mod]["var_names"][2],
            mu=est, sd=sd, observed=np.zeros(iop.X_all.shape[0]))
        # Sample predictions for trace
        ppc = pm.sample_posterior_predictive(
            trace, samples=n_samples)

    return ppc


def run_mod_assessment(
        handyfilesvars, trace, mod, iog, refs, bldg_type_vint, opts, update_n):

    # Determine file name based on whether model assessment is being run
    # for an updating routine or not
    if update_n:
        fig1_file_name = ("_update_" + update_n +
                          handyfilesvars.mod_dict[mod]["fig_names"][0])
        fig2_file_name = ("_update_" + update_n +
                          handyfilesvars.mod_dict[mod]["fig_names"][1])
    else:
        fig1_file_name = handyfilesvars.mod_dict[mod]["fig_names"][0]
        fig2_file_name = handyfilesvars.mod_dict[mod]["fig_names"][1]

    # Plot parameter traces
    pm.plot_trace(trace)
    fig1_path = path.join(
        "diagnostic_plots", bldg_type_vint, fig1_file_name)
    plt.gcf().savefig(fig1_path)
    # Plot parameter posterior distributions
    pm.plots.plot_posterior(
        trace, var_names=[handyfilesvars.mod_dict[mod]["var_names"][0]],
        textsize=24, ref_val=refs)
    fig2_path = path.join(
        "diagnostic_plots", bldg_type_vint, fig2_file_name)
    plt.gcf().savefig(fig2_path)

    # Only proceed further with diagnostics if model assessment is not being
    # run for an updating routine, and only for models other than the choice
    # model (only parameter traces/distributions are printed for choice model)
    if opts.mod_est is not True and mod != "choice":
        # Set testing data
        iot = ModelIOTest(iog, opts.mod_init, opts.mod_assess)
        # Re-initialize model with subset of data used for testing
        # (if applicable)
        with pm.Model() as var_mod:
            # Set parameter priors (betas, error)
            params = pm.Normal(
                handyfilesvars.mod_dict[mod][
                    "var_names"][0], 0, 10, shape=(iot.X.shape[1]))
            sd = pm.HalfNormal(
                handyfilesvars.mod_dict[mod]["var_names"][1], 20)
            # Likelihood of outcome estimator
            est = pm.math.dot(iot.X, params)
            # Likelihood of outcome
            var = pm.Normal(
                handyfilesvars.mod_dict[mod]["var_names"][2],
                mu=est, sd=sd, observed=iot.Y)
            output_diagnostics(
                handyfilesvars, trace, iot, mod, bldg_type_vint, var_mod)


def output_diagnostics(
        handyfilesvars, trace, iot, mod, bldg_type_vint, var_mod):

    # Posterior predictive
    ppc_var = pm.sample_posterior_predictive(trace)
    obs_data = iot.Y
    pred_data = ppc_var[handyfilesvars.mod_dict[mod][
        "var_names"][2]]

    # Histogram / posterior fit to data
    fig2, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].hist(
        [n.mean() for n in pred_data], bins=20, alpha=0.5,
        label="Posterior predicted")
    axs[0].axvline(obs_data.mean(), linewidth=3, label="Observed mean")
    axs[0].legend()
    axs[0].set_title('Posterior Predictive of the Mean', fontsize=16)
    axs[0].set_xlabel("Mean " + handyfilesvars.mod_dict[mod][
        "var_names"][2], fontsize=16)
    axs[0].set_ylabel('Frequency', fontsize=16)
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axs[0].tick_params(axis='y', labelsize=14)
    axs[1].set_title('Posterior Predictive Fit', fontsize=16)
    axs[1].set_ylabel('Density', fontsize=16)
    az.plot_ppc(az.from_pymc3(
        posterior_predictive=ppc_var, model=var_mod), color="C9", alpha=0.2,
        textsize=16, mean=False, ax=axs[1], legend=False)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].legend(['Observed', 'Posterior predicted'])
    mx = max([max([max(x) for x in pred_data]), max(obs_data)])
    mn = min([min([min(x) for x in pred_data]), min(obs_data)])
    axs[1].set_xlim([mn - mx * 0.1, mx + mx * 0.1])
    fig2_path = path.join(
        "diagnostic_plots", bldg_type_vint,
        handyfilesvars.mod_dict[mod]["fig_names"][2])
    fig2.savefig(fig2_path)

    # Scatter/line
    prediction_data = pd.DataFrame(
        {'Mean Predicted Value': pred_data.mean(axis=0),
         'Observed Value': obs_data})
    # sns.set_style("darkgrid")
    fig3 = sns.lmplot(
        y='Observed Value', x='Mean Predicted Value', data=prediction_data,
        line_kws={'color': 'red', 'alpha': 0.5},
        scatter_kws={'color': 'blue', 'alpha': 0.05}, fit_reg=True)
    ax2 = plt.gca()
    ax2.set_title(label=("Observed vs. Mean Predicted (" + mod + ")"))
    xticks, xticklabels = plt.xticks()
    xmin = (3*xticks[0] - xticks[1])/2.
    # shaft half a step to the right
    xmax = (3*xticks[-1] - xticks[-2])/2.
    plt.xlim(xmin, xmax)
    plt.xticks(xticks)
    fig3_path = path.join(
        "diagnostic_plots", bldg_type_vint,
        handyfilesvars.mod_dict[mod]["fig_names"][3])
    fig3.savefig(fig3_path)

def from_posterior(param, samples):

    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)
    # What was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return pm.Interpolated(param, x, y)


def plot_updating(handyfilesvars, param, traces, mod, bldg_type_vint, refs):

    # Set color map for plots
    cmap = mpl.cm.cool
    # Set variable name to plot
    if mod == "demand_ntherm":
        var_name = "Lighting Change Parameter Distribution"
        var_ind = 1
    else:
        var_name = "Set Point Change Parameter Distribution"
        var_ind = 1
    # Initialize subplots and figure object
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
    # Develop and plot kernel density estimators of parameter traces,
    # and plot these estimators across each successive parameter update
    for update_i, trace in enumerate(traces):
        if trace is not None:
            # Focus on set point parameter (4th in the demand model)
            samples_sp = np.array([x[var_ind] for x in trace[param]])
            smin_sp, smax_sp = np.min(samples_sp), np.max(samples_sp)
            x_sp = np.linspace(smin_sp, smax_sp, 100)
            y_sp = stats.gaussian_kde(samples_sp)(x_sp)
            if update_i == 0:
                ax.axvline(x=refs[var_ind], label='Frequentist estimate',
                           color="slategray", linewidth=3)
                ax.plot(
                    x_sp, y_sp, color=cmap(update_i / (len(traces)-1)),
                    label="Initial Estimate")
            elif update_i == (len(traces) - 1):
                ax.plot(
                    x_sp, y_sp, color=cmap(update_i / (len(traces)-1)),
                    label="After " + str(update_i) + " Updates")
            else:
                ax.plot(
                    x_sp, y_sp, color=cmap(update_i / (len(traces)-1)))

    # Set plot title, axis labels, and legend
    ax.set_title(var_name, fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_xlabel('Parameter Value', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend()
    # Set figure layout parameter
    fig.tight_layout(pad=1.0)
    # Determine figure path and save figure
    fig_path = path.join(
        "diagnostic_plots", bldg_type_vint,
        handyfilesvars.mod_dict[mod]["fig_names"][4])
    fig.savefig(fig_path)


if __name__ == '__main__':
    # Handle optional user-specified execution arguments
    parser = ArgumentParser()
    # Optional flag to change default model execution mode (prediction)
    # to initialization, estimation, and/or assessment
    parser.add_argument("--mod_init", action="store_true",
                        help="Initialize a model")
    parser.add_argument("--mod_est", action="store_true",
                        help="Re-estimate a model")
    parser.add_argument("--mod_assess", action="store_true",
                        help="Assess a model")
    parser.add_argument("--base_pred", action="store_true",
                        help="Make base-case demand predictions")
    # Required flags for building type and size
    parser.add_argument("--bldg_type", required=True, type=str,
                        choices=["mediumofficenew", "mediumofficeold",
                                 "stdaloneretailnew", "stdaloneretailold",
                                 "largeofficenew", "largeofficeold",
                                 "largeofficenew_elec", "largeofficeold_elec",
                                 "bigboxretail"],
                        help="Building type/vintage")
    parser.add_argument("--bldg_sf", required=False, type=int,
                        help="Building square footage")
    parser.add_argument("--null_strategy", action="store_true",
                        help="Add the baseline (do nothing) DR strategy")
    parser.add_argument("--dmd_thres", type=float,
                        help="Optional demand reduction threshold (kW)")
    parser.add_argument("--daylt_pct", type=float,
                        help="Optional daylighting fraction (%)")
    # Object to store all user-specified execution arguments
    opts = parser.parse_args()
    base_dir = getcwd()
    main(base_dir)
