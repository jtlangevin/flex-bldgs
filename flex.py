#!/usr/bin/env python3
import pymc3 as pm
import theano as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.special import softmax
from scipy import stats
from os import getcwd, path
from argparse import ArgumentParser
import pickle
import arviz as az
from matplotlib import pyplot as plt
import matplotlib as mpl
import json

from pyfmi import load_fmu
import pandas as pd
import random as random
import os as os
from datetime import date, datetime, time, timedelta
import datetime as dt

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
        dmd_tmp_dat, co2_dat, stored_tmp, stored_dmd, \
            stored_co2, lgt_dat, stored_lt = (None for n in range(7))

        # Set data input and output files for all models
        if bldg_type_vint == "mediumofficenew":
            # Handle data inputs differently for model initialization vs.
            # model re-estimation and prediction (the former uses different
            # CSVs for each building type, while the latter will only draw
            # from one CSV)
            if mod_init is True or mod_assess is True:
                dmd_tmp_dat = ("data", "MO_DR_new.csv")
                co2_dat = ("data", "CO2_MO.csv")
                pc_tmp_dat = ("data", "MO_Precooling_new.csv")
            elif mod_est is True:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (
                    ("data", "test_update.csv") for n in range(3))
            else:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (
                    ("data", "test_predict.csv") for n in range(3))
            # Set stored model data files
            stored_tmp = ("model_stored", "tmp_mo_n.pkl")
            stored_dmd = ("model_stored", "dmd_mo_n.pkl")
            stored_co2 = ("model_stored", "co2_mo.pkl")
            stored_pc_tmp = ("model_stored", "pc_mo_n.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_mo_n.csv")
        # Medium office, <2004 vintage
        elif bldg_type_vint == "mediumofficeold":
            if mod_init is True or mod_assess is True:
                dmd_tmp_dat = ("data", "MO_DR_old.csv")
                co2_dat = ("data", "CO2_MO.csv")
                pc_tmp_dat = ("data", "MO_Precooling_old.csv")
            elif mod_est is True:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (
                    ("data", "test_update.csv") for n in range(3))
            else:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (
                    ("data", "test_predict.csv") for n in range(3))
            stored_tmp = ("model_stored", "tmp_mo_o.pkl")
            stored_dmd = ("model_stored", "dmd_mo_o.pkl")
            stored_co2 = ("model_stored", "co2_mo.pkl")
            stored_pc_tmp = ("model_stored", "pc_mo_o.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_mo_o.csv")
        # Retail, >=2004 vintage
        elif bldg_type_vint == "retailnew":
            if mod_init is True or mod_assess is True:
                dmd_tmp_dat = ("data", "Retail_DR_new.csv")
                co2_dat = ("data", "CO2_Retail.csv")
                pc_tmp_dat = ("data", "Retail_Precooling_new.csv")
            elif mod_est is True:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (
                    ("data", "test_update.csv") for n in range(3))
            else:
                dmd_tmp_dat, co2_datm, pc_tmp_dat = (
                    ("data", "test_predict.csv") for n in range(3))
            stored_tmp = ("model_stored", "tmp_ret_n.pkl")
            stored_dmd = ("model_stored", "dmd_ret_n.pkl")
            stored_co2 = ("model_stored", "co2_ret.pkl")
            stored_pc_tmp = ("model_stored", "pc_ret_n.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_ret_n.csv")
        # Medium office, <2004 vintage
        elif bldg_type_vint == "retailold":
            if mod_init is True or mod_assess is True:
                dmd_tmp_dat = ("data", "Retail_DR_old.csv")
                co2_dat = ("data", "CO2_Retail.csv")
                pc_tmp_dat = ("data", "Retail_Precooling_old.csv")
            elif mod_est is True:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (
                    ("data", "test_update.csv") for n in range(3))
            else:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (
                    ("data", "test_predict.csv") for n in range(3))
            stored_tmp = ("model_stored", "tmp_ret_o.pkl")
            stored_dmd = ("model_stored", "dmd_ret_o.pkl")
            stored_co2 = ("model_stored", "co2_ret.pkl")
            stored_pc_tmp = ("model_stored", "pc_ret_o.pkl")
            # Regression coefficients
            self.coefs = ("data", "coefs_ret_o.csv")
        # Lighting models are not broken out by building type/vintage
        if mod_init is True or mod_assess is True:
            lgt_dat = ("data", "Illuminance.csv")
        elif mod_est is True:
            lgt_dat = ("data", "test_update.csv")
        else:
            lgt_dat = ("data", "test_predict.csv")
        stored_lt = ("model_stored", "lt.pkl")

        # Set data input file column names and data types for model
        # initialization; these are different by model type (though the same
        # for the temperature and demand models, which draw from the same CSV)
        if mod_init is True or mod_assess is True:
            tmp_dmd_names_dtypes = [
                ('id', 'vintage', 'day_typ', 'hour', 'climate',
                 'dmd_delt_sf', 't_in_delt', 'rh_in_delt', 't_out', 'rh_out',
                 'cloud_out', 'occ_frac', 'tsp_delt', 'lt_pwr_delt_pct',
                 'ven_delt_pct', 'mels_delt_pct', 'hrs_since_dr_st',
                 'hrs_since_dr_end', 'hrs_since_pc_end', 'hrs_since_pc_st',
                 'tsp_delt_lag', 'lt_pwr_delt_pct_lag', 'mels_delt_pct_lag',
                 'ven_delt_pct_lag', 'pc_tmp_inc', 'pc_length'),
                (['<i4'] * 4 + ['<U25'] + ['<f8'] * 21)]
            co2_names_dtypes = [(
                'id', 'vintage', 'day_typ', 'hour', 'climate',
                'co2_in_delt', 't_out', 'rh_out', 'occ_frac', 'tsp_delt',
                'lt_pwr_delt_pct', 'ven_delt_pct', 'mels_delt_pct',
                'hrs_since_dr_st', 'hrs_since_dr_end', 'hrs_since_pc_st',
                'hrs_since_pc_end', 'tsp_delt_lag', 'lt_pwr_delt_pct_lag',
                'mels_delt_pct_lag', 'ven_delt_pct_lag'), (
                ['<i4'] * 4 + ['<U25'] + ['<f8'] * 16)]
            lt_names_dtypes = [
                ('id', 'lt_in_delt_pct', 'lt_in_delt', 'hour', 'lt_nat',
                 'cloud_out', 'hrs_since_dr_st', 'lt_pwr_delt', 'base_lt_frac',
                 'lt_pwr_delt_pct'), (['<i4'] * 1 + ['<f8'] * 9)]
            pc_tmp_names_dtypes = [
                ('id', 'vintage', 'day_typ', 'hour', 'climate',
                 'dmd_delt_sf', 't_in_delt', 'rh_in_delt', 't_out', 'rh_out',
                 'cloud_out', 'occ_frac', 'tsp_delt', 'lt_pwr_delt_pct',
                 'ven_delt_pct', 'mels_delt_pct', 'hrs_since_dr_st',
                 'hrs_since_dr_end', 'hrs_since_pc_end', 'hrs_since_pc_st',
                 'tsp_delt_lag', 'lt_pwr_delt_pct_lag', 'mels_delt_pct_lag',
                 'ven_delt_pct_lag'),
                (['<i4'] * 4 + ['<U25'] + ['<f8'] * 19)]
            self.coef_names_dtypes = [
                ('demand', 'temperature', 'co2', 'lighting',
                 'temperature_precool'), (['<f8'] * 5)]
        # Set data input file column names and data types for model
        # re-estimation; these will be the same across models
        elif mod_est is True:
            tmp_dmd_names_dtypes, co2_names_dtypes, lt_names_dtypes, \
                pc_tmp_names_dtypes = (
                    [('id', 'vintage', 'day_typ', 'day_num', 'hour', 'climate',
                      'dmd_delt_sf', 't_in_delt', 'rh_in_delt', 't_out',
                      'rh_out', 'cloud_out', 'occ_frac', 'tsp_delt',
                      'lt_pwr_delt_pct', 'ven_delt_pct', 'mels_delt_pct',
                      'hrs_since_dr_st', 'hrs_since_dr_end',
                      'hrs_since_pc_end', 'hrs_since_pc_st',
                      'tsp_delt_lag', 'lt_pwr_delt_pct_lag',
                      'mels_delt_pct_lag', 'ven_delt_pct_lag', 'pc_tmp_inc',
                      'pc_length'),
                     (['<i4'] * 5 + ['<U25'] + ['<f8'] * 21)]
                    for n in range(4))
            self.coef_names_dtypes = None
        # Set data input file column names and data types for model
        # prediction; these will be the same across models
        else:
            tmp_dmd_names_dtypes, co2_names_dtypes, lt_names_dtypes, \
                pc_tmp_names_dtypes = ([(
                    'Name', 'Hr', 't_out', 'rh_out', 'lt_nat',
                    'occ_frac', 'base_lt_frac', 'delt_price_kwh',
                    'hrs_since_dr_st',
                    'hrs_since_dr_end', 'hrs_since_pc_st',
                    'hrs_since_pc_end', 'tsp_delt', 'lt_pwr_delt_pct',
                    'ven_delt_pct', 'mels_delt_pct', 'tsp_delt_lag',
                    'lt_pwr_delt_pct_lag', 'ven_delt_pct_lag',
                    'mels_delt_pct_lag', 'pc_tmp_inc', 'pc_length',
                    'lt_pwr_delt'),
                    (['<U50'] + ['<f8'] * 22)] for n in range(4))
            self.coef_names_dtypes = None

        # For each model type, store information on input/output data file
        # names, input/output data file formats, model variables, and
        # diagnostic assessment figure file names
        self.mod_dict = {
            "temperature": {
                "io_data": [dmd_tmp_dat, stored_tmp],
                "io_data_names": tmp_dmd_names_dtypes,
                "var_names": ['ta_params', 'ta_sd', 'ta'],
                "fig_names": [
                    "traceplots_tmp.png", "postplots_tmp.png",
                    "ppcheck_tmp.png", "scatter_tmp.png",
                    "update_tmp.png"]
            },
            "demand": {
                "io_data": [dmd_tmp_dat, stored_dmd],
                "io_data_names": tmp_dmd_names_dtypes,
                "var_names": ['dmd_params', 'dmd_sd', 'dmd'],
                "fig_names": [
                    "traceplots_dmd.png", "postplots_dmd.png",
                    "ppcheck_dmd.png", "scatter_dmd.png",
                    "update_dmd.png"]
            },
            "co2": {
                "io_data": [co2_dat, stored_co2],
                "io_data_names": co2_names_dtypes,
                "var_names": ['co2_params', 'co2_sd', 'co2'],
                "fig_names": [
                    "traceplots_co2.png", "postplots_co2.png",
                    "ppcheck_co2.png", "scatter_co2.png",
                    "update_co2.png"]
            },
            "lighting": {
                "io_data": [lgt_dat, stored_lt],
                "io_data_names": lt_names_dtypes,
                "var_names": ['lt_params', 'lt_sd', 'lt'],
                "fig_names": [
                    "traceplots_lt.png", "postplots_lt.png",
                    "ppcheck_lt.png", "scatter_lt.png",
                    "update_lt.png"]
            },
            "temperature_precool": {
                "io_data": [pc_tmp_dat, stored_pc_tmp],
                "io_data_names": pc_tmp_names_dtypes,
                "var_names": ['ta_pc_params', 'ta_pc_sd', 'ta_pc'],
                "fig_names": [
                    "traceplots_tmp_pc.png", "postplots_tmp_pc.png",
                    "ppcheck_tmp_pc.png", "scatter_tmp_pc.png",
                    "update_tmp_pc.png"]
            }

        }
        self.predict_out = ("data", "recommendations.json")


class ModelDataLoad(object):
    """Load the data files needed to initialize, estimate, or run models.

    Attributes:
        dmd_tmp (numpy ndarray): Input data for demand and temperature
            model initialization.
        co2 (numpy ndarray): Input data for CO2 model initialization.
        lt (numpy ndarray): Input data for lighting model initialization.
        pc_tmp (numpy ndarray): Input data for pre-cooling model init.
        coefs (tuple): Path to CSV file with regression coefficients for
            use in model re-estimation.
        oaf_delt (numpy ndarray): Outdoor air adjustment fractions by DR
            strategy for use in model prediction.
        plug_delt (numpy ndarray): Plug load adjustment fractions by DR
            strategy for use in model prediction.
        price_delt (numpy ndarray): $/kWh incentive by DR strategy for use
            in model prediction.
        hr (numpy ndarray): Hours covered by model prediction input data
        strategy (numpy ndarray): Names of strategies to make predictions for.

    """

    def __init__(self, handyfilesvars, mod_init, mod_assess,
                 mod_est, update, ndays_update):
        """Initialize class attributes."""

        # Initialize OAF delta, plug load delta and price delta as None
        self.coefs, self.oaf_delt, self.plug_delt, self.price_delt = (
            None for n in range(4))
        # Data read-in for model initialization is specific to each type
        # of model (though demand/temperature share the same input data);
        if mod_init is True or mod_assess is True:
            # Read in data for initializing demand/temperature models
            self.dmd_tmp = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict[
                            "temperature"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "temperature"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "temperature"]["io_data_names"][1])
            # Read in data for initializing CO2 model
            self.co2 = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict["co2"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict["co2"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict["co2"]["io_data_names"][1])
            # Read in data for initializing lighting model
            self.lt = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict["lighting"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict["lighting"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict["lighting"]["io_data_names"][1])
            # Read in data for initializing temperature pre-cooling model
            self.pc_tmp = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict[
                            "temperature_precool"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "temperature_precool"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "temperature_precool"]["io_data_names"][1])
            # Stop routine if files were not properly read in
            if any([len(x) == 0 for x in [
                    self.dmd_tmp, self.co2, self.lt, self.pc_tmp]]):
                raise ValueError("Failure to read input file(s)")
            self.coefs = np.genfromtxt(
                path.join(base_dir, *handyfilesvars.coefs),
                skip_header=True, delimiter=',',
                names=handyfilesvars.coef_names_dtypes[0],
                dtype=handyfilesvars.coef_names_dtypes[1])
        # Data read-in for model re-estimation/prediction is common across
        # model types
        else:
            common_data = np.genfromtxt(
                path.join(base_dir,
                          *handyfilesvars.mod_dict[
                            "temperature"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "temperature"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "temperature"]["io_data_names"][1])
            # Restrict prediction input file to appropriate prediction or
            # model estimation update data (if applicable)
            if mod_est is False:
                self.hr = common_data['Hr']
                self.strategy = common_data['Name']
            elif mod_est is True and update is not None and \
                    ndays_update is not None:
                day_sequence = list(
                    range((update * ndays_update) + 1,
                          ((update + 1) * ndays_update) + 1))
                common_data = common_data[
                    np.in1d(common_data['day_num'], day_sequence)]
                self.hr = None
                self.strategy = None
            # Set inputs to demand, temperature, co2, and lighting models
            # from prediction input file
            self.dmd_tmp, self.co2, self.lt, self.pc_tmp = (
                common_data for n in range(4))

            # Set plug load delta and price delta to values from prediction
            # input file (these are not predicted via a Bayesian model)
            if mod_est is False:
                self.oaf_delt = common_data['ven_delt_pct']
                self.plug_delt = common_data['mels_delt_pct']
                self.price_delt = common_data['delt_price_kwh']
                self.pc_temp = common_data['pc_tmp_inc']
            else:
                self.oaf_delt, self.plug_delt, self.price_delt = (
                    None for n in range(2))


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
            self, handyfilesvars, mod_init, mod_est, mod_assess, mod, data):
        """Initialize class attributes."""

        # If model is being initialized and model assessment is requested,
        # set the portion of the data that should be used across models for
        # training vs. testing each model
        if (mod_init is True and mod_assess is True) or mod_assess is True:
            train_pct = 0.7
        else:
            train_pct = None

        if mod == "temperature" or mod == "demand":
            # If model is being initialized and model assessment is requested,
            # or previously initialized model is being assessed,
            # set training/testing indices to use on the demand/temp. data
            if (mod_init is True and mod_assess is True) or mod_assess is True:
                # Set training indices
                self.train_inds = np.random.randint(
                    0, len(data.dmd_tmp),
                    size=int(len(data.dmd_tmp) * train_pct))
                # Set testing indices
                self.test_inds = [
                    x for x in range(len(data.dmd_tmp)) if
                    x not in self.train_inds]

            # Initialize variables for temperature and demand models

            # Whole building occupancy fraction
            occ_frac = data.dmd_tmp['occ_frac']
            # Outdoor air temperature
            temp_out = data.dmd_tmp['t_out']
            # Outdoor relative humidity
            rh_out = data.dmd_tmp['rh_out']
            # Temperature set point difference
            tmp_delta = data.dmd_tmp['tsp_delt']
            # Temperature set point difference lag
            tmp_delta_lag = data.dmd_tmp['tsp_delt_lag']
            # Hours since DR event started (adjustment to normal op. condition)
            dr_start = data.dmd_tmp['hrs_since_dr_st']
            # Hours since DR event ended (adjustment to normal op. condition)
            dr_end = data.dmd_tmp['hrs_since_dr_end']
            # Hours since pre-cooling ended (if applicable)
            pcool_duration = data.dmd_tmp['pc_length']
            # Magnitude of pre-cooling temperature offset
            pcool_magnitude = data.dmd_tmp['pc_tmp_inc']
            # OA ventilation fraction reduction
            oaf_delta = data.dmd_tmp['ven_delt_pct']
            # OA ventilation fraction reduction lag
            oaf_delta_lag = data.dmd_tmp['ven_delt_pct_lag']
            # Lighting fraction reduction
            lt_delta = data.dmd_tmp['lt_pwr_delt_pct']
            # Lighting fraction reduction lag
            lt_delta_lag = data.dmd_tmp['lt_pwr_delt_pct_lag']
            # Plug load power fraction reduction
            plug_delta = data.dmd_tmp['mels_delt_pct']
            # Plug load power fraction reduction
            plug_delta_lag = data.dmd_tmp['mels_delt_pct_lag']
            # Set a vector of ones for intercept estimation
            intercept = np.ones(len(occ_frac))
            # Initialize interactive terms
            # Temp. set point and lighting difference
            tmp_lt = tmp_delta * lt_delta
            # Temp. set point and outdoor air fraction difference
            tmp_oaf = tmp_delta * oaf_delta
            # Pre-cool duration, and magnitude
            pcool_interact = pcool_duration * pcool_magnitude
            # Temp. set point difference, outdoor temperature, and DR st. time
            tmp_out_tmp_delt_dr_start = tmp_delta * temp_out * dr_start
            # Temp. set point diff, outdoor temperature
            tmp_out_tmp_delt = tmp_delta * temp_out
            # Outdoor temperature, DR st. time
            tmp_out_dr_start = temp_out * dr_start
            # Temp set point diff, since DR started
            tmp_delt_dr_start = tmp_delta * dr_start

            # Set model input (X) variables
            self.X_all = np.stack([
                intercept, temp_out, rh_out, occ_frac, tmp_delta, lt_delta,
                oaf_delta, plug_delta, dr_start, dr_end, tmp_delta_lag,
                lt_delta_lag, plug_delta_lag, oaf_delta_lag, pcool_magnitude,
                pcool_duration, tmp_lt, tmp_oaf,
                tmp_out_tmp_delt, tmp_out_dr_start, tmp_delt_dr_start,
                pcool_interact, tmp_out_tmp_delt_dr_start], axis=1)
            # Set model output (Y) variable for estimation cases
            if mod_init is True or mod_est is True or mod_assess is True:
                if mod == "temperature":
                    self.Y_all = data.dmd_tmp['t_in_delt']
                else:
                    self.Y_all = data.dmd_tmp['dmd_delt_sf']

        elif mod == "co2":
            # If model is being initialized and model assessment is requested,
            # or previously initialized model is being assessed,
            # set training/testing indices to use on the demand/temp. data
            if (mod_init is True and mod_assess is True) or mod_assess is True:
                # Set training indices
                self.train_inds = np.random.randint(
                    0, len(data.co2), size=int(len(data.co2) * train_pct))
                # Set testing indices
                self.test_inds = [
                    x for x in range(len(data.co2)) if
                    x not in self.train_inds]

            # Outdoor air temperature
            temp_out_co2 = data.co2['t_out']
            # Relative humidity
            rh_out_co2 = data.co2['rh_out']
            # Occupancy fraction
            occ_frac_co2 = data.co2['occ_frac']
            # Temperature set point difference
            tmp_delt_co2 = data.co2['tsp_delt']
            # Outdoor air fraction setpoint difference
            oaf_delt_co2 = data.co2['ven_delt_pct']
            # Temperature set point difference, previous time step
            tmp_delt_lag_co2 = data.co2['tsp_delt_lag']
            # Outdoor set point difference, previous time step
            oaf_delt_lag_co2 = data.co2['ven_delt_pct_lag']
            # Temp. set point and outdoor air fraction difference interaction
            tmp_oaf_co2 = tmp_delt_co2 * oaf_delt_co2
            # Outdoor temperature and temperature delta
            tmp_out_tmp_delt = temp_out_co2 * tmp_delt_co2
            # Outdoor humidity and temperature delta
            rh_out_tmp_delt = rh_out_co2 * tmp_delt_co2

            # Intercept term
            intercept_co2 = intercept = np.ones(len(occ_frac_co2))

            # Initialize variables for CO2 model
            self.X_all = np.stack([
                intercept_co2, temp_out_co2, rh_out_co2, occ_frac_co2,
                tmp_delt_co2, oaf_delt_co2, tmp_delt_lag_co2, oaf_delt_lag_co2,
                tmp_oaf_co2, tmp_out_tmp_delt, rh_out_tmp_delt], axis=1)
            # Set model output (Y) variable for estimation cases
            if mod_init is True or mod_est is True or mod_assess is True:
                self.Y_all = data.co2['co2_in_delt']

        elif mod == "lighting":
            # If model is being initialized and model assessment is requested,
            # or previously initialized model is being assessed,
            # set training/testing indices to use on the demand/temp. data
            if (mod_init is True and mod_assess is True) or mod_assess is True:
                # Draw a training subset from the full dataset
                # Set training indices
                self.train_inds = np.random.randint(
                    0, len(data.lt), size=int(len(data.lt) * train_pct))
                # Set testing indices
                self.test_inds = [
                    x for x in range(len(data.lt)) if
                    x not in self.train_inds]

            # Natural illuminance
            lt_out = data.lt['lt_nat']
            # Base lighting schedule
            lt_base = data.lt['base_lt_frac']
            # Lighting power difference
            lt_delt_pct = data.lt['lt_pwr_delt_pct']
            # Natural illuminance * lighting power reduction
            lt_nat_pwr = lt_out * lt_delt_pct
            # Base lighting * lighting power reduction
            lt_base_delt = lt_base * lt_delt_pct
            # Natural illuminance * base lighting * lighting
            # Intercept term
            intercept_lt = intercept = np.ones(len(lt_out))

            # Set model input (X) variables
            self.X_all = np.stack([
                intercept_lt, lt_delt_pct, lt_nat_pwr, lt_base_delt], axis=1)
            # Set model output (Y) variable for estimation cases
            if mod_init is True or mod_est is True or mod_assess is True:
                self.Y_all = data.lt['lt_in_delt_pct']

        elif mod == "temperature_precool":
            # If model is being initialized and model assessment is requested,
            # or previously initialized model is being assessed,
            # set training/testing indices to use on the demand/temp. data
            if (mod_init is True and mod_assess is True) or mod_assess is True:
                # Set training indices
                self.train_inds = np.random.randint(
                    0, len(data.pc_tmp),
                    size=int(len(data.pc_tmp) * train_pct))
                # Set testing indices
                self.test_inds = [
                    x for x in range(len(data.pc_tmp)) if
                    x not in self.train_inds]

            # Initialize variables for temperature and demand models

            # Whole building occupancy fraction
            occ_frac = data.pc_tmp['occ_frac']
            # Outdoor air temperature
            temp_out = data.pc_tmp['t_out']
            # Outdoor relative humidity
            rh_out = data.pc_tmp['rh_out']
            # Temperature set point difference
            tmp_delta = data.pc_tmp['tsp_delt']
            # Temperature set point difference lag
            tmp_delta_lag = data.pc_tmp['tsp_delt_lag']
            # Hours since pre-cooling started
            pcool_start = data.pc_tmp['hrs_since_pc_st']
            # Temp. set point difference, outdoor temp.
            tmp_out_tmp_delt = temp_out * tmp_delta
            # Outdoor temp., since pre-cooling started
            tmp_out_pcool_start = temp_out * pcool_start
            # Temp. set point difference, since pre-cooling started
            tmp_delta_pcool_start = tmp_delta * pcool_start
            # Temp. set point difference, outdoor temperature, and DR st. time
            tmp_out_pc_start = tmp_delta * temp_out * pcool_start
            # Set a vector of ones for intercept estimation
            intercept = np.ones(len(occ_frac))

            # Set model input (X) variables
            self.X_all = np.stack([
                intercept, temp_out, rh_out, occ_frac, tmp_delta, pcool_start,
                tmp_out_tmp_delt, tmp_out_pcool_start, tmp_delta_pcool_start,
                tmp_out_pc_start], axis=1)
            # Set model output (Y) variable for estimation cases
            if mod_init is True or mod_est is True or mod_assess is True:
                self.Y_all = data.pc_tmp['t_in_delt']


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

    # Initialize building type and square footage
    bldg_type_vint = opts.bldg_type
    sf = opts.bldg_sf

    # Instantiate useful input file and variable data object
    handyfilesvars = UsefulFilesVars(
        bldg_type_vint, opts.mod_init, opts.mod_est, opts.mod_assess)

    # Set numpy and theano RNG seeds for consistency across runs
    np.random.seed(123)
    th_rng = MRG_RandomStreams()
    th_rng.seed(123)

    # Proceed with model inference only if user flags doing so; otherwise,
    # use previously estimated models to conduct assessment or make predictions
    if opts.mod_init is True:

        print("Loading input data...", end="", flush=True)
        # Read-in input data
        dat = ModelDataLoad(handyfilesvars, opts.mod_init, opts.mod_assess,
                            opts.mod_est, update=None, ndays_update=None)
        print("Complete.")

        # Loop through all model types (temperature, demand, co2, lighting)
        for mod in handyfilesvars.mod_dict.keys():

            print("Initializing " + mod + " sub-model variables...",
                  end="", flush=True)
            # Initialize variable inputs and outputs for the given model type
            iog = ModelIO(handyfilesvars, opts.mod_init, opts.mod_est,
                          opts.mod_assess, mod, dat)
            # Restrict model data to training subset (pertains to model
            # initialization only)
            iot = ModelIOTrain(iog, opts.mod_init, opts.mod_assess)
            print("Complete.")

            # Perform model inference
            with pm.Model() as var_mod:
                print("Setting " + mod + " sub-model priors and likelihood...",
                      end="", flush=True)
                # Set parameter priors (betas, error)
                params = pm.Normal(
                    handyfilesvars.mod_dict[mod]["var_names"][0], 0, 10,
                    shape=(iot.X.shape[1]))
                sd = pm.HalfNormal(
                    handyfilesvars.mod_dict[mod]["var_names"][1], 20)
                # Likelihood of outcome estimator
                est = pm.math.dot(iot.X, params)
                # Likelihood of outcome
                var = pm.Normal(handyfilesvars.mod_dict[mod]["var_names"][2],
                                mu=est, sd=sd, observed=iot.Y)
                print("Complete.")
                # Draw posterior samples
                trace = pm.sample(chains=2, cores=1, init="advi")

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
                refs = list(
                    dat.coefs[mod][np.where(np.isfinite(dat.coefs[mod]))])
                run_mod_assessment(handyfilesvars, trace, mod, iog, refs)
                print("Complete.")

    elif opts.mod_est is True:

        # Set total number of model updates to execute
        updates = 8
        # Set number of days/events per update data batch
        ndays_update = 10
        # Initialize model parameter traces across all updates, stored in list
        traces = []
        # Loop through all model update instances and update parameter
        # estimates in accordance with new data added with each update
        for update in range(updates):
            print("Loading input data...", end="", flush=True)
            # Read-in input data
            dat = ModelDataLoad(handyfilesvars, opts.mod_init, opts.mod_assess,
                                opts.mod_est, update, ndays_update)
            print("Complete.")

            # Loop through model types (restrict to demand model for now)
            for mod in ["demand"]:     #handyfilesvars.mod_dict.keys():
                print("Initializing " + mod + " sub-model variables...",
                      end="", flush=True)
                # Initialize variable inputs/outputs for the given model type
                iog = ModelIO(handyfilesvars, opts.mod_init, opts.mod_est,
                              opts.mod_assess, mod, dat)
                # Finalize variable inputs/outputs for the given model type
                iot = ModelIOTrain(iog, opts.mod_init, opts.mod_assess)
                print("Complete.")

                # Perform model inference
                with pm.Model() as var_mod:
                    # For the first update, reload existing trace from model
                    # initialization run
                    if update == 0:
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
                    # Pull beta trace and set shorthand name
                    t_params = trace[
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
                        handyfilesvars.mod_dict[mod]["var_names"][1], trace[
                            handyfilesvars.mod_dict[mod]["var_names"][1]])
                    # Likelihood of outcome estimator
                    est = pm.math.dot(iot.X, params)
                    # Likelihood of outcome
                    var = pm.Normal(
                        handyfilesvars.mod_dict[mod]["var_names"][2],
                        mu=est, sd=sd, observed=iot.Y)
                    print("Complete.")
                    # Draw posterior samples
                    trace = pm.sample(chains=2, cores=1, init="advi")
                    # Append current updates' traces to the traces from
                    # all previous updates
                    traces.append(trace)

            # After the last update, generate some diagnostic plots showing
            # how parameter estimates evolved across all updates
            if update == (updates - 1):
                plot_updating(
                    handyfilesvars,
                    handyfilesvars.mod_dict[mod]["var_names"][0], traces, mod)

            # # Store model, trace, and predictor variables
            # with open(path.join(base_dir, *handyfilesvars.mod_dict[mod][
            #         "io_data"][1]), 'wb') as co_s:
            #     print("Writing out " + mod + " sub-model...",
            #           end="", flush=True)
            #     pickle.dump({'trace': trace, 'model': var_mod}, co_s)

            # If model assessment is desired, generate diagnostic plots
            if opts.mod_assess is True:
                print("Starting " + mod + " sub-model assessment...", end="",
                      flush=True)
                # Set reference coefficient values, estimated using a
                # frequentist regression approach
                refs = list(
                    dat.coefs[mod][np.where(np.isfinite(dat.coefs[mod]))])
                run_mod_assessment(handyfilesvars, trace, mod, iog, refs)
                print("Complete.")

    elif opts.mod_assess is True:

        print("Loading input data...", end="", flush=True)
        # Read-in input data
        dat = ModelDataLoad(handyfilesvars, opts.mod_init, opts.mod_assess,
                            opts.mod_est, update=None, ndays_update=None)
        print("Complete.")

        # Loop through all model types (temperature, demand, co2, lighting)
        for mod in handyfilesvars.mod_dict.keys():
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
                          opts.mod_assess, mod, dat)

            # Set reference coefficient values, estimated using a frequentist
            # regression approach
            refs = list(
                dat.coefs[mod][np.where(np.isfinite(dat.coefs[mod]))])
            run_mod_assessment(handyfilesvars, trace, mod, iog, refs)
            print("Complete.")

    elif opts.mod_cosimulate is True:
        print("Cosimulation....")
        handyfilesvars = UsefulFilesVars(
            bldg_type_vint, opts.mod_init, opts.mod_est, opts.mod_assess)
        #cosimulate(handyfilesvars, bldg_type_vint, sf)
        cosimulate(handyfilesvars, bldg_type_vint, sf)
        print("Complete.")

    else:
        # Generate predictions for a next-day DR event with conditions and
        # candidate strategies described by an updated input file
        # (test_predict.csv, which is loaded into handyfilesvars)
        predict_out = gen_recs(handyfilesvars, sf)

        # Write summary dict with predictions out to JSON file
        with open(path.join(
                base_dir, *handyfilesvars.predict_out), "w") as jso:
            json.dump(predict_out, jso, indent=2)


def gen_recs(handyfilesvars, sf):

    # Notify user of input data read
    print("Loading input data...")
    # Read-in input data for scenario
    dat = ModelDataLoad(
        handyfilesvars, opts.mod_init, opts.mod_assess,
        opts.mod_est, update=None, ndays_update=None)
    # Set number of hours to predict across
    n_hrs = len(np.unique(dat.hr))
    # Find names of candidate DR strategies
    names = dat.strategy[np.where(dat.hr == 1)]
    # Set number of DR strategies to predict across
    n_choices = len(names)
    # Set number of samples to draw. for predictions
    n_samples = 1000
    # Initialize posterior predictive data dict
    pp_dict = {
        key: [] for key in handyfilesvars.mod_dict.keys()}
    # Sample noise to use in the choice model
    rand_elem = np.random.normal(
        loc=0, scale=1, size=(n_samples, n_choices))
    # Initialize a numpy array that stores the count of the number of
    # times each candidate DR strategy is selected across simulated hours
    counts = np.zeros(n_choices)
    # Initialize a total count to use in normalizing number of selections
    # by strategy such that it is reported out as a % of simulated hours
    counts_denom = 0

    # Use latest set of coefficients from DCE future scenario results
    # (Order: economic benefit, temperature, temperature precool, lighting,
    # OAF pct delta, plug load pct delta)

    # ORIGINAL
    # betas_choice = np.array([
    #     0.000345, -0.491951, 0.127805, -1.246971, 0.144217, -2.651832])

    # UPDATED
    betas_choice = np.array([0.025, -0.31, 0.14, -1.4, 0.05, 0.26])

    # Loop through all hours considered for the event (event plus rebound)
    for hr in range(n_hrs):
        print("Making predictions for hour " + str(hr+1))
        # Set data index that is unique to the current hour
        inds = np.where(dat.hr == (hr+1))
        for mod in handyfilesvars.mod_dict.keys():
            # Reload trace
            with open(path.join(base_dir, *handyfilesvars.mod_dict[mod][
                    "io_data"][1]), 'rb') as store:
                trace = pickle.load(store)['trace']
            pp_dict[mod] = run_mod_prediction(
                handyfilesvars, trace, mod, dat, n_samples, inds)
        # Multiply change in demand/sf by sf and price delta to get total
        # cost difference for the operator; reflect DCE units of $100
        cost_delt = (
            pp_dict["demand"]['dmd'] * sf * dat.price_delt[inds]) / 100
        # Extend oaf delta values for each choice across all samples
        oaf_delt = np.tile(dat.oaf_delt[inds], (n_samples, 1))
        # Extend plug load delta values for each choice across all samples
        plug_delt = np.tile(dat.plug_delt[inds], (n_samples, 1))
        # Extend pre-cooling values for each choice across all samples
        pc_temp = np.tile(dat.pc_temp[inds], (n_samples, 1))
        # Extend intercept input for each choice across all samples
        # NOTE: CURRENTLY NO INTERCEPT TERM IN DCE ANALYSIS OUTPUTS
        # intercept = np.tile(np.ones(n_choices), (n_samples, 1))
        # Stack all model inputs into a single array
        # x_choice = np.stack([
        #     cost_delt, pp_dict["temperature"]["ta"],
        #     pp_dict["co2"]["co2"], pp_dict["lighting"]["lt"],
        #     plug_delt, intercept])
        x_choice = np.stack([
            cost_delt, pp_dict["temperature"]["ta"],
            pc_temp, pp_dict["lighting"]["lt"], oaf_delt, plug_delt])
        # Multiply model inputs by betas to yield choice logits
        choice_logits = np.sum([x_choice[i] * betas_choice[i] for
                               i in range(len(x_choice))], axis=0) + \
            rand_elem
        # Softmax transformation of logits into choice probabilities
        choice_probs = softmax(choice_logits, axis=1)
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
    # Store summary of the percentage of simulated hours that each
    # DR strategy was predicted to be selected in a dict
    predict_out = {
        "notes": (
            "Percentage of simulations in which each candidate DR "
            "strategy is chosen for current event"),
        "units": "%",
        "predictions": {
            x: round(((y / counts_denom) * 100), 1) for
            x, y in zip(names, counts)}
    }

    return predict_out


def simBaseline(cz, baseline_csv, fmu_path, sf, sch_base, io_dict):
    """ This function is used to collect results from  baseline runs
    @params:
    cz - climate zone
    baseline_csv - path to csv file
    fmu_path - path to the fmu format of .idf file
    sf - square feet
    """

    # starting date
    dt_jan1 = datetime(2006, 1, 1, 1)
    # simulated number of days
    sim_days=365
    # starting simulated time
    tStart = 0
    # ending simulated time
    tStop = 3600*1*24*sim_days  # change the timestep in EPlus to 1
    # timestep
    hStep = 3600 # 60 mins

    # numpy array representationof steps
    t = np.arange(tStart, tStop, hStep)
    n_steps = len(t)
    # load and initialize fmu file
    model = load_fmu(fmu_path, log_level=7)
    model.initialize(tStart, tStop)

    io_vals = np.empty(shape=(len(io_dict), n_steps))

    i = 0
    # Main simulation loop
    while True:

        hour = int((t[i]/3600)%24)
        io_vals[io_dict['inven']][i] = sch_base[0][hour]
        io_vals[io_dict['inclg']][i] = sch_base[2][hour]
        io_vals[io_dict['inlgt']][i] = sch_base[3][hour]
        io_vals[io_dict['inplg']][i] = sch_base[4][hour]

        ###############################################################
        model.set(['InMELsSch','InLightSch','InCoolingSch', 'InVentSch'],
                  [io_vals[io_dict['inplg']][i],io_vals[io_dict['inlgt']][i],
                   io_vals[io_dict['inclg']][i],io_vals[io_dict['inven']][i]])

        model.do_step(current_t = t[i], step_size=hStep, new_step=True)

        # Get the outputs of the simulation
        temp_np = np.array([])
        ppl_np = np.array([])
        rh_np = np.array([])
        illum_np = np.array([])

        for zoneid in range(0, 34):
            temp_np = np.append(temp_np, (model.get('ZAT_' + str(zoneid))))
            ppl_np = np.append(ppl_np, (model.get('PEOPLE_' + str(zoneid))))
            rh_np = np.append(rh_np, (model.get('ZRH_' + str(zoneid))))
            if zoneid < 23:
                illum_np = np.append(illum_np, (model.get('ZNatIllum_' + str(zoneid))))


        # temp_wght = np.sum((temp_np * ppl_np)) / np.sum(ppl_np)
        # ppl_wght = (np.sum(ppl_np * ppl_np) / np.sum(ppl_np)) / np.sum(ppl_np)
        rh_wght = np.sum((rh_np * ppl_np)) / np.sum(ppl_np)
        illum_wght = np.mean(illum_np)
        temp_wght = np.mean(temp_np)
        ppl_wght = np.mean(ppl_np)
        io_vals[io_dict['zlux']][i] = illum_wght

        io_vals[io_dict['zat']][i] = (temp_wght * 9 / 5) + 32 #farenheit
        io_vals[io_dict['zati']][i] = (model.get('ZAT_31') * 9 / 5) + 32
        io_vals[io_dict['zpp']][i] = ppl_wght
        io_vals[io_dict['zrh']][i] = rh_wght

        io_vals[io_dict['zcot']][i] = model.get('ZoneCOTwo')
        io_vals[io_dict['zpmv']][i] = model.get('ZonePMV')
        io_vals[io_dict['zlgt']][i] = model.get('LightsEnergy') / 3600000 # kilowatt-hour
        io_vals[io_dict['zplg']][i] = model.get('MelsEnergy') / 3600000
        io_vals[io_dict['pwr']][i] = model.get('BldgPwr') / 1000

        io_vals[io_dict['osk']][i] = model.get('OutSkyClear')
        io_vals[io_dict['oat']][i] = (model.get('OutDrybulb') * 9 / 5) + 32 #farenheit
        io_vals[io_dict['orh']][i] = model.get('OutRH')

        # z_clg1[i] = model.get('CoolingEnergy1') / 3600000
        # z_clg2[i] = model.get('CoolingEnergy2') / 3600000
        # outdoor_natlt[i] = model.get('OutSkyIllum') + model.get('OutBeamIllum')

        i += 1
        if (i == n_steps):
            break

    #epout_df = pd.read_csv(path.join(ep_dir,bldg_file + '.csv'), parse_dates=True)
    hrtime = pd.date_range(start=dt_jan1, periods=8760, freq='60min').values
    result = pd.DataFrame(data={'datetime':hrtime,'ID':21,'Vintages':'2004',
        'Day.type':1,'Day.number':1,'Hour.number':0,'Climate.zone':cz,
        'Demand.Power.sf.':io_vals[io_dict['pwr']],'Indoor.Temp.F.':io_vals[io_dict['zat']],
        'Tzonei':io_vals[io_dict['zati']],'Nat.Lt.':io_vals[io_dict['zlux']],
        'Indoor.Humid.':io_vals[io_dict['zrh']],'Outdoor.Temp.F.':io_vals[io_dict['oat']],
        'Outdoor.Humid.':io_vals[io_dict['orh']],'Outdoor.Sky.Clearness.':io_vals[io_dict['osk']],
        'Occ.Fraction.':io_vals[io_dict['zpp']],
        'Cooling.Setpoint.':io_vals[io_dict['inclg']],'Lighting.Power.pct.':io_vals[io_dict['inlgt']],
        'Ventilation.pct.':io_vals[io_dict['inven']],'MELs.pct.':io_vals[io_dict['inplg']]
    })

    result.to_csv(baseline_csv, index=False)


def updateTestPredictCSV(handyfilesvars, baseline_csv, hr_dr_start, hr_dr_end, hrs_rbnd):

    base_df = pd.read_csv(baseline_csv, parse_dates=True)
    predict_csv = path.join(*handyfilesvars.mod_dict["temperature"]["io_data"][0])
    predict_df = pd.read_csv(predict_csv).head(19)

    hrs_dr = list(range(hr_dr_start,(hr_dr_end + hrs_rbnd)))

    d = 0
    # the first DR event hour
    predict_df.loc[:,'Hr'] = d + 1
    predict_df.loc[:,'OAT'] = base_df.iloc[hrs_dr[d]]['Outdoor.Temp.F.']
    predict_df.loc[:,'RH'] = base_df.iloc[hrs_dr[d]]['Outdoor.Humid.']
    predict_df.loc[:,'Lt_Nat'] = base_df.iloc[hrs_dr[d]]['Nat.Lt.']
    predict_df.loc[:,'Lt_Base'] = 0
    predict_df.loc[:,'Occ_Frac'] = base_df.iloc[hrs_dr[d]]['Occ.Fraction.']
    predict_df.loc[:,'Delt_Price_kWh'] = 0.1

    # the next DR event and rebound hours
    for d in range(1,len(hrs_dr)):
        temp_df = predict_df.copy().head(19)

        temp_df.loc[:,'Hr'] = d + 1
        temp_df.loc[:,'OAT'] = base_df.iloc[hrs_dr[d]]['Outdoor.Temp.F.']
        temp_df.loc[:,'RH'] = base_df.iloc[hrs_dr[d]]['Outdoor.Humid.']
        temp_df.loc[:,'Lt_Nat'] = base_df.iloc[hrs_dr[d]]['Nat.Lt.']
        temp_df.loc[:,'Lt_Base'] = 0
        temp_df.loc[:,'Occ_Frac'] = base_df.iloc[hrs_dr[d]]['Occ.Fraction.']
        temp_df.loc[:,'Delt_Price_kWh'] = 0.1
        temp_df.loc[:,'h_DR_Start'] = d if (d < (len(hrs_dr) - hrs_rbnd)) else 0
        temp_df.loc[:,'h_DR_End'] = (d + 3 - len(hrs_dr)) if (d >= (len(hrs_dr) - hrs_rbnd)) else 0 #####
        temp_df.loc[:,'h_PCool_End'] = temp_df.loc[:,'h_PCool_End'] * (d + 1)
        if (d >= (len(hrs_dr) - hrs_rbnd)): temp_df.loc[:,'Delt_CoolSP'] = 0
        if (d >= (len(hrs_dr) - hrs_rbnd)): temp_df.loc[:,'Delt_LgtPct'] = 0
        if (d >= (len(hrs_dr) - hrs_rbnd)): temp_df.loc[:,'Delt_OAVent_Pct'] = 0
        if (d >= (len(hrs_dr) - hrs_rbnd)): temp_df.loc[:,'Delt_PL_Pct'] = 0
        temp_df.loc[:,'Delt_CoolSP_Lag'] = (temp_df.loc[:,'Delt_CoolSP_Lag'] * -1) if (d == (len(hrs_dr) - hrs_rbnd)) else 0
        temp_df.loc[:,'Delt_LgtPct_Lag'] = (temp_df.loc[:,'Delt_LgtPct_Lag'] * -1) if (d == (len(hrs_dr) - hrs_rbnd)) else 0
        temp_df.loc[:,'Delt_OAVent_Pct_Lag'] = (temp_df.loc[:,'Delt_OAVent_Pct_Lag'] * -1) if (d == (len(hrs_dr) - hrs_rbnd)) else 0
        temp_df.loc[:,'Delt_PL_Pct_Lag'] = (temp_df.loc[:,'Delt_PL_Pct_Lag'] * -1) if (d == (len(hrs_dr) - hrs_rbnd)) else 0
        if (d >= (len(hrs_dr) - hrs_rbnd)): temp_df.loc[:,'Pcool_Mag'] = 0
        if (d >= (len(hrs_dr) - hrs_rbnd)): temp_df.loc[:,'Pcool_Dur'] = 0

        predict_df = predict_df.append(temp_df)

    predict_df.to_csv(predict_csv, index=False)


def updateTestUpdateCSV(cz, baseline_csv, update_csv, day_i, dt_cosim_i, io_dict, io_vals, dts_dr_start, dts_dr_end, hrs_dr_start, hrs_dr_end, hrs_rbnd):
    # the file with Na's data at the initial, cleared, and repopulated from 
    # Energyplus output at each DR event (per-day) and 2-hour rebound period.
    # Run, for example, every 5 events and re-initialized with new data.
    out_dt_dr_start = (dts_dr_start[dt_cosim_i] + timedelta(hours=1))
    out_dt_dr_end = (dts_dr_end[dt_cosim_i] + timedelta(hours=2))
    out_hr_dr_start = hrs_dr_start[dt_cosim_i] + 1
    out_hr_dr_end = hrs_dr_end[dt_cosim_i] + 1 + hrs_rbnd

    # print('TIME')
    # print('dt_cosim_index {!s} out_dt_dr_start {!s} out_dt_dr_end {!s} out_hr_dr_start {!s} out_hr_dr_end {!s}'.
    #     format(dt_cosim_index, out_dt_dr_start, out_dt_dr_end, out_hr_dr_start, out_hr_dr_end))
    
    #hrtime = pd.date_range(start=out_dt_dr_start, end=out_dt_dr_end, freq='60min').values
    hrtime = pd.date_range(start=out_dt_dr_start, end=out_dt_dr_end, freq='H').to_pydatetime()
    dr_r = np.s_[out_hr_dr_start:out_hr_dr_end]
    update_res = pd.DataFrame(data={'datetime':hrtime,
        'ID': 21,
        'Vintages': '2004',
        'Day.type': 1,
        'Day.number': day_i,
        'Hour.number': [x.hour for x in hrtime],
        'Climate.zone': cz,
        'Demand.Power.sf.': io_vals[io_dict['pwr']][dr_r],
        'Indoor.Temp.F.': io_vals[io_dict['zat']][dr_r],
        'Indoor.Humid.': io_vals[io_dict['zrh']][dr_r],
        'Outdoor.Temp.F.': io_vals[io_dict['oat']][dr_r],
        'Outdoor.Humid.': io_vals[io_dict['orh']][dr_r],
        'Outdoor.Sky.Clearness.': io_vals[io_dict['osk']][dr_r],
        'Occ.Fraction.': io_vals[io_dict['zpp']][dr_r],
        'Cooling.Setpoint.': io_vals[io_dict['inclg']][dr_r],
        'Lighting.Power.pct.': io_vals[io_dict['inlgt']][dr_r],
        'Ventilation.pct.': io_vals[io_dict['inven']][dr_r],
        'MELs.pct.': io_vals[io_dict['inplg']][dr_r],
        'Since.DR.Started.': io_vals[io_dict['hdrst']][dr_r],
        'Since.DR.Ended.': io_vals[io_dict['hdren']][dr_r],
        'Since.Pre.cooling.Ended.': 0,
        'Since.Pre.cooling.Started.': 0,
        'Cooling.Setpoint.Diff.One.Step.': io_vals[io_dict['inclg_delt']][dr_r],
        'Lighting.Power.Diff.pct.One.Step.': io_vals[io_dict['inlgt_delt']][dr_r],
        'MELs.Power.Diff.pct.One.Step.': io_vals[io_dict['inplg_delt']][dr_r],
        'Ventilation.Diff.pct.One.Step.': io_vals[io_dict['inven_delt']][dr_r],
        'Pre.cooling.Temp.Increase.': 0,
        'Pre.cooling.Duration.': 0
    })

    baseline_df = pd.read_csv(baseline_csv, parse_dates=True, index_col='datetime')
    update_res = update_res.set_index('datetime')

    update_res['Demand.Power.Diff.sf.'] = (update_res['Demand.Power.sf.'] - baseline_df['Demand.Power.sf.']) * -1
    update_res['Indoor.Temp.Diff.F.'] = update_res['Indoor.Temp.F.'] - baseline_df['Indoor.Temp.F.']
    update_res['Indoor.Humid.Diff.'] = update_res['Indoor.Humid.'] - baseline_df['Indoor.Humid.']
    update_res['Cooling.Setpoint.Diff.'] = (update_res['Cooling.Setpoint.'] - baseline_df['Cooling.Setpoint.'])
    update_res['Lighting.Power.Diff.pct.'] = (update_res['Lighting.Power.pct.'] - baseline_df['Lighting.Power.pct.'])
    update_res['Ventilation.Diff.pct.'] = (update_res['Ventilation.pct.'] - baseline_df['Ventilation.pct.'])
    update_res['MELs.Diff.pct.'] = (update_res['MELs.pct.'] - baseline_df['MELs.pct.']) * -1

    update_res.reset_index(inplace=True)
    update_res = update_res[[
        'ID','Vintages','Day.type','Day.number','Hour.number',
        'Climate.zone','Demand.Power.Diff.sf.','Indoor.Temp.Diff.F.',
        'Indoor.Humid.Diff.','Outdoor.Temp.F.','Outdoor.Humid.',
        'Outdoor.Sky.Clearness.','Occ.Fraction.','Cooling.Setpoint.Diff.',
        'Lighting.Power.Diff.pct.','Ventilation.Diff.pct.',
        'MELs.Diff.pct.','Since.DR.Started.','Since.DR.Ended.',
        'Since.Pre.cooling.Ended.','Since.Pre.cooling.Started.',
        'Cooling.Setpoint.Diff.One.Step.','Lighting.Power.Diff.pct.One.Step.',
        'MELs.Power.Diff.pct.One.Step.','Ventilation.Diff.pct.One.Step.',
        'Pre.cooling.Temp.Increase.','Pre.cooling.Duration.']]

    if os.path.exists(update_csv):
        update_res.to_csv(update_csv, mode='a', header=False, index=False)
    else:
        update_res.to_csv(update_csv, index=False)


def cosimulate(handyfilesvars, bldg_type_vint, sf):
    # get the data of the choice strategy and store to respected schedule values
    climate_zones = ['2A','2B','3A','3B','3C','4A','4B','4C','5A','5B','5C','6A','6B','7A']
    # mock up test to always select climate zone 3A - Atlanta, GA
    cz = '3A'

    fmu_dir = 'fmu_files'
    cosim_dir = 'cosim_outputs'
    data_dir = 'data'

    bldg_file = 'Baseline_MediumOfficeDetailed_2004_' + cz
    fmu_path = path.join(fmu_dir, bldg_file + '.fmu')
    update_csv = path.join(data_dir + 'test_update.csv')
    baseline_csv = path.join(cosim_dir, bldg_file + '.csv')

    # starting date
    dt_jan1 = datetime(2006, 1, 1, 1)
    # starting date of cosimulation
    dt_cosim_start = datetime(2006, 8, 21)
    # ending date of cosimulation
    dt_cosim_end = datetime(2006, 8, 25)
    # dates between the starting and ending of cosimulation dates
    dts_cosim = [dt_cosim_start + timedelta(days=x) 
                 for x in range(0, (dt_cosim_end - dt_cosim_start).days)]
    # d dictionary of  Summer peak hours
    dr_dict = {"2A":[17,20],"2B":[17,20],"3A":[19,22],"3B":[18,21],
        "3C":[19,22],"4A":[12,15],"4B":[17,20],"4C":[17,20],"5A":[20,23],
        "5B":[17,20],"6A":[16,19],"6B":[17,20],"7A":[16,19]}

    # dts_dr_start = [datetime.combine(x, time(dr_dict[cz][0])) for x in dts_cosim]
    # dts_dr_end = [datetime.combine(x, time(dr_dict[cz][1])) for x in dts_cosim]
   
    # mock up test to always select DR periods between Noon and 4pm
    dts_dr_start = [datetime.combine(x, time(12)) for x in dts_cosim]
    dts_dr_end = [datetime.combine(x, time(16)) for x in dts_cosim]


    # list of hours of DR-event-days within 0-8759
    hrs_cosim_start = [int((x - dt_jan1).total_seconds() / 3600) - 1 for x in dts_cosim]
    hrs_cosim_end =  [(24 + x) for x in hrs_cosim_start]
    hrs_dr_start = [int((x - dt_jan1).total_seconds() / 3600) - 1 for x in dts_dr_start]
    hrs_dr_end = [int((x - dt_jan1).total_seconds() / 3600) - 1 for x in dts_dr_end]
    hrs_rebound = 2

    # print('dts_dr_start {!s} \ndts_dr_end {!s} \nhrs_cosim_start {!s} \
    #       \nhrs_cosim_end {!s} \nhrs_dr_start {!s} \nhrs_dr_end {!s}'.
    #       format(dts_dr_start,dts_dr_end,hrs_cosim_start,hrs_cosim_end,
    #              hrs_dr_start,hrs_dr_end))

    # default schedules from the E+ file for MediumOfficeDetailed Vintage 2004

    sch_base = np.array([
        # E+ input to 'OfficeMedium MinOA_MotorizedDamper_Sched'
        (0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0),
        # E+ input to 'OfficeMedium HTGSETP_SCH_PACU_VAV_bot'
        (15.6,15.6,15.6,15.6,15.6,21,21,21,21,21,21,21,21,21,21,21,21,15.6,15.6,15.6,15.6,15.6,15.6,15.6),
        # E+ input to 'OfficeMedium CLGSETP_SCH_NO_SETBACK'
        (24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24),
        # E+ input to 'OfficeMedium BLDG_LIGHT_SCH_2004'
        (0.05,0.05,0.05,0.05,0.1,0.3,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9, 0.5, 0.3,0.3,0.2,0.2,0.1,0.05,0.05),
        # E+ input to 'OfficeMedium BLDG_EQUIP_SCH_2004'
        (0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.9,0.9,0.9,0.9,0.8,0.9,0.9,0.9,0.9,0.5,0.4,0.4,0.4,0.4,0.4,0.4,0.4)])

    # EnergyPlus I/O variables via FMU
    # OUTPUT VARIABLES
    # oat - 'Site Outdoor Air Drybulb Temperature'
    # orh - 'Site Outdoor Air Relative Humidity'
    # osk - 'Site Daylighting Model Sky Clearness'
    # pwr - 'Facility Total Electric Demand Power'
    # zat - 'aggregate of 'Zone Air Temperature' 35 occupied zones
    # zati - 'Zone Air Temperature' of Top_1 zone
    # zcot - Zone Air CO2 Concentration of Top_1 zone
    # zpp - aggregate of 'People Occupant Count' 35 occupied zones
    # zrh - aggregate of 'Zone Air Relative Humidity' 35 occupied zones
    # zpmv - 'Zone Thermal Comfort Fanger Model PMV' of Top_1 zone
    # zlgt - 'Lights Electric Energy'
    # zlux - aggregate of 'Daylighting Reference Point 1 Illuminance' 22 zones
    # zplg - 'Electric Equipment Electric Energy'

    # INPUT VARIABLES
    # inclg - input to 'OfficeMedium CLGSETP_SCH_NO_SETBACK'
    # inplg - input to 'OfficeMedium BLDG_EQUIP_SCH_2004'
    # inlgt - input to 'OfficeMedium BLDG_LIGHT_SCH_2004'
    # inven - input to 'OfficeMedium MinOA_MotorizedDamper_Sched'
    # inclg_delt - difference of 'inclg' between t and t-1
    # inplg_delt - difference of 'inplg' between t and t-1
    # inlgt_delt - difference of 'inlgt' between t and t-1
    # inven_delt - difference of 'inven' between t and t-1
    # hdrst -
    # hdren -
    io_dict = {'oat':0,'orh':1,'osk':2,'pwr':3,'zat':4,'zati':5,'zcot':6,
               'zpp':7,'zrh':8,'zpmv':9,'zlgt':10,'zlux':11,
               'zplg':12,'inclg':13,'inplg':14,'inlgt':15,'inven':16,
               'inclg_delt':17,'inplg_delt':18,'inlgt_delt':19,'inven_delt':20,
               'hdrst':21,'hdren':22}
                # 'zclg1':17,'zclg2':18,'zhtgsp':19,'zclgsp':20,'olux':21

    # Simulate the baseline for environment data and results
    simBaseline(cz, baseline_csv, fmu_path, sf, sch_base, io_dict)
    # simulated number of days
    sim_days=365
    # starting simulated time
    tStart = 0
    # ending simulated time
    tStop = 3600*1*24*sim_days  # change the timestep in EPlus to 1
    # timestep
    hStep = 3600 # 60 mins

    # numpy array representationof steps
    t = np.arange(tStart, tStop, hStep)
    n_steps = len(t)
    # load and initialize fmu file
    model = load_fmu(fmu_path, log_level=7)
    # initialize cosimulation routine
    model.initialize(tStart, tStop)

    io_vals = np.empty(shape=(len(io_dict),n_steps))

    i = 0
    dt_cosim_i = 0

    while True:
        hour = int((t[i]/3600)%24)
        day_i = int((i-hrs_cosim_start[0])/24) + 1
        ###############################################################

        if i >= hrs_cosim_start[dt_cosim_i] and i <= hrs_cosim_end[dt_cosim_i]:

            if i == hrs_cosim_start[dt_cosim_i]:
                ## rank strategies and getting recommended strategy at hour 0 day-ahead
                print("RECOMMENDED STRATEGY... [", end="", flush=True)

                updateTestPredictCSV(handyfilesvars,baseline_csv, hrs_dr_start[dt_cosim_i], hrs_dr_end[dt_cosim_i], hrs_rebound)

                #rank_strategies(handyfilesvars, bldg_type_vint, sf)

                # recommendations = ('data','recommendations.json')
                # with open(path.join(base_dir, *recommendations), 'r') as pred:
                #     predictions = json.load(pred)["predictions"]

                recommendations = gen_recs(handyfilesvars, sf)
                
                max_value = max(predictions.values())
                for key, value in predictions.items():
                    if (value == max_value):
                        strategy_name = key
                print(strategy_name + "] is selected, and schedule storing is... ", end="", flush=True)
                all_strategy_data = ModelDataLoad(
                        handyfilesvars, opts.mod_init, opts.mod_assess,
                        opts.mod_est, update=None, ndays_update=None)

                strategy_dat = all_strategy_data.dmd_tmp[np.where(all_strategy_data.dmd_tmp['Name'] == strategy_name)]

                mag_pc = strategy_dat['pc_tmp_inc'][0]
                dur_pc = strategy_dat['pc_length'][0]
                delt_clg = strategy_dat['tsp_delt'][0] #np.append(dat['tsp_delt'], np.zeros(14))
                pct_lgt = strategy_dat['lt_pwr_delt_pct'][0] #np.append(dat['lt_pwr_delt_pct'], np.zeros(14))
                pct_plg = strategy_dat['mels_delt_pct'][0] #np.append(dat['mels_delt_pct'], np.zeros(14))
                pct_ven = strategy_dat['ven_delt_pct'][0]


            if i > (hrs_dr_start[dt_cosim_i] - dur_pc) and i <= (hrs_dr_end[dt_cosim_i]):

                if i <= hrs_dr_start[dt_cosim_i]: 
                    # useful for pre-cooling
                    io_vals[io_dict['inclg']][i] = sch_base[2][hour] - mag_pc
                else:
                    io_vals[io_dict['inven']][i] = sch_base[0][hour] * (1 - pct_ven)
                    io_vals[io_dict['inclg']][i] = sch_base[2][hour] + delt_clg
                    io_vals[io_dict['inlgt']][i] = sch_base[3][hour] * (1 - pct_lgt)
                    io_vals[io_dict['inplg']][i] = sch_base[4][hour] * (1 - pct_plg)
                    io_vals[io_dict['hdrst']][i] = i - io_vals[io_dict['hdrst']][dt_cosim_i]
                    io_vals[io_dict['hdren']][i] = 0

                    print('delt_clg {!s} dur_pc {!s} mag_pc {!s} pct_lgt {!s} pct_plg {!s} pct_ven {!s}'.
                          format(delt_clg, dur_pc, mag_pc, pct_lgt, pct_plg, pct_ven))

            else:
                io_vals[io_dict['inven']][i] = sch_base[0][hour]
                io_vals[io_dict['inclg']][i] = sch_base[2][hour]
                io_vals[io_dict['inlgt']][i] = sch_base[3][hour]
                io_vals[io_dict['inplg']][i] = sch_base[4][hour]
                io_vals[io_dict['hdrst']][i] = 0
                if i > (hrs_dr_end[dt_cosim_i]) and i <= (hrs_dr_end[dt_cosim_i] + 2):
                    io_vals[io_dict['hdren']][i] = i - hrs_dr_end[dt_cosim_i]
                    if i == (hrs_dr_end[dt_cosim_i] + 2):
                        #cosim_updatecsv(dt_cosim_i)
                        
                        updateTestUpdateCSV(cz, baseline_csv, update_csv, day_i, dt_cosim_i, io_dict, io_vals, dts_dr_start, dts_dr_end, hrs_dr_start, hrs_dr_end, hrs_rebound)
                        dt_cosim_i += 1
                    if dt_cosim_i >= len(hrs_dr_start):
                        dt_cosim_i -= 1
                else:
                    io_vals[io_dict['hdren']][i] = 0

        else:
            io_vals[io_dict['inven']][i] = sch_base[0][hour]
            io_vals[io_dict['inclg']][i] = sch_base[2][hour]
            io_vals[io_dict['inlgt']][i] = sch_base[3][hour]
            io_vals[io_dict['inplg']][i] = sch_base[4][hour]

        if i > 0:
            io_vals[io_dict['inclg_delt']][i] =\
                io_vals[io_dict['inclg']][i] - io_vals[io_dict['inclg']][i-1]
            io_vals[io_dict['inlgt_delt']][i] =\
                io_vals[io_dict['inlgt']][i] - io_vals[io_dict['inlgt']][i-1]
            io_vals[io_dict['inplg_delt']][i] =\
                io_vals[io_dict['inplg']][i] - io_vals[io_dict['inplg']][i-1]
            io_vals[io_dict['inven_delt']][i] =\
                io_vals[io_dict['inven']][i] - io_vals[io_dict['inven']][i-1]
        ###############################################################
        model.set(['InMELsSch', 'InLightSch', 'InCoolingSch', 'InVentSch'],
                  [io_vals[io_dict['inplg']][i],io_vals[io_dict['inlgt']][i],
                   io_vals[io_dict['inclg']][i],io_vals[io_dict['inven']][i]])

        model.do_step(current_t = t[i], step_size=hStep, new_step=True)

        # Get the outputs of the simulation
        temp_np = np.array([])
        ppl_np = np.array([])
        rh_np = np.array([])
        for zoneid in range(0,34):
            temp_np = np.append(temp_np, (model.get('ZAT_' + str(zoneid))))
            ppl_np = np.append(ppl_np, (model.get('PEOPLE_' + str(zoneid))))
            rh_np = np.append(rh_np, (model.get('ZRH_' + str(zoneid))))


        # temp_wght = np.sum((temp_np * ppl_np)) / np.sum(ppl_np)
        # ppl_wght = (np.sum(ppl_np * ppl_np) / np.sum(ppl_np)) / np.sum(ppl_np)
        rh_wght = np.sum((rh_np * ppl_np)) / np.sum(ppl_np)

        temp_wght = np.mean(temp_np)
        ppl_wght = np.mean(ppl_np)

        io_vals[io_dict['zat']][i] = (temp_wght * 9 / 5) + 32   #farenheit
        io_vals[io_dict['zati']][i] = (model.get('ZAT_31'))     # * 9 / 5) + 32
        io_vals[io_dict['zpp']][i] = ppl_wght
        io_vals[io_dict['zrh']][i] = rh_wght

        io_vals[io_dict['zcot']][i] = model.get('ZoneCOTwo')
        io_vals[io_dict['zpmv']][i] = model.get('ZonePMV')
        io_vals[io_dict['zlgt']][i] = model.get('LightsEnergy') / 3600000 # kwh
        io_vals[io_dict['zplg']][i] = model.get('MelsEnergy') / 3600000
        io_vals[io_dict['pwr']][i] = model.get('BldgPwr') / (1000 * sf)

        io_vals[io_dict['osk']][i] = model.get('OutSkyClear')
        io_vals[io_dict['oat']][i] = (model.get('OutDrybulb') * 9 / 5) + 32 # F
        io_vals[io_dict['orh']][i] = model.get('OutRH')

        i += 1
        if (i == n_steps):
            break


def run_mod_prediction(handyfilesvars, trace, mod, dat, n_samples, inds):

    # Initialize variable inputs and outputs for the given model type
    iop_all = ModelIO(handyfilesvars, opts.mod_init, opts.mod_est,
                      opts.mod_assess, mod, dat)
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


def run_mod_assessment(handyfilesvars, trace, mod, iog, refs):

    # Plot parameter traces
    az.plot_trace(trace)
    fig1_path = path.join(
        "diagnostic_plots", handyfilesvars.mod_dict[mod]["fig_names"][0])
    plt.gcf().savefig(fig1_path)
    # Plot parameter posterior distributions
    az.plot_posterior(
        trace, var_names=[handyfilesvars.mod_dict[mod]["var_names"][0]],
        ref_val=refs)
    fig2_path = path.join(
        "diagnostic_plots", handyfilesvars.mod_dict[mod]["fig_names"][1])
    plt.gcf().savefig(fig2_path)

    # Set testing data
    iot = ModelIOTest(iog, opts.mod_init, opts.mod_assess)
    # Re-initialize model with subset of data used for testing (
    # for model initialization case only)
    if opts.mod_assess is True:
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
                mu=est, sd=sd, observed=np.zeros(iot.X.shape[0]))
            output_diagnostics(handyfilesvars, trace, iot, mod)
    else:
        output_diagnostics(handyfilesvars, trace, iot, mod)


def output_diagnostics(handyfilesvars, trace, iot, mod):

    # Posterior predictive
    ppc_var = pm.sample_posterior_predictive(trace, samples=500)
    obs_data = iot.Y
    pred_data = ppc_var[handyfilesvars.mod_dict[mod][
        "var_names"][2]]

    # Histogram
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.hist(
        [n.mean() for n in pred_data], bins=20, alpha=0.5)
    ax1.axvline(obs_data.mean())
    ax1.set(title='Posterior Predictive of the Mean',
            xlabel=("Mean (" + mod + ")"), ylabel='Frequency')
    fig1_path = path.join(
        "diagnostic_plots", handyfilesvars.mod_dict[mod]["fig_names"][2])
    fig1.savefig(fig1_path)

    # Scatter/line
    b, m = polyfit(pred_data.mean(axis=0), obs_data, 1)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.scatter(
        x=pred_data.mean(axis=0), y=obs_data, alpha=0.05)
    ax2.plot(pred_data.mean(axis=0),
             b + m * pred_data.mean(axis=0),
             linestyle='-', color="mediumseagreen")
    ax2.set_aspect('equal', 'box')
    ax2.set(title='Observed vs. Predicted',
            xlabel=("Mean Predicted (" + mod + ")"),
            ylabel=("Observed"))
    plt.rc('grid', linestyle="-", color='silver')
    plt.grid(True)
    fig2_path = path.join(
        "diagnostic_plots", handyfilesvars.mod_dict[mod]["fig_names"][3])
    fig2.savefig(fig2_path)


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


def plot_updating(handyfilesvars, param, traces, mod):

    # Set color map for plots
    cmap = mpl.cm.autumn
    # Initialize subplots and figure object
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 4))
    # Develop and plot kernel density estimators of parameter traces,
    # and plot these estimators across each successive parameter update
    for update_i, trace in enumerate(traces):
        # Outdoor temperature
        samples_oat = np.array([x[1] for x in trace[param]])
        smin_oat, smax_oat = np.min(samples_oat), np.max(samples_oat)
        x_oat = np.linspace(smin_oat, smax_oat, 100)
        y_oat = stats.gaussian_kde(samples_oat)(x_oat)
        axs[0].plot(x_oat, y_oat, color=cmap(1 - update_i / len(traces)))
        # Set point adjustment level
        samples_sp = np.array([x[4] for x in trace[param]])
        smin_sp, smax_sp = np.min(samples_sp), np.max(samples_sp)
        x_sp = np.linspace(smin_sp, smax_sp, 100)
        y_sp = stats.gaussian_kde(samples_sp)(x_sp)
        axs[1].plot(x_sp, y_sp, color=cmap(1 - update_i / len(traces)))
    # Set OAT plot title and axis labels
    axs[0].set_title("Outdoor Temperature")
    axs[0].set_ylabel('Frequency')
    # Set set point temperature plot title and axis labels
    axs[1].set_title("Set Point Offset")
    axs[1].set_ylabel('Frequency')
    # Set figure layout parameter
    fig.tight_layout(pad=1.0)
    # Determine figure path and save figure
    fig_path = path.join(
        "diagnostic_plots", handyfilesvars.mod_dict[mod]["fig_names"][4])
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
    parser.add_argument("--mod_cosimulate", action="store_true",
                        help="Cosimulate with EnergyPl")
    # Required flags for building type and size
    parser.add_argument("--bldg_type", required=True, type=str,
                        choices=["mediumofficenew", "mediumofficeold",
                                 "retailnew", "retailold"],
                        help="Building type/vintage")
    parser.add_argument("--bldg_sf", required=True, type=int,
                        help="Building square footage")
    # Object to store all user-specified execution arguments
    opts = parser.parse_args()
    base_dir = getcwd()
    main(base_dir)
