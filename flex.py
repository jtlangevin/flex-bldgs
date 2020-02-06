#!/usr/bin/env python3
import pymc3 as pm
import theano as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.special import softmax
from os import getcwd, path
from argparse import ArgumentParser
import pickle
import arviz as az
from matplotlib import pyplot as plt
tt.config.compute_value = "ignore"


class UsefulFilesVars(object):
    """Summarize key model input, estimation, and output information.

    Attributes:
        mod_dict (dict): Dict with information on input data, variables, and
            output plotting file names for each model type.
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
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (None for n in range(2))
            else:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (
                    ("data", "test.csv") for n in range(3))
            # Set stored model data files
            stored_tmp = ("model_stored", "tmp_mo_n.pkl")
            stored_dmd = ("model_stored", "dmd_mo_n.pkl")
            stored_co2 = ("model_stored", "co2_mo.pkl")
            stored_pc_tmp = ("model_stored", "pc_mo_n.pkl")
        # Medium office, <2004 vintage
        elif bldg_type_vint == "mediumofficeold":
            if mod_init is True or mod_assess is True:
                dmd_tmp_dat = ("data", "MO_DR_old.csv")
                co2_dat = ("data", "CO2_MO.csv")
                pc_tmp_dat = ("data", "MO_Precooling_old.csv")
            elif mod_est is True:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (None for n in range(2))
            else:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (
                    ("data", "test.csv") for n in range(3))
            stored_tmp = ("model_stored", "tmp_mo_o.pkl")
            stored_dmd = ("model_stored", "dmd_mo_o.pkl")
            stored_co2 = ("model_stored", "co2_mo.pkl")
            stored_pc_tmp = ("model_stored", "pc_mo_o.pkl")
        # Retail, >=2004 vintage
        elif bldg_type_vint == "retailnew":
            if mod_init is True or mod_assess is True:
                dmd_tmp_dat = ("data", "Retail_DR_new.csv")
                co2_dat = ("data", "CO2_Retail.csv")
                pc_tmp_dat = ("data", "Retail_Precooling_new.csv")
            elif mod_est is True:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (None for n in range(3))
            else:
                dmd_tmp_dat, co2_datm, pc_tmp_dat = (
                    ("data", "test.csv") for n in range(2))
            stored_tmp = ("model_stored", "tmp_ret_n.pkl")
            stored_dmd = ("model_stored", "dmd_ret_n.pkl")
            stored_co2 = ("model_stored", "co2_ret.pkl")
            stored_pc_tmp = ("model_stored", "pc_ret_n.pkl")
        # Medium office, <2004 vintage
        elif bldg_type_vint == "retailold":
            if mod_init is True or mod_assess is True:
                dmd_tmp_dat = ("data", "Retail_DR_old.csv")
                co2_dat = ("data", "CO2_Retail.csv")
                pc_tmp_dat = ("data", "Retail_Precooling_old.csv")
            elif mod_est is True:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (None for n in range(2))
            else:
                dmd_tmp_dat, co2_dat, pc_tmp_dat = (
                    ("data", "test.csv") for n in range(2))
            stored_tmp = ("model_stored", "tmp_ret_o.pkl")
            stored_dmd = ("model_stored", "dmd_ret_o.pkl")
            stored_co2 = ("model_stored", "co2_ret.pkl")
            stored_pc_tmp = ("model_stored", "pc_ret_o.pkl")
        # Lighting models are not broken out by building type/vintage
        if mod_init is True or mod_assess is True:
            lgt_dat = ("data", "Illuminance.csv")
        elif mod_est is True:
            lgt_dat = None
        else:
            lgt_dat = ("data", "test.csv")
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
                 'cloud_out', 'hrs_since_dr_st', 'lt_pwr_delt',
                 'lt_pwr_delt_pct_lag'), (['<i4'] * 1 + ['<f8'] * 8)]
            pc_tmp_names_dtypes = [
                ('id', 'vintage', 'day_typ', 'hour', 'climate',
                 'dmd_delt_sf', 't_in_delt', 'rh_in_delt', 't_out', 'rh_out',
                 'cloud_out', 'occ_frac', 'tsp_delt', 'lt_pwr_delt_pct',
                 'ven_delt_pct', 'mels_delt_pct', 'hrs_since_dr_st',
                 'hrs_since_dr_end', 'hrs_since_pc_end', 'hrs_since_pc_st',
                 'tsp_delt_lag', 'lt_pwr_delt_pct_lag', 'mels_delt_pct_lag',
                 'ven_delt_pct_lag'),
                (['<i4'] * 4 + ['<U25'] + ['<f8'] * 19)]
        # Set data input file column names and data types for model
        # re-estimation; these will be the same across models
        elif mod_est is True:
            tmp_dmd_names_dtypes, co2_names_dtypes, lt_names_dtypes, \
                pc_tmp_names_dtypes = (None for n in range(3))
        # Set data input file column names and data types for model
        # prediction; these will be the same across models
        else:
            tmp_dmd_names_dtypes, co2_names_dtypes, lt_names_dtypes, \
                pc_tmp_names_dtypes = ([(
                    'Name', 'Scn', 't_out', 'rh_out', 'lt_nat',
                    'occ_frac', 'delt_price_kwh', 'hrs_since_dr_st',
                    'hrs_since_dr_end', 'hrs_since_pc_st',
                    'hrs_since_pc_end', 'tsp_delt', 'lt_pwr_delt_pct',
                    'ven_delt_pct', 'mels_delt_pct', 'tsp_delt_lag',
                    'lt_pwr_delt_pct_lag', 'ven_delt_pct_lag',
                    'mels_delt_pct_lag', 'pc_tmp_inc', 'pc_length',
                    'lt_pwr_delt'),
                    (['<U25'] + ['<f8'] * 21)] for n in range(3))

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
                    "ppcheck_tmp.png", "scatter_tmp.png"]
            },
            "demand": {
                "io_data": [dmd_tmp_dat, stored_dmd],
                "io_data_names": tmp_dmd_names_dtypes,
                "var_names": ['dmd_params', 'dmd_sd', 'dmd'],
                "fig_names": [
                    "traceplots_dmd.png", "postplots_dmd.png",
                    "ppcheck_dmd.png", "scatter_dmd.png"]
            },
            "co2": {
                "io_data": [co2_dat, stored_co2],
                "io_data_names": co2_names_dtypes,
                "var_names": ['co2_params', 'co2_sd', 'co2'],
                "fig_names": [
                    "traceplots_co2.png", "postplots_co2.png",
                    "ppcheck_co2.png", "scatter_co2.png"]
            },
            "lighting": {
                "io_data": [lgt_dat, stored_lt],
                "io_data_names": lt_names_dtypes,
                "var_names": ['lt_params', 'lt_sd', 'lt'],
                "fig_names": [
                    "traceplots_lt.png", "postplots_lt.png",
                    "ppcheck_lt.png", "scatter_lt.png"]
            },
            "temperature (pre-cool)": {
                "io_data": [pc_tmp_dat, stored_pc_tmp],
                "io_data_names": pc_tmp_names_dtypes,
                "var_names": ['ta_pc_params', 'ta_pc_sd', 'ta_pc'],
                "fig_names": [
                    "traceplots_tmp_pc.png", "postplots_tmp_pc.png",
                    "ppcheck_tmp_pc.png", "scatter_tmp_pc.png"]
            },

        }


class ModelDataLoad(object):
    """Load the data files needed to initialize, estimate, or run models.

    Attributes:
        dmd_tmp (numpy.ndarray): Input/output data for demand/temp. models.
        co2 (numpy.ndarray): Input/output data for CO2 model.
        lt (numpy.ndarray): Input/output data for lighting model.
    """

    def __init__(self, handyfilesvars, mod_init, mod_assess, scn):
        """Initialize class attributes."""

        # Initialize plug load delta and price delta as None
        self.plug_delt, self.price_delt = (None for n in range(2))
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
                            "temperature (pre-cool)"]["io_data"][0]),
                skip_header=True, delimiter=',',
                names=handyfilesvars.mod_dict[
                    "temperature (pre-cool)"]["io_data_names"][0],
                dtype=handyfilesvars.mod_dict[
                    "temperature (pre-cool)"]["io_data_names"][1])
            # Stop routine if files were not properly read in
            if any([len(x) == 0 for x in [
                    self.dmd_tmp, self.co2, self.lt, self.pc_tmp]]):
                raise ValueError("Failure to read input file(s)")
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
            # Restrict prediction input file to appropriate scenario (if
            # applicable)
            if scn is not None:
                common_data = common_data[
                    np.where(common_data['Scn'] == (scn + 1))]
            # Set inputs to demand, temperature, co2, and lighting models
            # from prediction input file
            self.dmd_tmp, self.co2, self.lt, self.pc_tmp = (
                common_data for n in range(3))

            # Set plug load delta and price delta to values from prediction
            # input file (these are not predicted via a Bayesian model)
            self.plug_delt = common_data['mels_delt_pct']
            self.price_delt = common_data['delt_price_kwh']


class ModelIO(object):
    """Initialize input/output variables/data structure for each model

    Attributes:
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
            tmp_lt_interact = tmp_delta * lt_delta
            # Temp. set point and outdoor air fraction difference
            tmp_oaf_interact = tmp_delta * oaf_delta
            # Pre-cool duration, and magnitude
            pcool_interact = pcool_duration * pcool_magnitude
            # Temp. set point difference, outdoor temperature, and DR st. time
            tmp_out_dr_start_interact = tmp_delta * temp_out * dr_start

            # Set model input (X) variables
            self.X_all = np.stack([
                temp_out, rh_out, occ_frac, tmp_delta, lt_delta, oaf_delta,
                plug_delta, dr_start, dr_end, pcool_duration, pcool_magnitude,
                tmp_delta_lag, lt_delta_lag, oaf_delta_lag, plug_delta_lag,
                tmp_lt_interact, tmp_oaf_interact, pcool_interact,
                tmp_out_dr_start_interact, intercept], axis=1)
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
            # Hours since DR event started
            dr_start_co2 = data.co2['hrs_since_dr_st']
            # Hours since DR event ended
            dr_end_co2 = data.co2['hrs_since_dr_st']
            # Hours since pre-cooling started
            pcool_start_co2 = data.co2['hrs_since_pc_st']
            # Temperature set point difference, previous time step
            tmp_delt_lag_co2 = data.co2['tsp_delt_lag']
            # Outdoor set point difference, previous time step
            oaf_delt_lag_co2 = data.co2['ven_delt_pct_lag']
            # Temp. set point and outdoor air fraction difference interaction
            tmp_oaf_interact_co2 = tmp_delt_co2 * oaf_delt_co2
            # Intercept term
            intercept_co2 = intercept = np.ones(len(occ_frac_co2))

            # Initialize variables for CO2 model

            # Set model input (X) variables
            self.X_all = np.stack([
                temp_out_co2, rh_out_co2, occ_frac_co2,
                tmp_delt_co2, oaf_delt_co2, dr_start_co2, dr_end_co2,
                tmp_delt_lag_co2, oaf_delt_lag_co2, tmp_oaf_interact_co2,
                pcool_start_co2, intercept_co2], axis=1)
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
            # Lighting power difference
            lt_delta = data.lt['lt_pwr_delt']
            # Intercept term
            intercept_lt = intercept = np.ones(len(lt_out))

            # Set model input (X) variables
            self.X_all = np.stack([lt_out, lt_delta, intercept_lt], axis=1)
            # Set model output (Y) variable for estimation cases
            if mod_init is True or mod_est is True or mod_assess is True:
                self.Y_all = data.lt['lt_in_delt_pct']

        elif mod == "temperature (pre-cool)":
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
            # Temp. set point difference, outdoor temperature, and DR st. time
            tmp_out_pc_start_interact = tmp_delta * temp_out * pcool_start
            # Set a vector of ones for intercept estimation
            intercept = np.ones(len(occ_frac))       

            # Set model input (X) variables
            self.X_all = np.stack([
                temp_out, rh_out, occ_frac, tmp_delta, pcool_start,
                tmp_delta_lag, tmp_out_pc_start_interact, intercept], axis=1)
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
    if opts.mod_init is True or opts.mod_est is True:

        print("Loading input data...", end="", flush=True)
        # Read-in input data
        dat = ModelDataLoad(handyfilesvars, opts.mod_init, opts.mod_assess,
                            scn=None)
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
                run_mod_assessment(handyfilesvars, trace, mod, iog)
                print("Complete.")

    elif opts.mod_assess is True:

        print("Loading input data...", end="", flush=True)
        # Read-in input data
        dat = ModelDataLoad(handyfilesvars, opts.mod_init, opts.mod_assess,
                            scn=None)
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
            run_mod_assessment(handyfilesvars, trace, mod, iog)
            print("Complete.")
    else:

        # Set number of control choices
        n_choices = 16
        # Set number of samples to draw. for predictions
        n_samples = 1000
        # Set number of scenarios to test
        n_scenarios = 1
        # Initialize posterior predictive data dict
        pp_dict = {
            key: [] for key in handyfilesvars.mod_dict.keys()}

        # Set the constant set of betas across alternatives to use in choice
        # betas_choice = np.array([0.05, -20, -50, -50, -100, 10])

        betas_choice = np.array([0.01, 0, 0, 0, -100, 10])

        # Loop through the set of scenarios considered for FY19 EOY deliverable
        for scn in [1]:
            # Sample noise to use in the choice model
            rand_elem = np.random.normal(
                loc=0, scale=1, size=(n_samples, n_choices))
            print(("Running input scenario " + str(scn+1)) + "...")

            print("Loading input data...")
            # Read-in input data for scenario
            dat = ModelDataLoad(
                handyfilesvars, opts.mod_init, opts.mod_assess, scn)

            for mod in handyfilesvars.mod_dict.keys():
                # Reload trace
                with open(path.join(base_dir, *handyfilesvars.mod_dict[mod][
                        "io_data"][1]), 'rb') as store:
                    trace = pickle.load(store)['trace']
                pp_dict[mod] = run_mod_prediction(
                    handyfilesvars, trace, mod, dat, n_samples)
            # print("Complete.")

            print("Making predictions...")
            # Multiply change in demand/sf by sf and price delta to get total
            # cost difference for the operator
            cost_delt = pp_dict["demand"]['dmd'] * sf * dat.price_delt
            # Extend plug load delta values for each choice across all samples
            plug_delt = np.tile(dat.plug_delt, (n_samples, 1))
            # Extend intercept input for each choice across all samples
            intercept = np.tile(np.ones(n_choices), (n_samples, 1))
            # Stack all model inputs into a single array
            x_choice = np.stack([
                cost_delt, pp_dict["temperature"]["ta"],
                pp_dict["co2"]["co2"], pp_dict["lighting"]["lt"],
                plug_delt, intercept])
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
            unique, counts = np.unique(choice_out, return_counts=True)
            print(
                "Choice number and probability of selecting "
                "(choices not listed are zero): " + str(dict(zip(
                    (unique + 1), (counts / np.sum(counts))))))


def run_mod_prediction(handyfilesvars, trace, mod, dat, n_samples):
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


def run_mod_assessment(handyfilesvars, trace, mod, iog):

    # Plot parameter traces
    az.plot_trace(trace)
    fig1_path = path.join(
        "diagnostic_plots", handyfilesvars.mod_dict[mod]["fig_names"][0])
    plt.gcf().savefig(fig1_path)
    # Plot parameter posterior distributions
    az.plot_posterior(trace)
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
