#https://stackoverflow.com/questions/20625774/reading-a-comma-delimited-file-with-a-date-object-and-a-float-with-python
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date,datetime, time, timedelta
import datetime as dt
import numpy as np
from os import getcwd, path

# fmu_names_dtypes = [
#             ('date','outdoor_drybulb','outdoor_rh','building','lights','plugloads',
#              'cooling','heating','zone_ppl','zone_temp',' zone_pmv','cotwo',
#              'zone_htgsp','zone_clgsp','in_htg','in_clg'),
#             (['datetime64'] * 1 + ['<f8'] * 15)]
# baseline_dat = genfromtxt('FMU_result_baseline.csv', 
#     skip_header=True, delimiter=',',
#     names=fmu_names_dtypes[0],
#     dtype=fmu_names_dtypes[1])

#baseline_dat = pd.read_csv("cosim_outputs/baseline_MO_3A.csv", parse_dates=True, index_col='datetime')
# dat = pd.read_csv("cosim_outputs/results_MO_3A.csv", parse_dates=True)


# dat['datetime'] =  pd.to_datetime(dat['datetime'],format='%m/%d/%Y %H:%M').apply(lambda x: x.hour)

# print(dat.dtypes)

# dat['cooling1_diff'] = dat['cooling1_diff'] / 1000 * -1
# dat['building_diff'] = dat['building_diff'] / 1000 * -1

# dat = dat[1:22]

# dat.plot(y='cooling1_diff', x='datetime', kind='bar', title='Cooling Energy Saving in KWh')
# dat.plot(y='building_diff', x='datetime', kind='bar', title='Building Energy Saving in KWh')
# plt.show()

# print(dr["2A"]["start"])

#print(baseline_dat.dtypes)
#########################################################################################
# dr_s_hr = 15
# dr_e_hr = 18
# dt_jan1 = datetime(2006, 1, 1)
# dt_cosim = datetime(2006, 8, 24)
# dt_dr_start = datetime.combine(dt_cosim, time(dr_s_hr))
# dt_dr_end = datetime.combine(dt_cosim, time(dr_e_hr))

#baseline_dat['datetime'] = pd.to_datetime(baseline_dat['datetime'])
#print(baseline_dat[(baseline_dat['datetime']>=dt_dr_start) & (baseline_dat['datetime']<(dt_dr_end + timedelta(hours=2)))])
#################################################################################################
# dr_s_hr = 15
# dr_e_hr = 18
# dt_jan1 = datetime(2006, 1, 1)
# dt_cosim = datetime(2006, 8, 24)
# dt_dr_start = datetime.combine(dt_cosim, time(dr_s_hr))
# dt_dr_end = datetime.combine(dt_cosim, time(dr_e_hr))


# hours_cosim = int((dt_cosim - dt_jan1).total_seconds() / 3600)
# hours_dr_start = int((dt_dr_start - dt_jan1).total_seconds() / 3600)
# hours_dr_end = int((dt_dr_end - dt_jan1).total_seconds() / 3600)

# print(dt_cosim)
# print(dt_dr_start)
# print(dt_dr_start + timedelta(hours=5))
# print(hours_cosim)
# print(hours_dr_start)
# print(hours_dr_end)

# print()

############################################################################
# dt_jan1 = datetime(2006, 1, 1)
# dt_cosim_start = datetime(2006, 8, 21)
# dt_cosim_end = datetime(2006, 8, 25)
# dts_cosim = [dt_cosim_start + timedelta(days=x) for x in range(0, (dt_cosim_end-dt_cosim_start).days)]

# dts_dr_start = [datetime.combine(x, time(14)) for x in dts_cosim]
# dts_dr_end = [datetime.combine(x, time(18)) for x in dts_cosim]

# hrs_cosim_start = [int((x - dt_jan1).total_seconds() / 3600) for x in dts_cosim]
# hrs_cosim_end =  [(24 + x) for x in hrs_cosim_start]

# hrs_dr_start = [int((x - dt_jan1).total_seconds() / 3600) for x in dts_dr_start] # int((dt_dr_start - dt_jan1).total_seconds() / 3600)
# hrs_dr_end = [int((x - dt_jan1).total_seconds() / 3600) for x in dts_dr_end]

# hrs_cosim_start_24 = [x % 24 for x in hrs_cosim_start]
# hrs_cosim_end_24 = [x % 24 for x in hrs_cosim_end]
# hrs_dr_start_24 = [x % 24 for x in hrs_dr_start]
# hrs_dr_end_24 = [x % 24 for x in hrs_dr_end]

# sim_days=365
# tStart = 0
# tStop = 3600*1*24*sim_days   ## change the timestep in EPlus to 1
# hStep = 3600 # 60 mins

# out_dt_dr_start = (dts_dr_start[0] + timedelta(hours=1))
# out_dt_dr_end = (dts_dr_end[0] + timedelta(hours=2))
# hrtime = pd.date_range(start=out_dt_dr_start, end=out_dt_dr_end, freq='H').to_pydatetime()


# t = np.arange(tStart, tStop, hStep)
# n_steps = len(t)

# print(out_dt_dr_start)
# print(out_dt_dr_end)
# print([x.hour for x in hrtime])

# print(dt_jan1)
# print(dt_cosim_start)
# print(dt_cosim_end)
# print(dts_cosim)
# print()
# print(dts_dr_start)
# print(dts_dr_end)

#print([x.hour for x in dts_dr_start])
# print()

# print(hrs_cosim_start)
# print(hrs_cosim_end)
# print()

# print(hrs_dr_start)
# print(hrs_dr_end)
# print()
# print(hrtime)
# print(range((hrs_dr_start[0]+1),(hrs_dr_end[0]+3)))
# print(range(hrs_dr_start[0],hrs_dr_end[0]))
# print(hrs_dr_start)
# print(len(hrs_dr_start))

# hrs_dr = []
# for x in range(0,len(hrs_dr_start)):
#     for item in list(range(hrs_dr_start[x],hrs_dr_end[x])):
#         hrs_dr.append(item)

# print(hrs_dr)


# print()
# print(hrs_cosim_start_24)
# print(hrs_cosim_end_24)
# print(hrs_dr_start_24)
# print(hrs_dr_end_24)
# print()
# print (t)

# i = 0
# dt_cosim_i = 0
# while i < n_steps:
#     hour = int((t[i]/3600)%24)
#     next_hour = (hour+1)%24
#     i += 1
#############################################################################
# data = pd.DataFrame(columns = ['name', 'physics','chemistry','algebra'])
# new_row = {'name':'Geo', 'physics':87, 'chemistry':92, 'algebra':97}
# data = data.append(new_row, ignore_index=True)

# print(data)
#############################################################################

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
                      'dmd_delt_sf', 't_in_delt','rh_in_delt', 't_out',
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
                    'Name', 'Hr', 't_out', 'rh_out', 'lt_nat', 'base_lt_frac',
                    'occ_frac', 'delt_price_kwh',
                    'hrs_since_dr_st','hrs_since_dr_end', 
                    'hrs_since_pc_st','hrs_since_pc_end',
                    'tsp_delt', 'lt_pwr_delt_pct','ven_delt_pct', 'mels_delt_pct', 
                    'tsp_delt_lag','lt_pwr_delt_pct_lag', 'ven_delt_pct_lag','mels_delt_pct_lag', 
                    'pc_tmp_inc', 'pc_length','lt_pwr_delt'),
                    (['<U25'] + ['<f8'] * 22)] for n in range(4))
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
            else:
                self.oaf_delt, self.plug_delt, self.price_delt = (
                    None for n in range(3))

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


# bldg_type_vint = 'mediumofficenew'
# sf = 400
# mod_init = False
# mod_est = False
# mod_assess = False

# handyfilesvars = UsefulFilesVars(bldg_type_vint, mod_init, mod_est, mod_assess)
# path_to_predictcsv = path.join(*handyfilesvars.mod_dict["temperature"]["io_data"][0])

# base_dir = getcwd()

######################
# predict_df = pd.read_csv(path_to_predictcsv)
# predict_df.reset_index(drop=True, inplace=True)

# print(predict_df)
# print(predict_df['Hr'].mean() == 1)

# predict_df_new = predict_df[predict_df['Hr'] == 1]
# predict_df_new['Hr'] = 3
# print(predict_df_new)
# print(predict_df)
# predict_df.to_csv(predict_csv, index=False)
#####################

# predict_np = ModelDataLoad(handyfilesvars, mod_init, mod_assess,
# 	mod_est, update=None, ndays_update=None)
# with open(path_to_predictcsv, encoding='utf-8-sig') as f:
#     predict_np_header = f.readline()
# print(handyfilesvars.mod_dict.keys())
# iog = ModelIO(handyfilesvars, mod_init, mod_est, mod_assess, 'demand', predict_np)

# #predict_np_new = np.copy(predict_np.dmd_tmp[0:15,])
# import copy
# predict_np_new = copy.deepcopy(predict_np)

# predict_np_new.dmd_tmp['Hr'] = 3

#print(predict_np.dmd_tmp)
#print(predict_np_new['Hr'])

# print(predict_np.dmd_tmp['Name'])
# print(predict_np.dmd_tmp['Hr'])
# print(predict_np.dmd_tmp['t_out'])
# print(predict_np.dmd_tmp['rh_out'])
# print(predict_np.dmd_tmp['lt_nat'])
# print(predict_np.dmd_tmp['base_lt_frac'])
# print(predict_np.dmd_tmp['occ_frac'])
# print(predict_np_new['delt_price_kwh'])
# print(predict_np.dmd_tmp['Hr'])
# print(predict_np_new.dmd_tmp['Hr'])


# print(predict_np.dmd_tmp)
# print(predict_np.dmd_tmp)
# print(predict_np.hr)
# print(predict_np_header)
# print(path_to_predictcsv)


# with open(path_to_predictcsv,'ab') as f:
#     np.savetxt(f, predict_np.dmd_tmp, fmt='%s', delimiter=',',comments='',encoding=None)

# baseline_csv = 'cosim_outputs/baseline_MO_' + '3A' + '.csv'
# baseline_df = pd.read_csv(baseline_csv, parse_dates=True)
# print(baseline_df[8750:8751])


###############################################################################
from numpy.lib import recfunctions as rfn

sch_vals = np.array([(0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0), # ventilation
            (15.6,15.6,15.6,15.6,15.6,21,21,21,21,21,21,21,21,21,21,21,21,15.6,15.6,15.6,15.6,15.6,15.6,15.6), # heating
            (24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24), # cooling
            (0.05,0.05,0.05,0.05,0.1,0.3,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9, 0.5, 0.3,0.3,0.2,0.2,0.1,0.05,0.05), #lighting
            (0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.9,0.9,0.9,0.9,0.8,0.9,0.9,0.9,0.9,0.5,0.4,0.4,0.4,0.4,0.4,0.4,0.4)])

#print(sch_vals)
#print(sch_vals[1][5])


sch_ven = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
sch_htg = [15.6,15.6,15.6,15.6,15.6,21,21,21,21,21,21,21,21,21,21,21,21,15.6,15.6,15.6,15.6,15.6,15.6,15.6]
sch_clg = [24 for i in range(24)]
sch_lgt = [0.05,0.05,0.05,0.05,0.1,0.3,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9, 0.5, 0.3,0.3,0.2,0.2,0.1,0.05,0.05]
sch_plg = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.9,0.9,0.9,0.9,0.8,0.9,0.9,0.9,0.9,0.5,0.4,0.4,0.4,0.4,0.4,0.4,0.4]

sch_vals2 = np.vstack([sch_ven, sch_htg, sch_clg, sch_lgt, sch_plg])
r = np.s_[5:8]
print(sch_vals2[1][r])

empty_np = np.empty(shape=(2,80))
io_dict = {'oat':0,'orh':1,'osk':2,'pwr':3,'zat':4,'zati':5,'zcot':6,
               'zpp':7,'zrh':8,'zpmv':9,'zhtgsp':10,'zclgsp':11,'zlgt':12,
               'zplg':13,'inclg':15,'inplg':16,'inlgt':17,'inven':18}
# print(len(io_dict))
# print(empty_np)
# print(empty_np[0][11:15])