.. Substitutions
.. |--| unicode:: U+2013   .. en dash
.. |---| unicode:: U+2014  .. em dash, trimming surrounding whitespace
   :trim:

.. _execute:

Execution Guide
================

FlexAssist supports multiple execution modes:

* Selecting from a set of candidate DR response strategies.
* Generating predictions of baseline demand (without implementation of DR strategies).
* Generating new models of building demand/services under DR and baseline conditions.
* Updating existing models of building demand/services given new data.

Guidance on preparing inputs, running the model, and interpreting output for each of these execution modes is provided below.

.. _mod-pred:

Selecting from set of DR response strategies
--------------------------------------------

Given a set of candidate DR strategies and day-ahead DR event conditions as an input, this model execution mode will determine which candidate strategy is most likely to maximize operator utility and provide that information as an output. Selection likelihoods are generated based on the predicted change in demand and building services under each of the candidate DR strategies and DR event conditions, and considering operator valuations of various service loss risks.

Required input files
********************

The input file for this execution mode is found in ./data/test_predict.csv. :numref:`predict-input` outlines the columns that must be present in this file.

.. _predict-input:
.. table:: Description of the input file for generating the recommendations.

   +-------------------------+-------------------------------------------------+
   | Column name             | Description                                     |
   +=========================+=================================================+
   | Name                    | Name of the DR strategy                         |
   +-------------------------+-------------------------------------------------+
   | Hr                      | Hours into the DR event, starting from 1        |
   +-------------------------+-------------------------------------------------+
   | OAT                     | Outdoor air temperature (ºF)                    |
   +-------------------------+-------------------------------------------------+
   | RH                      | Outdoor relative humidity (%)                   |
   +-------------------------+-------------------------------------------------+
   | Lt_Nat                  | Outdoor natural illuminance (lux)               |
   +-------------------------+-------------------------------------------------+
   | Lt_Base                 | Indoor lighting schedule fraction               |
   +-------------------------+-------------------------------------------------+
   | Occ_Frac                | Occupant schedule fraction                      |
   +-------------------------+-------------------------------------------------+
   | Delt_Price_kWh          | Utility incentive for DR                        |
   +-------------------------+-------------------------------------------------+
   | h_DR_Start              | Hours into the DR window                        |
   +-------------------------+-------------------------------------------------+
   | h_DR_End                | Hours since DR window ended                     |
   +-------------------------+-------------------------------------------------+
   | h_PCool_Start           | Hours since pre-cooling started                 |
   +-------------------------+-------------------------------------------------+
   | h_PCool_End             | Hours since pre-cooling ended                   |
   +-------------------------+-------------------------------------------------+
   | Delt_CoolSP             | Change in temperature set point (ºF)            |
   +-------------------------+-------------------------------------------------+
   | Delt_LgtPct             | Reduction in lighting power (%)                 |
   +-------------------------+-------------------------------------------------+
   | Delt_OAVent_Pct         | Reduction in outdoor air fraction               |
   +-------------------------+-------------------------------------------------+
   | Delt_Pl_Pct             | Reduction in plug load power (%)                |
   +-------------------------+-------------------------------------------------+
   | Delt_CoolSP_Lag         | Change in temperature set point since previous  |
   |                         | hour (ºF)                                       |
   +-------------------------+-------------------------------------------------+
   | Delt_LgtPct_Lag         | Change in lighting power since previous hour (%)|
   |                         |                                                 |
   +-------------------------+-------------------------------------------------+
   | Delt_OAVent_Pct-Lag     | Change in outdoor air fraction since previous   |
   |                         | hour (%)                                        |
   +-------------------------+-------------------------------------------------+
   | Delt_Pl_Pct_Lag         | Change in plugload power since previous hour (%)|
   +-------------------------+-------------------------------------------------+
   | Pcool_Mag               | Magnitude of pre-cooling before event started   |
   |                         | (ºF temperature decrease)                       |
   +-------------------------+-------------------------------------------------+
   | Pcool_Dur               | Duration of precooling before DR event started  |
   +-------------------------+-------------------------------------------------+


An example of this input file is `available`_ that reflects the candidate DR strategies shown in Table :numref:`DR-strategy`. 

.. _DR-strategy:
.. table:: Example set of DR strategy characteristics.

   +-------------------------+------------------------------+------------------+
   | DR strategy name        | Description                  | Magnitude        |
   +=========================+==============================+==================+
   | GTA-Low                 | Global temperature adjustment| +2ºF             |
   +-------------------------+------------------------------+------------------+
   | GTA-Moderate            | Global temperature adjustment| +4ºF             |
   +-------------------------+------------------------------+------------------+
   | GTA-High                | Global temperature adjustment| +6ºF             |
   +-------------------------+------------------------------+------------------+
   | Precool-Low             | Global temperature adjustment| +2ºF             |
   |                         | Precooling temperature change| -2ºF             |
   +-------------------------+------------------------------+------------------+
   | Precool-Moderate        | Global temperature adjustment| +4ºF             |
   |                         | Precooling temperature change| -3ºF             |
   +-------------------------+------------------------------+------------------+
   | Precool-High            | Global temperature adjustment| +6ºF             |
   |                         | Precooling temperature change| -5ºF             |
   +-------------------------+------------------------------+------------------+
   | Dimming-Low             | Lighting dimming             | -10%             |
   +-------------------------+------------------------------+------------------+
   | Dimming-Moderate        | Lighting dimming             | -20%             |
   +-------------------------+------------------------------+------------------+
   | Dimming-High            | Lighting dimming             | -30%             |
   +-------------------------+------------------------------+------------------+
   | Plug Load-Low           | Plug load reduction          | -10%             |
   +-------------------------+------------------------------+------------------+
   | Plug Load-Moderate      | Plug load reduction          | -20%             |
   +-------------------------+------------------------------+------------------+
   | Plug Load-High          | Plug load reduction          | -30%             |
   +-------------------------+------------------------------+------------------+
   | Package-Low             | Global temperature adjustment| +2ºF             |
   |                         | Lighting dimming             | -10%             |
   |                         | Plug load reduction          | -10%             |
   +-------------------------+------------------------------+------------------+
   | Package-Moderate        | Global temperature adjustment| +4ºF             |
   |                         | Lighting dimming             | -30%             |
   |                         | Plug load reduction          | -30%             |
   +-------------------------+------------------------------+------------------+
   | Package-High            | Global temperature adjustment| +6ºF             |
   |                         | Lighting dimming             | -30%             |
   |                         | Plug load reduction          | -30%             |
   +-------------------------+------------------------------+------------------+
   | Package-Low-Precool     | Global temperature adjustment| +2ºF             |
   |                         | Lighting dimming             | -10%             |
   |                         | Plug load reduction          | -10%             |
   |                         | Precooling temperature change| -2ºF             |
   +-------------------------+------------------------------+------------------+
   | Package-Moderate-Precool| Global temperature adjustment| +4ºF             |
   |                         | Lighting dimming             | -20%             |
   |                         | Plug load reduction          | -20%             |
   |                         | Precooling temperature change| -3ºF             |
   +-------------------------+------------------------------+------------------+
   | Package-High-Precool    | Global temperature adjustment| +6ºF             |
   |                         | Lighting dimming             | -30%             |
   |                         | Plug load reduction          | -30%             |
   |                         | Precooling temperature change| -5ºF             |
   +-------------------------+------------------------------+------------------+


.. _available: https://github.com/jtlangevin/flex-bldgs/blob/master/data/test_predict.csv


Running the model
******************

DR strategy recommendations and associated predictions are generated with the following command line/Terminal inputs on Windows and MacOS, respectively:

**Windows** ::

   cd Documents\projects\flex-bldgs
   py -3 flex.py --bldg_type <insert building type/vintage name> --bldg_sf <insert square footage>

**Mac** ::

   cd Documents/projects/flex-bldgs
   python3 flex.py --bldg_type <insert buildling type/vintage name> --bldg_sf <insert square footage>

Where building name options include `mediumofficenew` (~50K square foot medium office post-2004), `mediumofficeold` (~50K square foot medium office pre-2004, `retailnew` (~25K standalone retail building post-2004), and `retailold` (~25K standalone retail building pre-2004). Building square footage should be entered as-is (e.g., 50,000 for a 50K square foot building)

The model will load the input data and begin predicting the changes in demand and indoor services during each of the event hours reflected in the input file, drawing upon previously initialized models of building demand and services (see :ref:`Selecting from set of DR response strategies <mod-init>`). 


Interpretation of outputs
**************************

Outputs from execution of this mode are stored in the file ./data/recommendations.json. The file has the following structure: ::

    {
     "notes": <notes about the contents of the file>,
     "predictions": {
         "DR strategy name 1": <Percentage of simulations in which DR strategy 1 was selected>, ...
         "DR strategy name N": <Percentage of simulations in which DR strategy N was selected>, ...
      },
     "input output data": {
          "demand": {
             "DR strategy name 1": [<All sampled maximum hourly demand reduction values (W/sf) for DR strategy name 1>],
             "DR strategy name N": [<All sampled maximum hourly demand reduction values (W/sf) for DR strategy name N>]},
          "demand precool": {
             "DR strategy name 1": [<All sampled maximum hourly demand increase from precooling values (W/sf) for DR strategy name 1>],
             "DR strategy name N": [<All sampled maximum hourly demand increase from precooling values (W/sf) for DR strategy name N>]},
          "cost": {
             "DR strategy name 1": [<All sampled total economic benefit values ($) for DR strategy name 1>],
             "DR strategy name N": [<All sampled total economic benefit values ($) for DR strategy name N>]},
          "cost precool": {
             "DR strategy name 1": [<All sampled total economic loss from precooling values ($) for DR strategy name 1>],
             "DR strategy name N": [<All sampled total economic loss from precooling values ($) for DR strategy name N>]},
          "temperature": {
             "DR strategy name 1": [<All sampled maximum hourly temperature increase values (ºF) for DR strategy name 1>],
             "DR strategy name N": [<All sampled maximum hourly temperature increase values (ºF) for DR strategy name N>]},
          "lighting": {
             "DR strategy name 1": [<All sampled maximum hourly total illuminance reduction values (fraction) for DR strategy name 1>],
             "DR strategy name N": [<All sampled maximum hourly total illuminance reduction values (fraction) for DR strategy name N>]}
      }
    }


Generating baseline demand predictions
--------------------------------------

One of the functions of FlexAssist is to generate the baseline demand value given certain conditions, such as weather and occupancy, and building characteristics such as type and vintage. Models of baseline demand follow the same approach as those that were fit to predict the changes in demand and building servies under candidate DR strategies, which are used in :ref:`Generating new models <mod-pred>`.


.. The regression model for predicting the baseline demand value was already initialized and stored in the |html-filepath| ./model_stored |html-fp-end| directory, named as |html-filepath| dmd_mo_b.csv\ |html-fp-end| for medium office building type, and |html-filepath| dmd_ret_b.csv\ |html-fp-end| for retail building type. 

.. The approach of re-initializing the baseline demand model can be refered to :ref:`Generating new models <mod-init>`. The following instructions will focus on how to generate the prediction value given certain input information, leveraging the existing models.

Required input files
********************

The input file for this execution mode is found in ./data/test_predict_bl.csv. :numref:`baseline-input` outlines the columns that must be present in this file.

.. _baseline-input:
.. table:: Description of the input file for generating the baseline demand prediction.

   +-----------------------+-------------------------------------------------+
   | Column name           | Description                                     |
   +=======================+=================================================+
   | Hr                    | Hour into the DR event, starting from 1         |
   +-----------------------+-------------------------------------------------+
   | Vintage               | Four vintages are considered within the scope:  |
   |                       | 1980, 1980-2004, 2004, 2010                     |
   +-----------------------+-------------------------------------------------+
   | Hour_number           | Actual time based on 24-hour military time      |
   +-----------------------+-------------------------------------------------+
   | Climate               | Climate zone where the building is located,     |
   |                       | followed by IECC climate zone map               |
   +-----------------------+-------------------------------------------------+
   | OAT                   | Outdoor air temperature (ºF)                    |
   +-----------------------+-------------------------------------------------+
   | RH                    | Outdoor relative humidity (%)                   |
   +-----------------------+-------------------------------------------------+
   | Occ_Frac              | Occupancy schedule fraction                     |
   +-----------------------+-------------------------------------------------+
   | V1980                 |                                                 |
   | ...                   | Binary check box                                |
   | V19802004             |                                                 |
   +-----------------------+-------------------------------------------------+
   | CZ.2A                 |                                                 |
   | ...                   | Binary check box                                |
   | CZ.7A                 |                                                 |
   +-----------------------+-------------------------------------------------+


Running the model
******************

Baseline demand predictions are generated using the ``--base_pred`` option as below:

**Windows** ::

   cd Documents\projects\flex-bldgs
   py -3 flex.py --base_pred --bldg_type <insert bldg name> --bldg_sf <insert sf>

**Mac** ::

   cd Documents/projects/flex-bldgs
   python3 flex.py  --base_pred --bldg_type <insert bldg name> --bldg_sf <insert sf>

The model will automatically load in the input data and start calculating the hourly baseline demand values given the input information.

Interpretation of outputs
**************************

Predicted hourly baseline demand values are reported in ./database_predict_summary.csv. For each predicted hour, there will be 1) mean value (W/sf), and 2) standard deviation together indicating the predicted results. By default, the sample number for generating these results is set to 1000.



.. _mod-init:

Generating new models
----------------------

Users can use this mode to initialize/re-initialize all the models of building demand/services and operator utility that underly FlexAssist's predictions, given input CSV data that follows a certain data structure. The model list includes the following 6 regression models:

* Baseline demand value
* Demand changes during the DR period
* Demand changes during the pre-cooling period
* Indoor temperature changes during the DR period
* Indoor illuminance changes during the DR period
* Operator utility

.. * Indoor CO2 concentration changes during the DR period
.. * Indoor temperature changes during the pre-cooling period

Required input files
********************

Input files for this execution mode are found in the ./data directory. Current CSV `files`_ underlying the models of building demand and services were generated from a batch of simulations in EnergyPlus, where four scenarios of building types and vintages were considered. Another `file`_ with training data for the operator choice model was developed from discrete choice experiments with building operators. If users want to re-initialize the models using their own data, the format of their CSV files must be consistent with these current files. Table :numref:`init-input` shows example CSV file names underlying demand and service models sfor the medium office new vintage (post-2004); these example CSVs may serve as useful references for formatting and content. 

.. _init-input:
.. table:: Input files for generating new regression models.

   +-----------------------+-------------------------------------------------+ 
   | Input file            | Regression model                                | 
   +=======================+=================================================+
   | MO_B.csv              | Baseline demand prediction                      |
   +-----------------------+-------------------------------------------------+
   | MO_DR_new.csv         | Demand changes during the DR period             | 
   |                       +-------------------------------------------------+
   |                       | Indoor temperature changes during DR            |
   +-----------------------+-------------------------------------------------+
   | MO_Precooling_new.csv | Demand changes during the pre-cooling period    |
   |                       +-------------------------------------------------+
   |                       | Indoor temeprature changes during pre-cooling   |
   +-----------------------+-------------------------------------------------+
   | CO2_MO.csv            | Indoor CO2 concentration changes during DR      |
   +-----------------------+-------------------------------------------------+
   | Illuminance.csv       | Indoor illuminance changes during DR            |
   +-----------------------+-------------------------------------------------+
   
.. _files: https://github.com/jtlangevin/flex-bldgs/tree/master/data
.. _file: https://github.com/jtlangevin/flex-bldgs/blob/master/data/dce_dat.csv

Running the model
******************

New model initialization is executed using the ``--mod_init`` option as below:

**Windows** ::

   cd Documents\projects\flex-bldgs
   py -3 flex.py --mod_init --bldg_type <insert bldg name> --bldg_sf <insert sf>

**Mac** ::

   cd Documents/projects/flex-bldgs
   python3 flex.py --mod_init --bldg_type <insert bldg name> --bldg_sf <insert sf>

The model will start loading input data and initializing the variables for each regression model.

Interpretation of outputs
**************************

Model coefficient samples from the Bayesian inference framework are saved as pickled files (``.pkl`` ) to the ./model_stored directory. For example, for the medium office building type and new vintage (post-2004), each file represents specific model(s) as shown in :numref:`init-output`.

.. _init-output:
.. table:: Output files representing those generated regression models.

   +-----------------------+-------------------------------------------------+ 
   | Output file           | Regression model                                | 
   +=======================+=================================================+
   | dmd_mo_b.pkl          | Baseline demand prediction                      |
   +-----------------------+-------------------------------------------------+
   | dmd_mo_n.pkl          | Demand changes during the DR event period       | 
   +-----------------------+-------------------------------------------------+
   | pc_dmd_mo_n.pkl       | Demand changes during the pre-cooling period    |
   +-----------------------+-------------------------------------------------+
   | tmp_mo_n.pkl          | Indoor temperature changes during DR            |
   +-----------------------+-------------------------------------------------+
   | lt.pkl                | Indoor illuminance changes during DR            |
   +-----------------------+-------------------------------------------------+
   | dce.pkl               | Operator utility                                |
   +-----------------------+-------------------------------------------------+



..   +-----------------------+-------------------------------------------------+ 
   | Output file           | Regression model                                | 
   +=======================+=================================================+
   | dmd_mo_b.pkl          | Baseline demand prediction                      |
   +-----------------------+-------------------------------------------------+
   | dmd_mo_n.pkl          | Demand changes during the DR period             | 
   +-----------------------+-------------------------------------------------+
   | tmp_mo_n.pkl          | Indoor temperature changes during DR            |
   +-----------------------+-------------------------------------------------+
   | pc_dmd_mo_n.pkl       | Demand changes during the pre-cooling period    |
   +-----------------------+-------------------------------------------------+
   | pc_tmp_mo_n.pkl       | Indoor temeprature changes during pre-cooling   |
   +-----------------------+-------------------------------------------------+
   | co2_mo.pkl            | Indoor CO2 concentration changes during DR      |
   +-----------------------+-------------------------------------------------+
   | lt.pkl                | Indoor illuminance changes during DR            |
   +-----------------------+-------------------------------------------------+

..These models are used in other execution modes, such as :ref:`Selecting from set of DR response strategies <mod-pred>` and :ref:`Updating existing models <mod-est>`. 

.. _mod-est:

Updating existing models
-------------------------

FlexAssist enables updating of previously initialized models of change in building demand and services based on new data observations. Existing models are stored in ``.pkl`` files in the ./model_stored folder, as outlined in :numref:`init-output`. 

Required input files
********************

The input file for this execution mode is found in ./data/test_update.csv for updating the models of demand/services under DR strategies, and in ./data/test_update_bl.csv for updating the models of baseline demand. :numref:`update-input` outlines the columns that must be present in this file, and an example is found `here`_.

.. _update-input:
.. table:: Description of the input file for estimating the existing models.

   +-------------------------+-------------------------------------------------+
   | Column name             | Description                                     |
   +=========================+=================================================+
   | ID                      | DR strategy ID, not used, set to 999 for now    |
   +-------------------------+-------------------------------------------------+
   | Vintage                 | Four vintages are considered within the scope:  |
   |                         | 1980, 1980-2004, 2004, 2010                     |
   +-------------------------+-------------------------------------------------+
   | Day.type                | Weekdays as 1, weekends as 0                    |
   +-------------------------+-------------------------------------------------+
   | Day.number              | Number of the DR event, starting from 1         |
   +-------------------------+-------------------------------------------------+
   | Hour.number             | Actual time based on 24-hour military time      |
   +-------------------------+-------------------------------------------------+
   | Climate.zone            | Climate zone where the building is located,     |
   |                         | followed by IECC climate zone map               |
   +-------------------------+-------------------------------------------------+
   | Demand.Power.Diff.sf.   | Whole building reduction in electricity demand  |
   |                         | from baseline (W/sf)                            |
   +-------------------------+-------------------------------------------------+
   | Indoor.Temp.Diff.F.     | Indoor temperature difference from baseline (ºF)|
   +-------------------------+-------------------------------------------------+
   | Indoor.Humid.Diff.F.    | Indoor RH difference from baseline (ºF)         |
   +-------------------------+-------------------------------------------------+
   | Outdoor.Temp.F          | Outdoor air temperature (ºF)                    |
   +-------------------------+-------------------------------------------------+
   | Outdoor.Humid.          | Outdoor relative humidity (%)                   |
   +-------------------------+-------------------------------------------------+
   | Outdoor.Sky.Clearness.  | Outdoor sky clearness (unitless)                |
   +-------------------------+-------------------------------------------------+
   | Occ.Fraction.           | Occupancy schedule fraction (<=1)               |
   +-------------------------+-------------------------------------------------+
   | Cooling.Setpoint.Diff.  | Change in temperature set point (ºF)            |
   +-------------------------+-------------------------------------------------+
   | Lighting.Power.Diff.pct.| Reduction in lighting power (%)                 |
   +-------------------------+-------------------------------------------------+
   | Ventilation.Diff.pct.   | Reduction in outdoor air fraction               |
   +-------------------------+-------------------------------------------------+
   | MELs.Diff.pct.          | Reduction in plug load power (%)                |
   +-------------------------+-------------------------------------------------+
   | Since.DR.Started        | Hours into the DR window                        |
   +-------------------------+-------------------------------------------------+
   | Since.DR.Ended          | Hours since DR window ended                     |
   +-------------------------+-------------------------------------------------+
   | Since.Pre.cooling.Ended.| Hours since pre-cooling ended                   |
   +-------------------------+-------------------------------------------------+
   | Since.Pre.cooling.      | Hours since pre-cooling started                 |
   | Started.                |                                                 |
   +-------------------------+-------------------------------------------------+
   | Cooling.Setpoint.Diff.  | Change in temperature set point since previous  |
   | One.Step.               | hour (ºF)                                       |
   +-------------------------+-------------------------------------------------+
   | Lighting.Power.Diff.pct.| Change in lighting power since previous hour (%)|
   | One.Step.               |                                                 |
   +-------------------------+-------------------------------------------------+
   | MELs.Diff.pct.One.Step. | Change in plugload power since previous hour (%)|
   +-------------------------+-------------------------------------------------+
   | Ventilation.Diff.pct.   | Change in outdoor air fraction since previous   |
   | One.Step.               | hour (%)                                        |
   +-------------------------+-------------------------------------------------+
   | Pre.cooling.Temp.       | Magnitude of pre-cooling before event started   |
   | Increase.               | (ºF temperature decrease)                       |
   +-------------------------+-------------------------------------------------+
   | Pre.cooling.Duration    | Duration of precooling before DR event started  |
   +-------------------------+-------------------------------------------------+


.. _here: https://github.com/jtlangevin/flex-bldgs/blob/master/data/test_update.csv


Running the model
******************

Model updates are executed using the ``--mod_est`` option as below:

**Windows** ::

   cd Documents\projects\flex-bldgs
   py -3 flex.py --mod_est --bldg_type <insert bldg name> --bldg_sf <insert sf>

**Mac** ::

   cd Documents/projects/flex-bldgs
   python3 flex.py --mod_est --bldg_type <insert bldg name> --bldg_sf <insert sf>

The model will automatically load in the input data and start updating the previously initialized models.

Interpretation of outputs
**************************

Updated model parameter coefficient distribution estimates are written to the ``.pkl`` files in the ./model_stored folder. The different ``.pkl`` model types are enumerated in :numref:`init-output`.


