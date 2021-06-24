

###### (Automatically generated documentation)

# Dynamic DR

## Description
This measure implements demand flexibility measures, including lighting, plugloads, cooling, and heating, for Summer, Winter, and All year. Lighting and plugloads measures are applicable in all three scenarios, while cooling and heating are applicable only in the Summer scenario and Winter scenario, respectively.In the Summer scenario, as for example, four individual flexibility strategies, which are applied during the DR event hours of 3-7 PM include 1) lighting dimming, 2) plug load reduction through low-priority device switching, 3) global temperature adjustment (GTA), and 4) GTA + pre-cooling. The reductions are generated using a continuous uniform dbutions bounded from 0 to 100%, adjustment settings for GTA and pre-cooling are generated using a discrete uniform distribution; GTA cooling set point increases during the DR period are sampled between the range of 1F and 6F, while pre-cooling set point decreases are sampled between the range of 1F and 4F with the duration from 1 hour to 8 hours prior to the DR event start. The adjustments are applied on the baseline hourly settings using a Compact:Schedule to maintain the same Design Days settings as those in the baseline.

## Modeler Description
File:measure.rb, resources/original_schedule.csv, resources/ScheduleGenerator.rb. There are two steps to implement the measure. First, a modeler generates an hourly baseline schedule of the interest by running the model. A previously generated schedule is also available in the resources/original_schedule.csv. The selected schedules are available for three building types (medium office detailed, large office detailed, and retail stand alone) in two vintages (post-1980 and 2010) and a big box retail model in 2010 vintage. The big box retail model is only available in an EnergyPlus model, which this measure is not applicable. Second, a modeler loads the model and runs the measure by selecting "Apply Measure Now" under the menu "Components & Measure" in the top bar of OpenStudio GUI. The measure is located under "Whole Building" >> "Whole Building Schedules".

## Measure Type
ModelMeasure

## Taxonomy


## Arguments


### Select the building type

**Name:** buildingType,
**Type:** Choice,
**Units:** ,
**Required:** true,
**Model Dependent:** false

### Select the vintage

**Name:** vintage,
**Type:** Choice,
**Units:** ,
**Required:** true,
**Model Dependent:** false

### Select the period

**Name:** drPeriod,
**Type:** Choice,
**Units:** ,
**Required:** true,
**Model Dependent:** false

### Select the demand response type

**Name:** drType,
**Type:** Choice,
**Units:** ,
**Required:** true,
**Model Dependent:** false

### Choose a schedule to be replaced.

**Name:** schedule_old,
**Type:** Choice,
**Units:** ,
**Required:** true,
**Model Dependent:** false

### Use pre-defined schedule?

**Name:** usepredefined,
**Type:** Boolean,
**Units:** ,
**Required:** true,
**Model Dependent:** false




