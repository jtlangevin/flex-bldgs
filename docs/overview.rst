.. _overview:

An Overview of FlexAssist
=========================

What is FlexAssist and why is it needed?
----------------------------------------

* FlexAssist is a recommender engine that helps commercial building owners and operators of office and retail building types effectively participate in demand response (DR) programs.
* Given a set of contextual conditions forecasted for DR events on a day-ahead basis and owner/operator valuations of building services, FlexAssist determines which in a set of candidate building control strategies is likely to strike the best balance between the economic benefits of DR event participation and the risk of building service losses. 
* FlexAssist addresses the need to provide building operators with decision-support resources as utilities seek to expand commerical DR programs and leverage greater flexibility in energy demand to help integrate variable renewable energy supply and improve the resilience of the electric grid. Expanded DR programs offer increasing opportunities for commercial buildings to participate in DR -- e.g., through the services of load aggregators -- yet, potential participants often lack the resources needed to assess the benefits and risks of responding to frequent DR event calls from the grid. FlexAssist fills this gap by automating the process of conducting a benefit-risk assessment in advance of an event and determining how best to respond, and by learning from information collected during each event to continually improve the underlying prediction framework.

Operator preferences are directly accounted for
-----------------------------------------------

* `Discrete choice`_ modeling translates predicted changes in building demand and services under candidate DR strategies into an operator utility score.
* Choice model coefficients are drawn from discrete choice experiments that infer operator weightings of various load adjustments (e.g. temperature, lighting, etc.) against potential economic benefits.
* Currently considered inputs to the choice model include the expected economic benefit of implementing a candidate DR strategy during a given event, as well as the expected maximum increase in temperature, lighting reduction, and reduction in plug load power availability from implementing the candidate DR strategy.

Recommendations reflect decision-making uncertainties
-----------------------------------------------------

* Underlying models estimating the change in building demand and services under DR and operator utility are implemented using `probabilistic programming techniques`_ that directly represent the uncertainty in model predictions.
* Model outputs are provided as a range of values, rather than single point values, and communicate the likelihood of service losses and economic benefits under candidate DR strategies.
* Attaching operator valuations from the discrete choice experiments to probable service losses yields a powerful risk assessment framework to guide DR participation in a particular context and under a particular set of conditions.

Underlying models can be updated with new DR event data
-------------------------------------------------------

* FlexAssist is conceptualized as a `Bayesian decision network`_ that merges prior expectations about key modeled relationships (e.g., between change in thermostat set point from a DR strategy and demand reduction) with observed evidence on these relationships.
* Under this framework, model parameters are considered random and are assigned a probability distribution that is easily updated with new evidence.
* In practice, this means that FlexAssist's predictions may be automatically re-calibrated with evidence of the actual effects of DR strategies in real-world settings, given the collection of relevant event data (e.g., metered demand, indoor environmental conditions, weather, occupancy).


.. _Discrete choice: https://en.wikipedia.org/wiki/Discrete_choice
.. _probabilistic programming techniques: https://en.wikipedia.org/wiki/Probabilistic_programming
.. _Bayesian decision network: https://en.wikipedia.org/wiki/Bayesian_network