# DR_modeling_oemof
Some work in progress for further developping the SinkDSM component of the framework oemof

Examples have been based on those from this repo: https://github.com/windnode/SinkDSM_example

## Files overview
| File | description |
| --- | --- |
| oemof_sink_DSM.py | Demand response implementation by DIW, see Zerrahn & Schill (2015a, pp. 842-843) and Zerrahn and Schill (2015b) |
| oemof_DR_component_DLR_naming_adjusted_shifting_classes.py | Demand response implementation by DLR, see Gils (2015, pp. 67-70); implementation contains load shedding as well as demand response shifting classes; terminology from the oemof SinkDSM component is used |
| oemof_DR_component_DLR_naming_adjusted_no_shed.py | Demand response implementation by DLR, see Gils (2015, pp. 67-70); implementation doesn't contain load shedding as well as demand response shifting classes; terminology from the oemof SinkDSM component is used |
| oemof_DR_component_IER_naming_adjusted.py | Demand response implementation by IER, see Steurer (2017, pp. 80-82); terminology from the oemof SinkDSM component is used | oemof_DR_component_TUD_naming_adjusted.py | Demand response implementation by TU Dresden, see Ladwig (2018, pp. 90-93); implementation does not include Power-to-X; terminology from the oemof SinkDSM component is used |
| DSM-Modelling-Example.ipynb | A jupyter notebook containing some examples for usage of the components |
| plotting.py | A module for results extraction and plots (needs to be revised) |

## Usage
* Run DSM-Modelling-Example.ipynb to get an impression on how the different approaches behave.
* Have a look at the implementations for development issues.

## Literature
* Gils, Hans Christian (2015): 
Balancing of Intermittent Renewable Power Generation by Demand Response and 
Thermal Energy Storage, Stuttgart, http://dx.doi.org/10.18419/opus-6888, 
accessed 16.08.2019.
* Ladwig, Theresa (2018):
Demand Side Management in Deutschland zur Systemintegration erneuerbarer
Energien, Dresden, https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa-236074,
accessed 02.05.2020.
* Steurer, Martin (2017): 
Analyse von Demand Side Integration im Hinblick auf eine effiziente und 
umweltfreundliche Energieversorgung, Stuttgart, 10.18419/opus-9181,
accessed 17.08.2019.
* Zerrahn, Alexander and Schill,
Wolf-Peter (2015a): On the representation of demand-side management in power
system models, in: Energy (84), pp. 840-845, 10.1016/j.energy.2015.03.037.
accessed 16.08.2019, pp. 840-845.
* Zerrahn, Alexander and Schill, Wolf-Peter (2015b):
A Greenfield Model to Evaluate Long-Run Power Storage Requirements
for High Shares of Renewables, DIW Discussion Papers, No. 1457.
