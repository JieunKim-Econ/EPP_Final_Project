## Project for Effective Programming Practices for Economists | Winter 2020, M.Sc. Economics, Bonn University | Jieun Kim


This repository presents my replication of the results from the following paper:

> González, Libertad. (2013). ["The Effect of a Universal Child Benefit on Conceptions, Abortions, and Early Maternal Labor Supply."](https://www.aeaweb.org/articles?id=10.1257/pol.5.3.160) American Economic Journal: Economic Policy, 5 (3): 160-88.
 

### Replication of Libertad González (2013) 


González (2013) examines the effects of a child subsidy on fertility, household expenditure, and maternal labor supply in Spain. The distinguishing feature of this paper is that it applies regression discontinuity(RD) design as well as difference in differences (DID) estimates. From an “unanticipated” introduction of a new law with benefit eligibility based on a certain date, the author exploits a natural experiment with randomization and the sharp cutoff. A universal child benefit, a one-time cash payment of €2,500, was announced on July 3 in 2007 in Spain, and mothers were eligible if their child was born after July 1 and this date was not announced beforehand. This setup enables RD analysis where the treatment effect is the difference in outcomes between treated (who receive a benefit) and control families (who do not) close to the cutoff.      

The purpose of this notebook is to revisit the main result and explore the policy impacts from more expanded perspective. Thus, I extend the analysis of the labor supply effect with different subpopulation grouped by parent’s education level and marital status.


### Contents of the repository 
* **Replication_Project.ipynb**: My paper for the replication project.  

* **auxiliary**: All codes for tables and figures in the paper for replication as well as for extension are included. To check the accuracy of the replication, the replicated outcomes are compared with the actual outcomes from the paper via _unittest_. You can find these tests in the last part of ‘_Tables.py_’ and ‘_Table3.py_’. 

* **data**: Raw data from the author.

* **files**: Images for the intuitive description of the paper.


### References


*	**Lee, David S., and Thomas Lemieux. (2010).** “Regression Discontinuity Designs in Economics.” Journal of Economic Literature 48 (2): 281–355.

*	**Imbens, Guido W., and Thomas Lemieux. (2008).** “Regression Discontinuity Designs: A Guide to Practice.” Journal of Econometrics 142 (2): 615–35. 
 
*	**Olga Malkova. (2018).** “Can Maternity Benefits Have Long-Term Effects on Childbearing? Evidence from Soviet Russia.” The Review of Economics and Statistics 100:4, 691-703. 

*	**Soohyung Lee, Chiara Orsini. (2018).** “Girls and boys: Economic crisis, fertility, and birth outcomes”. Journal of Applied Econometrics 33:7, 1044-1063.
