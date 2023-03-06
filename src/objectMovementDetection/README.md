LIBRARY TO INSTALL
- TO BE CONTINUE

**main.py:** File to train **(RL_TRAIN)**, manually play **(MANUAL)**, file to deploy **(RL_DEPLOY)** <br>

**const.py:** Contains constant variables<br>

**statistic.py:** Plotting to observe result<br>

**table.py:** All Reinforcement Algorithm here<br>

**utils.py:** External function to caculate stuff<br>

**statistic_old_format:** For Thinh shit format data :)))<br>

**test.py:** Just for testing<br>

----------------------

Author: @MinhTuLeHoang

**Anytime_D_StartV2.py:** for D*
**plottingV2:** for plotting D*
| Name   | Type  | Description                        |
| :----- | :---: | :--------------------------------- |
| ADStar | Class | init class ADStar for path looking with parram: <br>  <ul> <li>**start:** Start Point</li> <li>**goal:** Goal Point</li> <li>**eps:** Epsilon</li> <li>**heuristic_type:** For distance calculation</li> <li>**maps:** Maps in array type</li> </ul> <br> First of all, we need to init this class, then call **run()** function. |
| onChange | Function of ADStar | Whenever map changes, call this function, then pass new **array** map as parameter. |
| getPath | Function of ADStar | To get current path (**list of points**) |


