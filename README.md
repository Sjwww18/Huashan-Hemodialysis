![image](https://github.com/user-attachments/assets/582ae051-441f-4713-ba20-8364ec127bb7)# Huashan-Hemodialysis
Use data to predict disease.

## `Notification`
`No standard DNB, only DNB that can predict disease matters.`

`We only need to predict the disease before doctors.`

`Not all the columns in .csv all useful.`

## 1. Five Possible States
### The Worst
No DNB.
### Bad Case
Some DNBs, but fail to predict seious disease.
### Neutral Case
Many DNBs, succeed in predicting certain disease but lack generalization ability. Many misjudgments.
### Good Case
Some DNBs to predict diseases; other DNBs mislead to serious illness, but the ratio is low(about 25%). In this case, we can start Overleaf.
### The Best
Some DNBs to predict diseases; some DNBs to predict illness like cold, fever and can be interpreted by LLM.

## 2. Main Idea
### Prediction
Use 70% Bad Cases to train the DNB Searching ability and the other 30% to valid our assumption and verify the generalization ability.
### Overfitting
Use 70% Good Cases to avoid the problem of overfitting and the other 30% to valid our assumption and increase the interpretation ability.

## 3. Main Problem
### Few Bad Cases
Few data to train the model's prediction ability.
### Outer and Inner Disease
Some diseases may occur far before we know.

`15	678 -	2023/8/29	低血压事件	2023/7/29`

`25	544 -	2024/8/31	冠心病	2024/4/11`

Some diseases cannot be predicted by blood information.

`39	468 -	2024/9/17	骨盆骨折	2024/9/1`

### The Most Important Disease
Low Blood Pressure Incident.

Disease that can be inferred by blood quality.

`12	329 -	2024/4/16	肺部感染	2024/4/13`

`33	520 -	2024/10/10	内瘘失功	2023/10/5`

### (Maybe) Too many DNBs
This case is possible because body can refresh itself.
