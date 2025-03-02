# Huashan-Hemodialysis
Use data to predict disease.

## Five Possible States
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

## Main Idea
### Prediction
Use 70% Bad Cases to train the DNB Searching ability and the other 30% to valid our assumption and verify the generalization ability.
### Overfitting
Use 70% Good Cases to avoid the problem of overfitting and the other 30% to valid our assumption and increase the interpretation ability.

## Main Problem
### Few Bad Cases
Few data to train the model's prediction ability.
### Outer and Inner Disease
Some diseases cannot be predicted by blood information.
### (Maybe) Too many DNBs
This case is possible because body can refresh itself.
