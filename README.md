# Rookie QB predictions
### Predict how well a QB will do over his entire career given his rookie stats using ML. 

## Summary
In this project, I attempt to prove the following:
- It is incredibly difficult to determine how good a QB after their rookie year will be, even using the latest state-of-the-art tabular prediction methods.
- It is possible to train a model to determine what "tier" (basically predict stdev) a QB will be with some level of accuracy.

I used this project as practice for learning the fastai machine learning framework after watching the updated [course](https://course.fast.ai/). 
See the notebooks for data and model exploration and implementation, as well as comments on the entier process. Final model saved to export.pkl. 
With the fastai tabular model, I was able to achieve a ~55% accuracy on the validation set.

Check out the [demo](https://huggingface.co/spaces/mattricesound/Rookie-QB-Predictions-Name) (hosted on Hugging Face Spaces!): 
In the demo, you input a QB name and it will output the tier prediction based on his rookie stats. 

There's also a separate [demo](https://huggingface.co/spaces/mattricesound/Rookie-QB-Predictions-Stats) where you can craft a QB with custom stats.
I've noticed that it performs very poorly if you give it terrible stats, since it was trained on QBs with some baseline stats.

Data from [Pro Football Reference](https://www.pro-football-reference.com/)

Model built with [fastai](https://www.fast.ai/)

Demo built with [Gradio](https://gradio.app/)

## Dataset
Pro Football Reference has a nice [page](https://www.pro-football-reference.com/players/qbindex.htm) with all the players who ever played QB. 
To remove some outliers, I restriced the list to QBs sincen 1969 (AFL-NFL Merger). When collecting the stats for each QB, I counted a QB's 'rookie year' as the first year they started in > 2 games, and filtered out any QB who didn't have > 3 such years in their career. This left me with a list of 406 QBs (see rookie_year.csv). I used the following columns as part of the tabular model: Completions, Attempts, Yards, Completion Percentage, Touchdowns, Interceptions, Sacks, Yards/Game. The dependent variable ("tier") is based on Pro Football Reference [Appoximate Value](https://www.pro-football-reference.com/about/approximate_value.htm) or AV.
Basically, I assume the AV is a normal distribution, then zscore every QBs AV into 5 "tiers" (< -1.5, -1.5 ↔ -0.5, -0.5 ↔ 0.5, 0.5 ↔ 1.5, > 1.5).  

## Model 
The goal is to train a model to accuratly predict a QB's tier given their selected rookie stat categories. 
First I tested out decision trees/random forests, but found that they tended to place all the QBs in the middle tier and get decent accuracy. 
I then tested fastai's [Tabular model](https://docs.fast.ai/tabular.model.html) which performed much better and would have a larger variety of predictions. 


<img width="935" alt="Hugging Face Spaces" src="https://user-images.githubusercontent.com/18355302/174906518-7fa50137-326d-4250-a454-015be3849891.png">



Hugging Face config:

---
title: Rookie QB Predictions
emoji: 🦀
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 3.0.19
app_file: app_data.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
