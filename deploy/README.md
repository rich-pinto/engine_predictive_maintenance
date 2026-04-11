---
title: Engine Predictive Maintenance
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Engine Predictive Maintenance

This Space hosts a Streamlit application for predicting whether an engine is operating in a normal condition or in a maintenance-required condition.

## How it works
- The app collects engine sensor inputs from the user
- It builds a single-row dataframe with the required feature structure
- It loads the trained predictive maintenance model
- It returns the predicted engine condition and, where available, the prediction probability

## Input Features
- Engine rpm
- Lub oil pressure
- Fuel pressure
- Coolant pressure
- lub oil temp
- Coolant temp

## Output
- Engine Condition
  - 0 = normal
  - 1 = maintenance required
