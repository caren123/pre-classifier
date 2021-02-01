# pre-classifier
There are two use cases to deploy pre-classifier:
1. Run on main content
2. Ran on video assets

To run on main content, we need to include two steps:
- Step1: Given a main content, segment it into 30-seconds mini clips
- Step2. For each 30-seconds mini clips, run pre-classifier

To run on video assets, we just need to run Step2.

In both use case, pre-classifier outputs top 3 prediction of class name and probability scores.  
