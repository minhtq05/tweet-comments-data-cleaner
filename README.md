# Tweet Comments Positivity Predictor
> **Warning**  
>
> This repo is a work in progress! It may contains a lot of bugs

This is my in-progress Tweets Comments Positivity Predictor project.

The repo contains raw supervised data to clean and train our model

# Usage

To clean the data, execute this command:

```bash
python clean.py
```

At the moment, We're using an SVM to predict the binary state (positive / negative) of comments.

To train the model with the cleaned dataset and find the accuracy of the model, execute this command:

```bash
python train.py
```

Features (arguments) for the clean program:

```bash
-h head: only clean the first 'head' number of items in the dataset
-s specific: clean a specific line of text in the dataset, count from 1
```
