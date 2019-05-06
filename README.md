# Whales classification

The template was taken from [stanford](https://cs230-stanford.github.io/project-code-examples.html) template and a popular more practical [template](https://www.reddit.com/r/MachineLearning/comments/7ven9f/p_best_practice_for_tensorflow_project_template/). We took best things from both, merged it with our experience and created this structure to be used even outside of DataRoot University. 

This template combines  **simplcity**, **best practice for folder structure** and **good OOP design**.


#Install requirements
```bash
pip install -r requirements.txt
```

Run dataset preprocessor
```bash
cd data
python prepare_whales.py
```

To train the model use:
```bash
cd mains
python main_whales.py -c ../configs/whales.json
```

Open one more terminal window and run tensorboard. Watch metrics in near real-time
```bash
tensorboard --logdir experiments/whales/summary
```
