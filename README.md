
# Prerequisits

The training is supervised via comet.ml. In order to run the scripts, it is necessary to sign up and provide an api key.
More information on comet.ml can be found here:
https://ftag-salt.docs.cern.ch/tutorial-Xbb/#1-fork-clone-and-install-salt


In the utils folder, place a json file "comet_setup.json". The content should be:

```
{
    "api_key": "your_api_key",
    "workspace": "your_workspace_name",
    "project_name": 
        {
            "phase2": "egamma_dnn_phase2",
            "run3_step1": "egamma_dnn_run3_step1",
            "run3_step2": "egamma_dnn_run3_step2",
            "hyperparameter_search": "egamma_dnn_hyperparameter_search"
        }
}
```


# Setup and dependencies

The following setup works for one of the lxplus-gpu nodes:

```
source /cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt/setup.sh

pip3 install virtualenv --user
virtualenv egamma_regression_env
source egamma_regression_env


pip install comet-ml
pip install -U "ray[data,train,tune,serve]"
```

# Example workflow

```
# Test samples
python3 -m plotting_scripts.plot_features_run3 --outDir validation_path

# Preprocessing: Step1
python3 -m preprocessing.preprocessing_run3_step1 --output_file_path test_step1

# Training and Validation: step1
python3 training_run3.py --step step1 --model_path model_path_step1 --model_name v1 --ideal_ic_file_path test_step1 --proc_real_ic_file_path test_step2
python3 inference.py --step step1 --model_path model_path_step1 --model_name v1 --ideal_ic_file_path test_step1 --proc_real_ic_file_path test_step2

# Preprocessing: Step2
python3 -m preprocessing.preprocessing_run3_step2 --model_path model_path_step1 --model_name v1/v1_best --input_file_path test_step1 --output_file_path test_step2

# Training and Validation: step2
python3 training_run3.py --step step2 --model_path model_path_step2 --model_name v1 --ideal_ic_file_path test_step1 --proc_real_ic_file_path test_step2
python3 inference.py --step step2 --model_path model_path_step2 --model_name v1 --ideal_ic_file_path test_step1 --proc_real_ic_file_path test_step2
```


