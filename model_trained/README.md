# model_trained Folder
See https://zenodo.org/uploads/15770500 for this part.  
⚠️ Note: Manual Modification Required to Load Pretrained Models  
The name of our model class was changed after the experiments were completed and the models were saved. As a result, when attempting to load these saved models in this folder, the class name stored in the checkpoint does not match the current model class name in the code.  

To successfully load the models, you will need to manually modify the following file in your installed scvi-tools package:

```
File path (approximate):
<your_env_path>/lib/python3.x/site-packages/scvi/model/base/_base_model.py
```
Around line 712, locate and comment out the following lines:
```
# if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
#     raise ValueError("It appears you are loading a model from a different class.")
```
This will disable the model class name check, allowing the model to be loaded even if the class name has changed.  
Sorry for the inconvenience.
