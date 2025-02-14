# SemTon

Semantics-driven Traditional Chinese Medicine Prescription Recommendation with Symptom-based Tongue Manifestation Completion

## Environment

```
python==3.9.18
torch==2.1.1
tqdm==4.66.1
scikit-learn==1.3.2
```

You can build the conda environment for our experiment using the following command:

```
conda env create -f environment.yml
```

### Datasets

We have disclosed the details of the dataset along with examples in the Appendix.

Here, we extracted 50 complete samples in each of RSJ1 and SZY for reference. Please go to the `data` folder to view them.

The full datasets will be made publicly avail able once the manuscript is accepted.

## Run the SemTon

You can train and test the model using the following command:

```python
cd src
# Stage 1 for debais if you need, please download Qwen2.5-VL weights at https://github.com/QwenLM/Qwen2-VL
python debias_fine-tuning.py
python list_chat.py
# Stage 2 for tongue completion
python local_fine-tuning.py
# Stage 3 for TCMPR
python joint_fine-tuning.py
```

We provide the weights of the fine-tuned TMD module as well as the STMC module, please download at: https://pan.quark.cn/s/fad8b87e0c5d   key：EbLW

## Acknowledgement

None
