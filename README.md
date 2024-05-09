# Physics-Driven Spectrum-Consistent Federated Learning for Palmprint Verification

This is the official implementation for ''Physics-Driven Spectrum-Consistent Federated Learning for Palmprint Verification". This work was accepted by International Journal of Computer Vision, and this paper can be found at [here](https://link.springer.com/article/10.1007/s11263-024-02077-9).

#### Abstract:
Palmprint as biometrics has gained increasing attention recently due to its discriminative ability and robustness. However, existing methods mainly improve palmprint verification within one spectrum, which is challenging to verify across different spectrums. Additionally, in distributed server-client-based deployment, palmprint verification systems predominantly necessitate clients to transmit private data for model training on the centralized server, thereby engendering privacy apprehensions. To alleviate the above issues, in this paper, we propose a physics-driven spectrum-consistent federated learning method for palmprint verification, dubbed as PSFed-Palm. PSFed-Palm draws upon the inherent physical properties of distinct wavelength spectrums, wherein images acquired under similar wavelengths display heightened resemblances. Our approach first partitions clients into short- and long-spectrum groups according to the wavelength range of their local spectrum images. Subsequently, we introduce anchor models for short- and long-spectrum, which constrain the optimization directions of local models associated with long- and short-spectrum images. Specifically, a spectrum-consistent loss that enforces the model parameters and feature representation to align with their corresponding anchor models is designed. Finally, we impose constraints on the local models to ensure their consistency with the global model, effectively preventing model drift. This measure guarantees spectrum consistency while protecting data privacy, as there is no need to share local data. Extensive experiments are conducted to validate the efficacy of our proposed PSFed-Palm approach. The proposed PSFed-Palm demonstrates compelling performance despite only a limited number of training data.

#### Requirements

Our codes were implemented by ```PyTorch 1.10``` and ```11.3``` CUDA version. If you wanna try our method, please first install necessary packages as follows:

```
pip install requirements.txt
```


#### Citation
If our work is valuable to you, please cite our work:
```
@article{yang2023physics,
  title={Physics-Driven Spectrum-Consistent Federated Learning for Palmprint Verification},
  author={Yang, Ziyuan and Teoh, Andrew Beng Jin and Zhang, Bob and Leng, Lu and Zhang, Yi},
  journal={International Journal of Computer Vision},
  year={2024}
}
