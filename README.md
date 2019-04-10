# DeepTP

Reliable 4D aircraft trajectory prediction, whether in a real-time setting or for analysis of counterfactuals, is important to the efficiency of the aviation community. In this paper, we first develop an efficient tree-based matching algorithm to construct image-like feature maps for historical flight trajectories from high-fidelity meteorological datasets – wind, temperature and convective weather. We then model the track points on trajectories as conditional gaussian mixtures with parameters to be learned from our proposed deep generative model, which is an end-to-end convolutional recurrent neural network that consists of a Long Short-Term Memory (LSTM) encoder network and a mixture density LSTM decoder network. The encoder network embeds last-filed flight plan information into fixed-length state variables and feed to the decoder network, which further learns the spatiotemporal correlations from the historical flight tracks and outputs the parameters of gaussian mixtures. Convolutional layers are integrated into the pipeline to learn feature representations from the high-dimensional weather feature maps. During the inference process, beam search and adaptive Kalman filter (with Rauch-Tung-Striebel smoother) algorithms are used to prune the variance of generated trajectories.

Manuscript: https://arxiv.org/abs/1812.11670

Suggested citation: 

```
@misc{1812.11670,
Author = {Liu, Yulin and Hansen, Mark},
Title = {Predicting Aircraft Trajectories: A Deep Generative Convolutional Recurrent Neural Networks Approach},
Year = {2018},
Eprint = {arXiv:1812.11670},
}
```

Inference framework:

![Inference](https://github.com/yulinliu101/DeepTP/blob/master/output_figs/Framework%20%E2%80%93%20Inference.gif)

To run feature engineering:

```
cd src
python run_feature_cube_generator.py
```

To train from scratch:

```
cd src
python Run_RNN_model_Lite.py --train_or_predict train --config configs/encoder_decoder_nn_lite.ini
```

To train from some pretrained models:

```
cd src
python Run_RNN_model_Lite.py --train_or_predict train --config configs/encoder_decoder_nn_lite.ini --name PATH/TO/MODEL --train_from_momdel True
```

To sample trajectories:

```
cd src
python Run_RNN_model_Lite.py --train_or_predict predict --config configs/encoder_decoder_nn_lite.ini --name PATH/TO/MODEL
```

The following are examples of generated flight tracks.

![Example 1 of generated flight tracks](https://github.com/yulinliu101/DeepTP/blob/master/output_figs/generated_samp1.png)
![Example 2 of generated flight tracks](https://github.com/yulinliu101/DeepTP/blob/master/output_figs/generated_samp6.png)
