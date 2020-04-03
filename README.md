# Information State Embedding in Partially Observable MARL 

This is the PyTorch implementation of the paper [Information State Embedding in Partially Observable Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2004.01098). Please consider citing our paper if you find this code useful:

```
@article{mao2020information,
  title={Information State Embedding in Partially Observable Cooperative Multi-Agent Reinforcement Learning},
  author={Mao, Weichao and Zhang, Kaiqing and Miehling, Erik and Başar, Tamer},
  journal={arXiv preprint arXiv:2004.01098},
  year={2020}
}
```


## Dependencies
- Python 3.5
- PyTorch 1.4
- scikit-learn 0.22.2


## Examples
- Default parameter values: To test the three embedding instances in their default settings, simply run:
```
python FM-E.py
python RNN-E.py
python PCA-E.py
```
- Specifying parameters: Performance varies when you use different parameter values on different tasks. To test with your own parameter values, run:
```
python FM-E.py --sequence_size 20 --length 4 --lr 0.01 --task 'boxpushing'
python RNN-E.py --sequence_size 10 --lr 0.01 --task 'grid3x3'
python PCA-E.py --sequence_size 4 --pca_length 8 --lr 0.01 --task 'dectiger'
```

