# Transformer.Pytorch

Pytorch implementation of the **Transformer** Model as proposed by the paper [Attention Is All You Need
](https://arxiv.org/abs/1706.03762).

## Model Architecture

![](./assets/transformer_encoder_decoder.png)

**Encoder:** The encoder is composed of a stack of `N = 6` identical layers. Each layer has two
sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise
fully connected feed-forward network. We employ a residual connection around each of
the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is
`LayerNorm(x + Sublayer(x))`, where `Sublayer(x)` is the function implemented by the sub-layer
itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
layers, produce outputs of dimension `d_model = 512`.

**Decoder:** The decoder is also composed of a stack of `N = 6` identical layers. In addition to the two
sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position `i` can depend only on the known outputs at positions less than `i`.

## Attention

![](./assets/scaled_dot_product_attention.png)

![](./assets/scaled_dot_product_attention_formula.gif)

![](./assets/multi_head_attention.png)

![](./assets/multihead_formula_1.gif)

such that, ![](./assets/multihead_formula_2.gif)

## Reference

```
@misc{
    1706.03762,
    Author = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    Title = {Attention Is All You Need},
    Year = {2017},
    Eprint = {arXiv:1706.03762},
}
```