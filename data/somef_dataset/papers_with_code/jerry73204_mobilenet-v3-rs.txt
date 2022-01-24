# MobileNetV3 in Rust

Google's MobileNetV3 implemented using [tch-rs](https://github.com/LaurentMazare/tch-rs) framework, a Rust-binding of libtorch. MobileNetV3 model design was proposed on [arXiv](https://arxiv.org/abs/1905.02244), and this project takes [pytorch-mobilenet-v3
](https://github.com/kuan-wang/pytorch-mobilenet-v3) as reference implementation.

## Usage

### Build

Make sure your stable Rust and Cargo are ready, and follow `cargo build` to compile project. It's recommended to check out [tch-rs' README](https://github.com/LaurentMazare/tch-rs) to download pre-built libtorch binaries to speed up build process.

This project provides a demo training executable on MNIST and CIFAR-10. The library interface is also available if you would like to integrate with your project.

### Run demo training

* CIFAR-10: Download binary version from [CIFAR site](https://www.cs.toronto.edu/~kriz/cifar.html), and run:

```sh
cargo run -- --dataset-name cifar-10 --dataset-dir /path/to/cifar-10-dir
```

* MNIST: Download and unpack all gzips to a directory from [MNIST site](http://yann.lecun.com/exdb/mnist/), and run:

```sh
cargo run -- --dataset-name mnist --dataset-dir /path/to/mnist-dir
```

### Use as library

Here is example usage. It's suggested to visit our source code to understand details.

```rust
// model init
let mut vs = VarStore::new(Device::Cuda(0));
let root = vs.root();
let model = MobileNetV3::new(
    &root / "mobilenetv3",
    input_channel,
    n_classes,
    dropout,
    width_mult,  // usually 1.0
    Mode::Large,
)?;
let opt = Adam::default().build(&vs, learning_rate)?;

// training
let logits = model.forward_t(&images, true);
let loss = prediction_logits.cross_entropy_for_logits(&labels);
opt.backward_step(&loss);
```

## License

MIT, see [LICENSE](LICENSE) file
