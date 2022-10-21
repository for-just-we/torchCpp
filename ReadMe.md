load [learch](https://github.com/eth-sri/learch) model with libtorch, torch version is 1.4.0

build

- `mkdir build && cd build`

- `cmake -DLIB_TORCH_ROOT=<path2libtorch> ..`

- `cmake --build`

currently we load the model in a blacl box way, the code in `learch_model.h` is unused, which we will supplement laterly.

You can refer to `load_learch_model.py` to see how we convert original learch model into model could be loaded with libtorch.