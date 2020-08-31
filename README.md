# net

simple implementation of FNN (Feedforward Neural Network) with DNA training. Using C++20 and CMake.

## Table of contents
- [How add to your project](#how-add-to-your-project)
- [How to use](#how-to-use)
	- [Template Parameters](#template-parameters)
	- [Construct object](#construct-object)
	- [Randomize networks](#randomize-networks)
	- [Feeding networks](#feeding-networks)
	- [Count score of networks](#count-score-of-networks)
	- [Reset score](#reset-score)
	- [Next generation](#next-generation)
	- [Get score](#get-score)
	- [Get result](#get-result)
- [Examples](#examples)
	- [XOR networks](#xor-networks)


## How add to your project
choose your variant:
* run in root of your project `git submodule add https://github.com/x1b6e6/net.git` and add line `add_subdirectory(net)` to your `CMakeLists.txt`.
* download the `Net.hh` to your project.

## How to use

### Template parameters
```c++
using net_type = net::Net<params>;
```
`params` here are neuron layers sizes.
Minimum is 2 digits:
* first param is size input layer.
* last param is size output layer.
* params between first and last is sizes of hidden layers.

example:
```c++
using net_type = net::Net<2, 4, 3>;
```
* `2` size input layer.
* `4` size hidden layer.
* `3` size output layer.

### Construct object
```c++
net_type nn{best_size, immutable};
```
* `best_size` is number of best neural networks to be  used for the next generation.
* `immutable` is number of neural networks what NOT used for the next generation but still exist in next generation (by default immutable=0).

### Randomize networks
Set random behavior for neurons.
```c++
nn.rand();
```

### Feeding networks
Compute result of neural network.
```c++
nn.feed(data);
```
* `data` here is variable (can be constant) of type `net_type::feed_type`. It's simple `std::array` with specified type and size provided by first layer.

### Count score of networks
Counting and storing score for each neural network.
```c++
auto counter = [](const net_type::result_type& result) {
  // TODO: count score of result
  return 0;
};
nn.count_score(counter);
```

### Reset score
After creating a new generation, the previous scores become irrelevant.
```c++
nn.reset_score();
```

### Next generation
Generating new generation. By default using neural networks with score that closes to 0.
```c++
nn.next();
```
You can specify the comparator (by default `Net::compare_default`). For example if your counter return error of neural network as score then you can use `std::less` for selecting best neural network by the lowest score.
```c++
nn.next<std::greater>(); // using networks with the biggest score
nn.next<std::less>(); // using networks with the lowest score
```
You can specify the number of mutations (2 by default).
Mutations can speed up training and reduce accuracy.

Note: It is recommended to reduce the number of mutations as you approach the required score.
```c++
nn.next(5);
```

### Get score
```c++
nn.score();
nn.best_score();
nn.best_score<Comparator>();
```
* `score()` return avg score of all neural networks.
* `best_score()` search score with best result selected by `Comparator` (by default `Net::compare_default`). You can also search for the worst score with this function.

### Get result
```c++
nn.result();
nn.best_result();
nn.best_result<Comparator>();
```
* `result()` return avg result of all neural networks.
* `best_result()` search best result with the best score selected by `Comparator` (by default `Net::compare_default`).

## Examples
You can also build your custom Trainer with using `SimpleNet`. Look examples network with `SimpleNet`.

### XOR networks:
* with Net: [net_xor](test/net_xor.cc)
* with SimpleNet: [simple_xor](test/simple_xor.cc)
