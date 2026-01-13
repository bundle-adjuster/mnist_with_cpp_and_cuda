# Backpropagation Implementation Explanation

This document explains how gradients are computed and how backpropagation is implemented in this CUDA-based neural network.

## Overview

The implementation uses **batch gradient descent** with backpropagation. The key steps are:
1. **Forward Pass**: Compute activations and store pre-activation values
2. **Loss Computation**: Calculate cross-entropy loss
3. **Backward Pass**: Compute error signals (deltas) for each layer
4. **Gradient Computation**: Calculate weight and bias gradients
5. **Weight Update**: Update weights and biases using gradients

---

## 1. Forward Pass

### What Happens:
For each layer `i` connecting layer `i` to layer `i+1`:

```cpp
// Step 1: Compute pre-activation (z = input * weights + bias)
z[i+1] = input[i] × W[i] + b[i]

// Step 2: Store pre-activation (needed for ReLU derivative later)
d_pre_activations_[i+1] = z[i+1]

// Step 3: Apply activation function
if (hidden layer):
    a[i+1] = ReLU(z[i+1])  // stored in d_activations_[i+1]
else (output layer):
    a[i+1] = softmax(z[i+1])  // stored in d_activations_[i+1]
```

### Implementation Details:
- **Matrix multiplication**: `output = input × weights` where:
  - `input` is `(batch_size × input_size)`
  - `weights` is `(input_size × output_size)` stored as `weights[input_idx * output_size + output_idx]`
  - `output` is `(batch_size × output_size)`
- **Pre-activations are stored** in `d_pre_activations_[i+1]` for later use in backpropagation
- **Post-activations** (after ReLU/softmax) are stored in `d_activations_[i+1]`

---

## 2. Loss Computation

### Cross-Entropy Loss:
For each sample in the batch:
```
L = -log(p_target_class)
```
where `p_target_class` is the softmax probability of the correct class.

### Implementation:
```cpp
cuda_cross_entropy_loss(output, target, loss, num_classes, batch_size)
```
Computes loss for each sample in the batch.

---

## 3. Backward Pass: Computing Deltas (Error Signals)

Backpropagation computes the gradient of the loss with respect to each layer's pre-activations. These gradients are called **deltas** (δ).

### 3.1 Output Layer Delta

**Mathematical Formula:**
For softmax + cross-entropy loss, the gradient is:
```
δ[L] = ∂L/∂z[L] = output - target_one_hot
```
where:
- `output` is the softmax probabilities `(batch_size × num_classes)`
- `target_one_hot` is one-hot encoded targets

**Implementation:**
```cpp
// In compute_output_delta_kernel:
delta[idx] = output[idx] - (class_idx == target_class ? 1.0f : 0.0f);
```

This is stored in `d_deltas_[num_layers_ - 2]` (delta for the last hidden layer before output).

### 3.2 Hidden Layer Deltas

**Mathematical Formula:**
For hidden layer `i`, the delta is computed as:
```
δ[i] = (W[i+1]^T × δ[i+1]) ⊙ ReLU'(z[i])
```

Where:
- `W[i+1]^T` is the transpose of weights connecting layer `i` to `i+1`
- `δ[i+1]` is the delta of the next layer
- `ReLU'(z[i])` is the ReLU derivative: `1` if `z[i] > 0`, else `0`
- `⊙` is element-wise multiplication

**Implementation:**
```cpp
// In compute_hidden_delta_kernel:
// For each neuron in current layer:
sum = 0
for each neuron j in next layer:
    sum += delta_next[j] * weights[current_neuron * next_size + j]

// Apply ReLU derivative
if (pre_activation[current_neuron] > 0):
    delta[current_neuron] = sum
else:
    delta[current_neuron] = 0
```

**Key Points:**
- Weights are stored as `(current_size × next_size)`, so `weights[neuron * next_size + j]` is the weight from `neuron` (current layer) to `j` (next layer)
- The ReLU derivative uses **pre-activation values** (`d_pre_activations_[i+1]`) to determine if the gradient should flow through
- This is computed backwards from the output layer to the input layer

---

## 4. Gradient Computation for Weights and Biases

Once we have the deltas, we compute the gradients of the loss with respect to weights and biases.

### 4.1 Weight Gradients

**Mathematical Formula:**
```
∂L/∂W[i] = (1/batch_size) × input[i]^T × δ[i]
```

Where:
- `input[i]` is `(batch_size × input_size)` - the activations from previous layer
- `δ[i]` is `(batch_size × output_size)` - the delta for current layer
- Result is `(input_size × output_size)` - same shape as weights

**Implementation:**
```cpp
// In update_weights_kernel:
// For each weight W[input_idx][output_idx]:
grad = 0
for each sample b in batch:
    grad += prev_activations[b][input_idx] * delta[b][output_idx]

gradient = grad / batch_size  // Average over batch
```

**Matrix Form:**
This computes the outer product averaged over the batch:
```
gradient[input_idx][output_idx] = 
    (1/batch_size) × Σ_b (prev_activations[b][input_idx] × delta[b][output_idx])
```

### 4.2 Bias Gradients

**Mathematical Formula:**
```
∂L/∂b[i] = (1/batch_size) × Σ_batch δ[i]
```

The bias gradient is simply the average of the deltas over the batch.

**Implementation:**
```cpp
// In update_biases_kernel:
grad = 0
for each sample b in batch:
    grad += delta[b][neuron_idx]

gradient = grad / batch_size
```

---

## 5. Weight Updates

After computing gradients, weights and biases are updated using gradient descent:

```
W[i] = W[i] - learning_rate × (∂L/∂W[i])
b[i] = b[i] - learning_rate × (∂L/∂b[i])
```

**Implementation:**
```cpp
weights[input_idx * output_size + output_idx] -= learning_rate * grad / batch_size;
biases[neuron_idx] -= learning_rate * grad / batch_size;
```

---

## Complete Flow Example

For a network with layers: **784 → 128 → 64 → 10**

### Forward Pass:
1. **Layer 0→1**: `z[1] = input × W[0] + b[0]`, then `a[1] = ReLU(z[1])`
2. **Layer 1→2**: `z[2] = a[1] × W[1] + b[1]`, then `a[2] = ReLU(z[2])`
3. **Layer 2→3**: `z[3] = a[2] × W[2] + b[2]`, then `a[3] = softmax(z[3])`

### Backward Pass:
1. **Output Layer (Layer 3)**: 
   - `δ[2] = a[3] - target_one_hot` (stored in `d_deltas_[2]`)

2. **Hidden Layer 2**:
   - `δ[1] = (W[2]^T × δ[2]) ⊙ ReLU'(z[2])` (stored in `d_deltas_[1]`)

3. **Hidden Layer 1**:
   - `δ[0] = (W[1]^T × δ[1]) ⊙ ReLU'(z[1])` (stored in `d_deltas_[0]`)

### Gradient Computation:
1. **For W[2]** (Layer 2→3):
   - `gradient = (1/batch) × a[2]^T × δ[2]`
   - Uses `d_activations_[2]` (post-ReLU) and `d_deltas_[2]`

2. **For W[1]** (Layer 1→2):
   - `gradient = (1/batch) × a[1]^T × δ[1]`
   - Uses `d_activations_[1]` (post-ReLU) and `d_deltas_[1]`

3. **For W[0]** (Layer 0→1):
   - `gradient = (1/batch) × input^T × δ[0]`
   - Uses `d_input_buffer_` (raw input) and `d_deltas_[0]`

### Weight Updates:
All weights and biases are updated simultaneously:
```
W[i] -= learning_rate × gradient_W[i]
b[i] -= learning_rate × gradient_b[i]
```

---

## Key Implementation Details

### Memory Layout:
- **Weights**: Stored as `(input_size × output_size)` in row-major format
  - `weights[input_idx * output_size + output_idx]` = weight from input neuron to output neuron
- **Activations**: Stored as `(batch_size × layer_size)` in row-major format
  - `activations[batch * layer_size + neuron]` = activation of neuron in layer for batch sample

### CUDA Parallelization:
- **Delta computation**: Each thread computes delta for one neuron in one batch sample
- **Gradient computation**: Each thread computes gradient for one weight or bias
- **Matrix operations**: Parallelized using 2D thread blocks

### Why Pre-Activations Are Stored:
Pre-activations (`d_pre_activations_`) are needed for the ReLU derivative:
- ReLU derivative = 1 if pre-activation > 0, else 0
- We need the **pre-activation value** (before ReLU) to determine if gradient should flow through
- Post-activation (after ReLU) would be 0 for negative values, losing information

---

## Mathematical Summary

For a layer with:
- Input: `a[l]` (activations from previous layer)
- Weights: `W[l]`
- Biases: `b[l]`
- Pre-activation: `z[l+1] = a[l] × W[l] + b[l]`
- Activation: `a[l+1] = f(z[l+1])` where `f` is ReLU or softmax

**Backpropagation formulas:**
1. **Delta for layer l+1**: `δ[l+1] = (W[l+1]^T × δ[l+2]) ⊙ f'(z[l+1])`
2. **Weight gradient**: `∂L/∂W[l] = (1/batch) × a[l]^T × δ[l+1]`
3. **Bias gradient**: `∂L/∂b[l] = (1/batch) × Σ_batch δ[l+1]`
4. **Update**: `W[l] -= lr × ∂L/∂W[l]`, `b[l] -= lr × ∂L/∂b[l]`

This implementation follows these formulas exactly, with CUDA kernels parallelizing the computations for efficiency.
