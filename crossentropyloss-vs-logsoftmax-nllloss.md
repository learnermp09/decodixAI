# CrossEntropyLoss vs LogSoftmax + NLLLoss (Mathematical Comparison)

Assume the neural network predicts 3 classes.

**True class = Dog (Class 1)**

**One-hot label**

$$
y =
\begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix}
$$

**Raw output (logits) from the network**

$$
z =
\begin{bmatrix}
0.9 \\
2.6 \\
0.1
\end{bmatrix}
$$

---

## Stage-by-Stage Comparison

| Stage | CrossEntropyLoss Approach | LogSoftmax + NLLLoss Approach |
|---|---|---|
| Output Layer | No activation | LogSoftmax(dim=1) |
| Network Output | Raw logits $z = [0.9, 2.6, 0.1]$ | Log probabilities $\log(p)$ |
| Meaning of Output | Arbitrary scores (not probabilities) | Logarithm of probabilities |
| Mathematical Form | $z_i = W_i x + b_i$ | $\log(p_i) = z_i - \log\left(\sum_j e^{z_j}\right)$ |
| Numerical Output | $[0.9, 2.6, 0.1]$ | $[-1.931, -0.234, -2.734]$ |
| Loss Function | `CrossEntropyLoss()` | `NLLLoss()` |
| What Loss Receives | Raw logits | Log probabilities |
| Internal Computation | LogSoftmax + NLLLoss | Only NLLLoss |
| Formula Used by Loss | $L = -\log\left(\dfrac{e^{z_y}}{\sum_j e^{z_j}}\right)$ | $L = -\log(p_y)$ |
| Calculation | $-\log(13.46 / 17.025)$ | $-(-0.234)$ |
| Loss Value | $0.234$ | $0.234$ |
| Output During Inference | Raw logits | Log probabilities |
| Convert to Probabilities | Softmax | Exponential |
| Formula | $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | $p_i = e^{\log(p_i)}$ |
| Recovered Probabilities | $[0.145, 0.791, 0.064]$ | $[0.145, 0.791, 0.064]$ |
| Prediction | $\arg\max(\text{logits}) = \text{Class } 1$ | $\arg\max(\text{log probabilities}) = \text{Class } 1$ |
| Probability of Prediction | Softmax → 79.1% | exp() → 79.1% |
| Recommended? | ✅ Standard PyTorch practice | Used mainly for learning or specialized models |

---

## What Happens Internally?

### Approach 1: CrossEntropyLoss

```
Input
   │
   ▼
Neural Network
   │
   ▼
Raw Logits
[0.9, 2.6, 0.1]
   │
   ▼
CrossEntropyLoss
   │
   ├── LogSoftmax
   │
   ├── NLLLoss
   │
   ▼
Loss = 0.234
```

Mathematically,

$$
z =
\begin{bmatrix}
0.9 \\
2.6 \\
0.1
\end{bmatrix}
\;\;\xrightarrow{\text{Softmax}}\;\;
\begin{bmatrix}
0.145 \\
0.791 \\
0.064
\end{bmatrix}
\;\;\xrightarrow{\text{Cross Entropy}}\;\;
-\sum_i y_i \log(p_i)
\;\;\longrightarrow\;\;
-\log(0.791) = 0.234
$$

### Approach 2: LogSoftmax + NLLLoss

```
Input
   │
   ▼
Neural Network
   │
   ▼
Raw Logits
[0.9, 2.6, 0.1]
   │
   ▼
LogSoftmax
   │
   ▼
Log Probabilities
[-1.931, -0.234, -2.734]
   │
   ▼
NLLLoss
   │
   ▼
Loss = 0.234
```

Mathematically,

$$
\text{Raw logits} =
\begin{bmatrix}
0.9 \\
2.6 \\
0.1
\end{bmatrix}
\;\;\xrightarrow{\text{LogSoftmax}}\;\;
\log\left(\frac{e^{z_i}}{\sum_j e^{z_j}}\right) =
\begin{bmatrix}
-1.931 \\
-0.234 \\
-2.734
\end{bmatrix}
\;\;\xrightarrow{\text{NLLLoss}}\;\;
L = -\log(p_y)
\;\;\longrightarrow\;\;
L = -(-0.234) = 0.234
$$

---

## The Big Picture

| Quantity | Raw Logits | Softmax Output | LogSoftmax Output |
|---|---|---|---|
| Class 0 | 0.9 | 0.145 | -1.931 |
| Class 1 | 2.6 | 0.791 | -0.234 |
| Class 2 | 0.1 | 0.064 | -2.734 |
| Sum | 3.6 (meaningless) | 1.0 | Not constrained |
| Interpretation | Scores | Probabilities | Log probabilities |

---

## A Simple Way to Remember

```
               SAME NEURAL NETWORK
                      │
                Raw Logits (z)
             [0.9, 2.6, 0.1]
               /             \
              /               \
     CrossEntropyLoss      LogSoftmax
      (does LogSoftmax)         │
              │                 │
              ▼                 ▼
      Computes Loss      Log Probabilities
              │                 │
              ▼                 ▼
           Loss=0.234       NLLLoss
                                  │
                                  ▼
                              Loss=0.234
```

---

## core concepts

Think of `CrossEntropyLoss` as a "2-in-1 package":

$$
\text{CrossEntropyLoss} = \text{LogSoftmax} + \text{NLLLoss}
$$

So you have two equivalent implementation choices:

- **Option 1 (recommended):** Network → Logits → CrossEntropyLoss
- **Option 2:** Network → LogSoftmax → Log Probabilities → NLLLoss

Both optimize the same objective, produce the same loss ($0.234$), and lead to the same trained model. The only difference is where the LogSoftmax computation takes place.
