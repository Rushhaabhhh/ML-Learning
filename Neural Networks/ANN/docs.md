# Deep Learning

- Deep Learning is a subfield of AI and ML inspired by the human brain, built on artificial neural networks with representation learning [ https://en.wikipedia.org/wiki/Feature_learning ].
- Deep Learning algorithms enables machines to draw human-like conclusions by continuously analyzing data through **Neural Networks**.
- Using multiple layers, it progressively extracts higher-level features from raw input for more accurate insights.

| Aspect | Machine Learning (ML) | Deep Learning (DL) |
| --- | --- | --- |
| Definition | Subset of AI that uses algorithms to learn from data and make predictions. | Subset of ML that uses multi-layered neural networks to learn complex patterns. |
| Data Requirement | Works well with smaller datasets. | Requires large amounts of data to perform effectively. |
| Feature Engineering | Manual feature extraction is often needed. | Automatically extracts features through neural networks. |
| Computation | Less computationally intensive. | Highly computationally intensive (requires GPUs/TPUs). |
| Interpretability | Models are more interpretable. | Models are often black-box and harder to interpret. |
| Training Time | Relatively shorter. | Longer due to deep architectures. |

https://docs.google.com/spreadsheets/d/e/2PACX-1vS9GnREhRJVofuqaIc9W_8XBP-er63xzhUE4P74Z3zp1TCybNdGuDddKtwu5-BQwG_2E69nwB9Yflq7/pubchart?oid=126587294&format=interactive

## **History of Deep Learning**

- **1943** – McCulloch & Pitts : First mathematical neuron model.
- **1958** – Rosenblatt’s ***Perceptron***, early neural network.
- **1969** – Minsky & Papert show Perceptron limits → *AI Winter*.
- **1986** – **Backpropagation** popularized (Rumelhart, Hinton, Williams).

[backprop_old.pdf](attachment:d004b14b-8bf8-4e45-8b6e-981217665250:backprop_old.pdf)

- **1990s** – ML (SVMs, trees) outperform NNs on small data.
- **2006** – Hinton coins *Deep Learning* ; revival with GPUs.
- **2012** – **AlexNet** wins ImageNet → Deep Learning breakthrough.
- **2017** – Transformers  (BERT, GPT, etc.) introduced and revolutionized NLP.
- **2020s** – Generative AI (GPT, Stable Diffusion, multimodal models).

---

---

# Perceptron

A perceptron is a type of **artificial neuron** and the simplest form of a feedforward neural network, used for **binary classification**. Perceptron was introduced by **Frank Rosenblatt** in 1958**.**

## **Components**

1. **Inputs $(x1,x2,...,xnx_1, x_2, ..., x_n)$** – Features of the data.
2. **Weights $(w1,w2,...,wnw_1, w_2, ..., w_n)$** – Parameters that scale input importance.
3. **Bias $(b)$** – Allows shifting the activation function, improving flexibility.
4. **Activation Function $(e)$** – Determines output :
    - Typically **step function** for binary classification:
        
        $y={ 
        \begin{cases} 
        1 & \text{if } \sum_i w_i x_i + b > 0 \\ 
        0 & \text{otherwise} 
        \end{cases}
        }$
        

### **Mathematical Formulation**

$$
⁍
$$

Where :

- $x_i$ = input features
- $w_i$ = weights
- $b$ = bias
- $f$ = activation function

---

## **Learning (Training)**

- **Goal:** Find weights ($w_i$) and bias ($b$) to correctly classify training data.
- **Algorithm:** Perceptron Learning Rule (Iterative):
    1. Initialize weights and bias (often to 0 or small random numbers).
    2. For each training example:
        - Predict output $y_{\text{pred}}.$
        - Update weights:
            
            $w_n= wi+η(y_{true}−y_{pred})xi$
            
            | y_true | y_pred | Error (y_true − y_pred) | Weight Update Direction |
            | --- | --- | --- | --- |
            | 1 | 0 | 1 | Increase w1 & w2 → line rotates **toward point** |
            | 0 | 1 | -1 | Decrease w1 & w2 → line rotates **away from point** |
            | 1 | 1 | 0 | No change → line stays |
            | 0 | 0 | 0 | No change → line stays |
        - Update bias:
            
            $b←b+η(y_{true}−y_{pred})$
            
    3. Repeat until convergence or max epochs.
- **Learning rate ($η$)**: Small constant controlling step size.
- https://karthikvedula.com/2024/01/05/visualizing-the-perceptron-learning-algorithm/

---

## Loss Functions

### 🔹 1. **Perceptron Loss**

**Definition** :

$$
L = \frac{1}{N} \sum_{i=1}^{N} \max \big(0, -y_i \cdot (w \cdot x_i + b)\big)
$$

$$
⁍
$$

- Here $y \in \{-1, +1\}$ , N = no of samples
- If **correctly classified**  $y\cdot (w \cdot x + b) > 0)$, loss = 0. If **misclassified**, loss grows linearly with distance from the correct side of the boundary.

**Key Points** :

- Designed for the original perceptron algorithm.
- Penalizes only misclassified points. Adjusts weights until all points are on the correct side of the hyperplane.
- Not smooth → not ideal for gradient descent.

### 🔹 2. **Hinge Loss** (Support Vector Machines)

**Definition**:

$$
L(y, \hat{y}) = \max(0, 1 - y \cdot (w \cdot x + b))
$$

- Encourages not just correct classification but also a **safety margin**.
- If a point lies correctly and beyond the margin$y \cdot (w \cdot x + b) \geq 1)$ , loss = 0. If it’s inside the margin or misclassified, it gets penalized.

**Key Points**:

- Drives the model to create a maximum margin hyperplane.
- Misclassified → large loss.
- Correct but too close to boundary → small loss.
- Correct & far from boundary → zero loss.
- Basis of **Support Vector Machines (SVMs)**.

### 🔹 3. **Binary Cross-Entropy Loss** (Log Loss)

**Definition**:

$$
L(y, \hat{y}) = - \Big[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \Big]
$$

where $\hat{y} = \sigma(w \cdot x + b)$ is the **sigmoid probability output**.

- Measures the distance between predicted probability and true label. If prediction is close to true class probability → low loss.
- If prediction is far from true probability → high loss.

**Key Points**:

- Provides **probabilistic interpretation**.
- Smooth and differentiable → ideal for gradient descent.
- Used in **logistic regression and neural networks** for binary classification.

### 🔹 4. **Categorical Cross-Entropy Loss** (Multi-class)

**Single-point** (one-hot encoded ):

$$

L(y, \hat{y}) = - \sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

- $y_c = 1$ if class c is the true class, else 0.
- $\hat{y}_c$  : predicted probability for class c from softmax.

$$
For ~N ~ points:
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
$$

$$

\hat{y}{i,c} = \frac{e^{z{i,c}}}{\sum_{k=1}^{C} e^{z_{i,k}}}
$$

where $z_{i,c} = w_c \cdot x_i + b_c$

**Code** :

```python
def categorical_cross_entropy(Y, Y_pred):
    eps = 1e-9
    Y_pred = np.clip(Y_pred, eps, 1 - eps)
    N = Y.shape[0]
    return -np.mean(np.sum(Y * np.log(Y_pred), axis=1))

# Example usage
Y = np.array([[1,0,0],[0,1,0],[0,0,1]])  # true one-hot labels
Y_pred = np.array([[0.9,0.05,0.05],[0.2,0.7,0.1],[0.1,0.2,0.7]])  # softmax probs
print("Categorical Cross-Entropy:", categorical_cross_entropy(Y, Y_pred))

```

- **Update Rule** using learning rate $\eta$

$$

w_c \; \leftarrow \; w_c - \eta \cdot \frac{\partial L}{\partial w_c}

$$

$$
b_c \; \leftarrow \; b_c - \eta \cdot \frac{\partial L}{\partial b_c}
$$

- **Key Intuitions :**
    - If predicted probability > true label (overconfident wrong class) → gradient positive → weight reduced.
    - If predicted probability < true label (underconfident correct class) → gradient negative → weight increased.
    - This makes CCE with softmax the **default choice** in modern neural networks for multi-class problems.

### 🔹 Quick Comparison

| Loss Function | Intuition | Zero Loss Condition | Strengths | Limitations |
| --- | --- | --- | --- | --- |
| Perceptron Loss | Updates only when misclassified | Correct classification | Simple, foundational | No margin, not smooth |
| Hinge Loss | Enforces correctness + margin | Correct & outside margin | Generalizes better, margin-based | Non-differentiable at kink |
| Binary Cross-Entropy | Matches predicted probability with true label (binary) | Prediction matches probability exactly | Probabilistic, smooth, widely used | Sensitive to outliers, requires sigmoid |
| Categorical Cross-Entropy (CCE) | Matches predicted probability distribution with true one-hot label (multi-class) | Predicted probability for true class = 1 | Standard for multi-class NN, works with softmax, smooth & differentiable | Sensitive to outliers, assumes one-hot (not label smoothing) |

---

### **Properties**

- **Binary classifier:** Only separates **linearly separable** data.
- **Linear decision boundary:** Forms a hyperplane in input space.
- **Limitations:** Cannot solve non-linear problems (e.g., XOR).

![](https://cdn.shopify.com/s/files/1/1905/9639/files/Perceptron_1024x1024.webp?v=1704706763)

### Research Paper

[Rosenblatt1958](https://www.notion.so/Rosenblatt1958-262003e596a681f5b7f1f220ef7fc500?pvs=21)

---

[https://playground.tensorflow.org](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa19kUDlhcGF0TjhSUUplZkIwVWZBVDNKbFNZQXxBQ3Jtc0ttVzI5b1ZJNmdtLXZSUFY0S0E3TWhRcGU1ajZ6QmNLVjdHQUJYVnVzaUhGaEFidXZhVUEwc1J2N3NhU2RtNWJReTZIdklOMFlJaTJEMnFuZXctZXFsaG1qa24wZ0JtU3JkNS1BMGI2MFFfT2I1ZlJITQ&q=https%3A%2F%2Fplayground.tensorflow.org%2F&v=Jp44b27VnOg)

# Multi-Layer Perceptron ( MLP )

A **feedforward neural network** composed of stacked layers :

- **Input layer** (no trainable params)
- One or more **hidden layers** (linear transform + nonlinearity)
- **Output layer** (task-dependent activation)

### Trainable Parameters

- In machine learning, trainable parameters are the internal values within a model that are learned and adjusted during the training process. These parameters are optimized by the learning algorithm to minimize the model's error and improve its performance in making predictions on new data.
- For each layer L :

$$

\text{params}(l) = n_{l-1}\cdot n_l \;(\text{weights}) + n_l \;(\text{biases})
$$

- Total :

$$

\text{params} = \sum_{l=1}^{L} \big(n_{l-1}n_l + b_l\big)
$$

![MLP-1.webp](attachment:18378caa-f326-4bef-8cf4-5c4aa06bdc82:MLP-1.webp)

### MLP Notations :

- **Weights :** $w_{ij}^{(k)}$ = weight from neuron $j$ in layer $(l-1)$ to neuron $i$ in layer $(l)$
- **Biases :** $b_i^{(l)}$ = bias of neuron $i$ in layer $(l)$
- **Outputs (activations) :** $O_i^{(l)}$ = output of neuron $i$ in layer $(l)$ after activation

$$
o_i^{(l)} = f\Big(\sum_{j=1}^{n_{l-1}} w_{ij}^{(l)} \cdot o_j^{(l-1)} + b_i^{(l)}\Big)

$$

![MLP-2.webp](attachment:fcecc6cf-4d7b-4c88-a206-182b0df19766:MLP-2.webp)

### MLP Layers :

![image.png](attachment:82a36cd7-8a4b-4911-85b1-60ac2326c05a:image.png)