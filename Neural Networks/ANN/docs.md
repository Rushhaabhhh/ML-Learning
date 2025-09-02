
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


# Perceptron

A perceptron is a type of **artificial neuron** and the simplest form of a feedforward neural network, used for **binary classification**. Perceptron was introduced by **Frank Rosenblatt** in 1958**.**

### **Components**

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
y = f\Big(\sum_{i=1}^{n} w_i x_i + b \Big)
$$

Where :

- $x_i$ = input features
- $w_i$ = weights
- $b$ = bias
- $f$ = activation function

---

### **Learning (Training)**

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

### **Properties**

- **Binary classifier:** Only separates **linearly separable** data.
- **Linear decision boundary:** Forms a hyperplane in input space.
- **Limitations:** Cannot solve non-linear problems (e.g., XOR).

![](https://cdn.shopify.com/s/files/1/1905/9639/files/Perceptron_1024x1024.webp?v=1704706763)

### Research Paper

[Rosenblatt1958](https://www.notion.so/Rosenblatt1958-262003e596a681f5b7f1f220ef7fc500?pvs=21)

---