# Neural networks

1. A dense neural network takes a 5x1 vector as input. It has a hidden layer with 10 neurons. Then, the final layer outputs a scalar. What is the total number of learnable parameters (weights and biases)?
   - Ans: 71

2. A non-linear model $\hat{y} = \sigma(w_0 + w_1x)$
 is to be optimized using the gradient descent algorithm over the loss term $$\mathit{L} = (y - \hat{y})^2$$
. Here, $
\sigma(\cdot)
$
 is the sigmoid function. The weights are chosen to be $ w_0 = \log_e(2)
  $and $ w_1 = \log_e(0.5) $ . The learning rate $\eta = 1 $. Find the updated value of $w_0$ after iterating once over the training data $(x = 2, y = 2/3)$.
   - Ans: $\log_e(2) + \frac{2}{27}$

3. $\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$. Derivative of $\tanh(x)$ with respect to $x$ at $x = \log_e 2$?
   - Ans: 0.64

4. A network is to be trained to detect a rose from input-target pairs (x, t). The final layer has one neuron with output $h$ and the final estimate $P(\text{Rose} | x; \theta) = \sigma(h)$. To train the network, binary cross-entropy loss function is used:
   $
   \mathcal{L}(\theta) = -t \ln P(\text{Rose} | x; \theta) - (1 - t) \ln P(\text{not Rose} | x; \theta)
   $
   - Ans: $(\sigma(h) - t) \frac{\partial h}{\partial \theta}$

5. $y = \text{ReLU}\left(\sum_{i=0}^{1} v_i d_i\right)$ where $d_i$ is a dropout layer with a dropout probability of 0.2. If $v_i$ = [-0.4, 0.8] for $i$ = [0, 1], the expected value $E\left[ X \right]$.
   - Ans: 0.38

6. For multi-label classification, a neural network has 3 inputs. There is no hidden layer. The output layer has 2 neurons.
   - What should be the non-linearity in the output layer?
   - What loss function should be used to train the network?
   - Derive the update rule for all the weights.
   - Ans:
     - Softmax
     - Categorical cross entropy
     - $w_{i_1i_0} \leftarrow w_{i_1i_0} - \eta(y_{i_1} - t_{i_1})x_{i_0}$

7. For a multi-class classification problem, a neural network is given as follows:
   \[
   \begin{align*}
   & h_{i_1} = \sum_{i_0=1}^{2} w_{i_0i_1} x_{i_0} \\
   & v_{i_1} = \sigma(h_{i_1}) \\
   & h_{i_1}' = \sum_{i_1=1}^{3} w_{i_0i_1}' v_{i_1} \\
   & y_{i_2} = \text{softmax}(h_{i_2}')
   \end{align*}
   \]
   Use categorical cross-entropy loss to compute the following:
   - Write the loss function \(E\) in terms of the variables defined above and the targets \(t_{i_2}\).
   - $\frac{\partial E}{\partial y_1}$ (i.e., $i_2$ = 1).
   - $\frac{\partial E}{\partial w_{11}'}$ (i.e., $i_1$ = $i_2$ = 1).
   - $\frac{\partial E}{\partial w_{11}}$ (i.e., $i_0$ = $i_1$ = 1).
   - Ans:
     1. $E = -\sum_{i_2} t_{i_2} \log y_{i_2}$
     2. $-\frac{t_1}{y_1}$
     3. $-\sum_{i_2} \left(\frac{t_{i_2}}{y_{i_2}}\right) (\delta_{i_2i_2'} - y_{i_2}) y_{i_2'} x_{i_1'}$
     4. $- \sum_{i_2'} \sum_{i_2} \left(\frac{t_{i_2}}{y_{i_2}}\right) (\delta_{i_2i_2'} - y_{i_2}) y_{i_2'} x_{i_1'} w_{i_1'i_2'} v_{i_1'} (1 - v_{i_1'}) x_{i_0'}$
