1. Two probability mass functions are given as

$p(x)=[0.1,0.1,0.8]$ for $x=0,1,2$

$q(x)=[0.8,0.1,0.1]$ for $x=0,1,2$.
Find the $\mathrm{KL}$ divergence $K L[p \| q]$ rounded off to the nearest integer. (Hint: $K L[p \| q]=-\sum_{x} p(x) \log \left(\frac{q(x)}{p(x)}\right)$ )
Ans: 2\\
2. The model that classifies the two classes x and o  is:
#add plot here
Ans: $y=\sigma\left(x_{1}+x_{2}-3\right)$

3. The confusion matrix for a detection model for red roses is shown below. The precision of the model is
|  	| Estimated True 	| Estimated False 	|
|---	|---	|---	|
| Ground True 	| 20 	| 5 	|
| Ground False 	| 15 	| 30 	|

Ans: None of the above. It is $4 / 7$

4. Three data points are given as shown in the table, for which a linear model is built as
$\hat{y}=w_0+w_1 x$

The value of $\left[\begin{array}{l}w_0 \\ w_1\end{array}\right]$, such that the squared error $(y-\hat{y})^2$ is minimum, is
\begin{itemize}
    \item $\left[\begin{array}{lcc} 3 & 6\\ 6&14\end{array}\right]^{-1}\left[\begin{array}{l}14 \\ 36\end{array}\right]$
    \item $\left[\begin{array}{lcc}1 & 1 & 1 \\ 1 & 2 & 3\end{array}\right]\left[\begin{array}{l}1 \\ 4 \\ 9\end{array}\right]$
    \item $\left[\begin{array}{lcc}2 & 3 & 4 \\ 3 & 5 & 7 \\ 4 & 7 & 10\end{array}\right]^{-1}\left[\begin{array}{l}1 \\ 4 \\ 9\end{array}\right]$
    \item $\left[\begin{array}{lcc}1 & 1 & 1 \\ 1 & 2 & 3\end{array}\right]\left[\begin{array}{lcc}2 & 3 & 4 \\ 3 & 5 & 7 \\ 4 & 7 & 10\end{array}\right]^{-1}\left[\begin{array}{l}1 \\ 4 \\ 9\end{array}\right]$
\end{itemize}
Ans: $\left[\begin{array}{lcc}2 & 3 & 4 \\ 3 & 5 & 7 \\ 4 & 7 & 10\end{array}\right]^{-1}\left[\begin{array}{l}1 \\ 4 \\ 9\end{array}\right]$

5. This task needs high precision, even if recall may be compromised, during automation. Mark True or False.
\begin{itemize}
    \item legal punishment of life imprisonment
    \item scanning at an airport security check
    \item gold search on the sandy surface of a beach
    \item COVID test to recommend quarantine
    \item tumour detection for leg amputation
    \item credit card fraud detection
\end{itemize}

6. The following code uses a sequential model from the Keras library. Find the total number of weights to be trained, including the biases.\\
model.add(Dense (24, input-dim =100, activation='relu'))\\
model.add (Dense (2, activation='softmax'))
