以下是论文《Learning spatiotemporal dynamics with a pretrained generative model》中核心的“Methodology”部分的Markdown格式内容：

# Methodology

## Overview of the S3GM

In the pretraining stage, the proposed S3GM first approximates the joint distribution of \(T\) snapshots \(X = \{x_t\}_{t=1}^T \in \mathbb{R}^d\) via a score-based generative model (SGM) [53,55]. After pretraining, we further estimate the conditioned generative probability \(p(X|y)\) with the observation \(y = H(X) + \epsilon\), as shown in equation (1), for generating the desired contents under certain observations \(y\).

## Model Pretraining for Estimating the Joint Distribution

The SGM [53,55] is employed to learn the joint distribution \(p(X)\) from a discretized dataset containing instances of \(X\). As shown in previous work [53,55], SGM does not directly fit the data distribution \(p(X)\) itself but approximates the score of the distribution \(s(X)\), defined as the gradient of the log-likelihood with regard to the sample, given by

\[ s(X) = \nabla_X \ln p(X). \]

The score of the distribution can be tightly approached [53] by introducing different levels of Gaussian noise \(\sigma_\tau\), \(\tau \in [0, 1]\) into the following score-matching objective:

\[ J(\theta) = \mathbb{E}_{\tau} \left[ \mathbb{E}_{X \sim p(X)} \left[ \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I_d)} \left[ \left\| s_\theta(X_\tau, \tau) - \nabla_X \ln p_{\sigma_\tau}(X_\tau|X) \right\|_2^2 \right] \right] \right], \]

where \(\sigma_\tau > 0\) is the scheduled standard deviation, \(p_{\sigma_\tau}(X_\tau|X) = \mathcal{N}(X_\tau; X, \sigma_\tau^2 I_d)\) represents a series of Gaussian distributions with data perturbed by different levels of noise, and the score approximating neural network \(s_\theta(X_\tau, \tau)\) evaluates the scores at each perturbing stage \(\tau\). Given a training sample \(X \sim p(X)\), we perturb it at step \(\tau\) as \(X_\tau = X + \sigma_\tau \epsilon\), where \(\epsilon \sim \mathcal{N}(0, I_d)\) denotes a random variable sampled from a standard Gaussian distribution. Considering that \(p_{\sigma_\tau}(X_\tau|X) = \mathcal{N}(X_\tau; X, \sigma_\tau^2 I_d)\) is a Gaussian distribution, we can easily obtain its score \(\nabla_{X_\tau} \ln p_{\sigma_\tau}(X_\tau|X) = \epsilon / \sigma_\tau\) using the reparameterization trick [82]. Substituting the expression of \(X_\tau\) and \(\nabla_{X_\tau} \ln p_{\sigma_\tau}(X_\tau|X)\) into equation (4), the neural network \(s_\theta(X_\tau, \tau)\) can be well trained with a properly chosen architecture and adequate data supplies. Here, to better characterize the temporal dynamics, we choose an architecture called Video U-Net [83,84] to model the sequential correlations among snapshots.

## Sampling Long Sequences from Pretrained Score

We further consider the controllable generation to generate samples conditioned on the observation \(y\). We note that the direct sampling from the pretrained score can only yield \(T\) snapshots \(X = \{x_t\}_{t=1}^T\), while the observation may depend on a long sequence \(y = H(\{x_t\}_{t=1}^{T'}) + \epsilon\) with \(T' \gg T\) as the total number of snapshots. To generate predictions over longer sequences, we divide the sequence into two parts by distinguishing them as observation-dependent or independent subsequences (Extended Data Fig. 2a).

For the subsequence depending on the observation \(y\), we first initialize \(B\) consecutive samples \(X^{(i)}_{\tau=1} = \{x^{(i)}_{t,\tau=1}\}_{t=(i-1)\cdot(T-m)+1}^{i\cdot(T-m)+m}\), where \(m\) is the number of snapshots overlapping with the previous sample and \(B\) the total number of decomposed pieces. Here, we assume \(\sigma_{\tau=1}\) is sufficiently large, and then obtain \(X^{(i)}_{\tau=1} \approx \sigma_{\tau=1} \epsilon\). Inspired by the conditional generation model [57], we start inferring each denoised sample \(X^{(i)}_{\tau=0}\) by solving the SDE as follows:

\[ \begin{bmatrix} X^{(1)}_\tau \\ X^{(2)}_\tau \\ \vdots \\ X^{(B)}_\tau \end{bmatrix} = -\frac{1}{2} \frac{d[\sigma^2_\tau]}{d\tau} \cdot \begin{bmatrix} s_\theta(X^{(1)}_\tau, \tau) \\ s_\theta(X^{(2)}_\tau, \tau) \\ \vdots \\ s_\theta(X^{(B)}_\tau, \tau) \end{bmatrix} d\tau + \sqrt{\frac{d[\sigma^2_\tau]}{d\tau}} d \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_B \end{bmatrix} - \alpha_\tau \nabla_{X^{(1)},\ldots,X^{(B)}} \left\| y - H(\hat{X}_\tau) \right\|^2_2 d\tau - \beta_\tau \nabla_{X^{(i)}} \left( \sum_{t=i\cdot(T-m)+1}^{(i+1)\cdot(T-m)} \left\| x^{(i+1)}_{t} - s_g(\hat{x}^{(i)}_{t}) \right\|^2_2 \right) d\tau, \]

where \(s_g(\cdot)\) is the stop-gradient operation, \(w_1, \ldots, w_B\) are \(B\) independent standard Wiener processes and \(X_\tau\) is ensembled by \(X^{(1)}, \ldots, X^{(B)}\). In equation (5), the last two terms respectively restrict the generated samples to agree with the observation \(y\) (termed as observation consistency in Fig. 1d) and to share similar contents over the overlapped snapshots (termed as sequence consistency in Fig. 1d). These two penalties were balanced by two hyperparameters \(\alpha_\tau\) and \(\beta_\tau\).

For the subsequence not depending on the observation \(y\), following the denoising SDE, we first initialize a sample \(X_{\tau=1} = \{x_{t,\tau=1}\}_{t=1}^T \sim \mathcal{N}(0, \sigma^2_{\tau=1} I_d)\) as the initial condition for solving the denoising SDE as follows:

\[ dX_\tau = -\frac{1}{2} \frac{d[\sigma^2_\tau]}{d\tau} \cdot s_\theta(X_\tau, \tau) d\tau + \sqrt{\frac{d[\sigma^2_\tau]}{d\tau}} dw - \gamma_\tau \nabla_{X_\tau} \left( \sum_{t=1}^{T_{\text{init}}} \left\| \hat{x}_t - x_t \right\|^2_2 \right) d\tau, \]

where \(\hat{x}_t \in \{\hat{x}_t\}_{t=1}^{T_{\text{init}}} = X_{\tau=0}\) and \(\{x_t\}_{t=1}^{T_{\text{init}}}\) are the given initial snapshots. The last term on the right-hand side of the above SDE constrains the generated samples to satisfy the given initial snapshots, and \(\gamma_\tau\) is the hyperparameter to balance the effect of this term.

By solving the SDE in equation (6) from \(\tau = 1\) to \(\tau = 0\), we can acquire a sample containing the prediction of the subsequent snapshots. In this way, a longer sequence can be generated by iteratively applying such a process in an autoregressive manner.

Although we can change the time step size \(\Delta t\) during the inference (generating) stage, the model performs best using a consistent time step size used for training and testing.

以上是论文Methodology部分的核心内容，涵盖了S3GM的两个阶段（预训练和生成）以及从预训练的分数中采样长序列的方法。这些内容展示了S3GM如何通过自监督学习捕获动态系统的先验知识，并在给定稀疏测量数据的情况下，利用条件采样过程有效重建完整的时空动态场。