# 0. Operator Dictionary

-   `L(x)`: Linear **WITH bias** $\rightarrow xW + b$

-   `L_hat(x)`: Linear **NO bias** $\rightarrow xW$

-   `LN(x)`: LayerNorm

-   `Sig(x)`: Sigmoid $\rightarrow \sigma(x)$

-   `Sftm(x, dim)`: Softmax

-   `Cat([x, y])`: Concatenate

-   `Emb(x)`: Embedding

-   `SinPE(x)`: Sinusoidal Positional Encoding

-   $\odot$: Element-wise multiplication

# 1. Feature Initialization + Atom Attention (3 Layers)

-   **Inputs**: Atom Feature ($F_A \in \mathbb{R}^{M \times d_0}$), Bond Feature ($F_b \in \mathbb{R}^{M \times M \times d_0}$), Token Mapping ($\tau (i) = u$)

-   **Outputs**: $S_i$, $Z_{ij}$

## 1.1 Embedding Initialization

$$
\begin{aligned} 

a^{(0)}_i &= \text{Emb}_{atom}(F_{A, i}) \\

p_{ij} &= \text{Emb}_{pos}(F_{B, ij}) \\ 

\end{aligned}

$$

## 1.2 *Token-to-Atom Broadcast (Only in AF3, not in Boltz)*

$$\begin{aligned} a_i = a_i + L(\text{LN}(s_{M(i)})) \end{aligned}$$

## 1.3 Atom Self-Attention

$$\begin{aligned} 

\tilde{a}_i = \text{LN}(a_i) &\Longrightarrow 

\left\{ \begin{aligned} q_i &= \hat{L}_q(\tilde{a}_i) \\ 

k_j &= \hat{L}_k(\tilde{a}_j) \\ v_j &= \hat{L}_v(\tilde{a}_j) \\ 

b_{ij} &= \hat{L}_b(bond_{ij}) \end{aligned} 

\right. \\ 

\\ &\Longrightarrow \begin{aligned} 

\left\{ 
\begin{aligned} w_{ij} &= \text{Sftm}\left(\frac{q_i \cdot k_j^T}{\sqrt{c}} + b_{ij} + \text{Mask}_{ij}  , \text{dim}=j\right) \\

o_{i} &= \sum_{j \in M^{-1}(I)} w_{ij} v_j \end{aligned} \right. \end{aligned} \\

&\Longrightarrow \mathbf{o^h_{i}} = \text{Concat}(o_i^1,\, o_i^2,\, ...,\, o_i^H) \; \in \mathbb{R}^{C} \; \text{for MHA } 

\\ &\Longrightarrow a'_i = a^{(l-1)}_i + L(o^h_{i}) 

\end{aligned}$$
$$
\begin{aligned} 
\text{where Mask}_{ij} = \left\{ 
\begin{aligned} 0 &\qquad \text{if } |\tau (i) - \tau(j)| \leq 2\\ 
-\infty & \qquad \text{otherwise} \end{aligned} \right.
\end{aligned}
$$


## 1.4 Atom Transition 


$$
\begin{aligned} 

\tilde{a}_i &= \text{LN}(a'_i)  \textbf{(Not in AF3/Boltz, but in Transformer)}\\ 

\\ &\Longrightarrow a^{(l)}_i = a'_i + L(\text{SiLU}(L(\mathbf{\tilde{a}_i/a'_i)})) \\

\end{aligned}

$$

## 1.5 Atom/Pair-to-Token Pooling

$$
\left\{
\begin{array}{l}
S_{i} = L\!\left(\text{LN}\!\left( \frac{1}{|A_{i}|} 
          \sum_{i \in \tau ^{-1}(I)} a_i \right)\right) \\

Z_{ij} = L\!\left(\text{LN}\!\left( \frac{1}{|A_i||A_{j}|} 
          \sum_{i \in \tau ^{-1}(I)} \sum_{j \in \tau ^{-1}(J)} p_{ij} \right)\right)
\end{array}
\right.
$$


## Questions

### How does the initial Embeddings/Featurization works?


|                | **Categorical Features**<br><br>$\hat{L}(OneHot(x))$/`nn.Embedding`                                                                                                                                                  | **Continuous Features**                                                                                                                                                                                                                           |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Atom Embedding | **- Atom Type:**  C, N, O, P<br><br> **- Molecular Type:** protein, DNA, RNA, Ligand<br><br> **- Cyclic / Modified:** Ring, Methylation (True/False)<br><br> **- Method Conditioning:** X-ray, NMR, Cryo-EM (One-hot) | **- Partial Charge :**(-0.35, 0.1)                                                                                                                                                                                                               |
| Pair Embedding | **- Bond Type:** single, double, aromatic, no bond (None)<br><br>**- Chain Flag / Relative Chain:**  check if  two nodes are on the same polymer chain<br><br>**- Symmetric Chain:** Homomultimer symmetry           | **• Relative Position (RelPos) : **  just relpos<br><br>  <br><br> **- Contact / Pocket Constraints:** Distance restrictions between two tokens, or several tokens with a small chain/molecule (Ligand), due to strict scientific methods done  |
|                |                                                                                                                                                                                                                      |                                                                                                                                                                                                                                                   |

- For Contact and Pocket Constraints, we will first use one-hot to label the relationship between the target tokens (Binder to Pocket? Pocket to Binder?) as **Categorical Features**. Then, Input the targer range (4-10A), then normalise and embed it with Fourier Embeddings as Continuous Features. 

- If is still don't works, we have another tough tactics as a "Reinsurer": Physics-based Steering during diffusion. When t is approaching 0 but the distance is still far apart, a gradient punishment will pull them together. 

- For Time-Dependent Contact Potential: 
$$E_{\text{contact}}(S_A, S_B)(\mathbf{x}) = \frac{\sum_{i \in S_A, j \in S_B} \exp\left(-\lambda_{\text{union}}^t \max(||\mathbf{x}_i - \mathbf{x}_j|| - r_{AB}, 0)\right) \max(||\mathbf{x}_i - \mathbf{x}_j|| - r_{AB}, 0)}{\sum_{i \in S_A, j \in S_B} \exp\left(-\lambda_{\text{union}}^t \max(||\mathbf{x}_i - \mathbf{x}_j|| - r_{AB}, 0)\right)}$$

- For Steric Clash Penalty: 

$$E_{\text{clash}}(\mathbf{x}) = \frac{1}{N_{\text{pairs}}} \sum_{i,j \text{ (no bond atom pair)}} \max\left( r_{\text{vdw}}^{(i)} + r_{\text{vdw}}^{(j)} - \tau - ||\mathbf{x}_i - \mathbf{x}_j||, 0 \right)^2$$
### Why use Mean Pooling instead of Cross-Attention for Atom-to-Token?

-   **Cross-Attention**: Requires computing an $O(N_{token} \times N_{atom})$ attention matrix. For large biomolecular complexes, tracking token-to-atom relationships parametrically introduces severe computational and memory bottlenecks.

-   **Mean Pooling**: Leverages the deterministic, known mapping $M^{-1}(I)$ between tokens (e.g., amino acids) and their constituent atoms. It efficiently compresses high-resolution local geometry back into the 1D token space ($s_I$) without needing to learn redundant alignment scores.

### Why use Bond features as an explicit Attention Bias ($b_{ij}$)?

-   **Without Bias**: The attention mechanism would have to implicitly deduce chemical connectivity and spatial constraints entirely from node embeddings.

-   **With Bias**: Injecting 2D topological priors (e.g., bond orders, aromaticity) directly into the softmax logits forces the attention matrix to prioritize valid local chemistry, preventing physical violations and stabilizing small-molecule pose predictions.

# 3. PairFormer Block (x64 Layers)


-   **Input**: $z_{ij}$

-   **Output**: Updated $z_{ij}$

## 3.1 Triangle Multiplicative Update (Outgoing)

  

$$

\begin{aligned}

\tilde{z}_{ij} = \text{LN}(z_{ij}) 

&\Longrightarrow 

\left\{

\begin{aligned}

L_{ij} &= \sigma_l(L(\tilde{z}_{ij})) \odot L(\tilde{z}_{ij}) \\

R_{ij} &= \sigma_r(L(\tilde{z}_{ij})) \odot L(\tilde{z}_{ij}) \\

g_{ij} &= \sigma(L(\tilde{z}_{ij}))

\end{aligned}

\right. \\

\\

&\Longrightarrow x_{ij} = \sum_k L_{ik} \odot R_{kj} \\

\\

&\Longrightarrow z_{ij} = z_{ij} + g_{ij} \odot L(\text{LN}(x_{ij}))

\end{aligned}

$$

## 3.2 Triangle Attention (Starting Node)

  

$$

\begin{aligned}

\tilde{z}_{ij} = \text{LN}(z_{ij}) 

&\Longrightarrow 

\left\{

\begin{aligned}

q_{ij} &= \hat{L}(\tilde{z}_{ij}) \\

k_{ij} &= \hat{L}(\tilde{z}_{ij}) \\

v_{ij} &= \hat{L}(\tilde{z}_{ij}) \\

b_{ij} &= \hat{L}(\tilde{z}_{ij}) \\

g_{ij} &= \text{Sig}(L(\tilde{z}_{ij}))

\end{aligned}

\right. \\

\\

&\Longrightarrow

\begin{aligned}

\left\{

\begin{aligned}

w_{ijk} &= \text{Sftm}\left(\frac{q_{ij} \cdot k_{ik}^T}{\sqrt{c}} + b_{jk},\, \text{dim}=k\right) \\

o_{ij} &= \sum_k w_{ijk} v_{ik}

\end{aligned}

\right. 

\end{aligned} \\

\\

&\Longrightarrow z_{ij} = z_{ij} + g_{ij} \odot L(o_{ij})

\end{aligned}

$$

## 3.3 Pair Transition

$$

\begin{aligned}

\tilde{z}_{ij} &= \text{LN}(z_{ij}) \\

\\

&\Longrightarrow z_{ij} = z_{ij} + L(\text{SiLU}(L(\tilde{z}_{ij})))

\end{aligned}

$$

## Questions

### Why Pre-LN, not Post-LN?


-   **Post-LN**: Places LayerNorm directly on the main residual path. This stabilizes forward activations but introduces a normalization bottleneck, causing gradients to diminish in deep networks.

-   **Pre-LN**: Applies LayerNorm to the branch inputs instead, preserving a clean addition-based residual highway. This ensures backward gradients flow smoothly while maintaining stability in forward computations.