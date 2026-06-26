# 0. Operator Dictionary

-   `L(x)`: Linear **WITH bias** <img src="https://githubusercontent.com" valign="middle">

-   `L_hat(x)`: Linear **NO bias** <img src="https://githubusercontent.com" valign="middle">

-   `LN(x)`: LayerNorm

-   `Sig(x)`: Sigmoid <img src="https://githubusercontent.com" valign="middle">

-   `Sftm(x, dim)`: Softmax

-   `Cat([x, y])`: Concatenate

-   `Emb(x)`: Embedding

-   `SinPE(x)`: Sinusoidal Positional Encoding

-   <img src="https://githubusercontent.com" valign="middle">: Element-wise multiplication

# 1. Feature Initialization + Atom Attention (3 Layers)

-   **Inputs**: Atom Feature (<img src="https://githubusercontent.com" valign="middle">), Bond Feature (<img src="https://githubusercontent.com" valign="middle">), Token Mapping (<img src="https://githubusercontent.com" valign="middle">)

-   **Outputs**: <img src="https://githubusercontent.com" valign="middle">, <img src="https://githubusercontent.com" valign="middle">

## 1.1 Embedding Initialization

$$
\begin{aligned} 

a^{(0)}_i &= \text{Emb}_{atom}(F_{A, i}) \\

p_{ij} &= \text{Emb}_{pos}(F_{B, ij}) \\ 

\end{aligned}

$$

## 1.2 *Token-to-Atom Broadcast (Only in AF3, Boltz only broadcast once)*

<br><img src="https://githubusercontent.com"><br>

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

o_{i} &= \sum_{j } w_{ij} v_j \end{aligned} \right. \end{aligned} \\

&\Longrightarrow \mathbf{o^h_{i}} = \text{Concat}(o_i^1,\, o_i^2,\, ...,\, o_i^H) \; \in \mathbb{R}^{C} \; \text{for MHA } 

\\ &\Longrightarrow a'_i = a^{(l-1)}_i + L(o^h_{i}) 

\end{aligned}$$
$$
\begin{aligned} 
\text{where Mask}_{ij} = \left\{ 
\begin{aligned} 0 &\qquad \text{if } |\tau (i) - \tau(j)| \leq 1\\ 
-\infty & \qquad \text{otherwise} \end{aligned} \right.
\end{aligned}
$$


## 1.4 Atom Transition 


$$
\begin{aligned} 

\tilde{a}_i &= \text{LN}(a'_i) \\ 

\\ &\Longrightarrow a^{(l)}_i = a'_i + L(\text{SiLU}(L(\mathbf{\tilde{a}_i)})) \\

\end{aligned}

$$
(Note: Basically, all FFN in AF3 and Boltz use SwiGLU, where <img src="https://githubusercontent.com" valign="middle">)

## 1.5 Atom/Pair-to-Token Pooling

$$
\left\{
\begin{array}{l}
S_{i} = L\!\left( \frac{1}{|A_{i}|} 
          \sum_{i \in \tau ^{-1}(I)} LN(a_i) \right) \\

\underset{ \text{not in AF3 and Boltz, due to } O(n^2) \text{, but in RFAA/Hierarchical GNNs etc}}{ \cancel{ Z_{ij} = L\!\left(\text{LN}\!\left( \frac{1}{|A_i||A_{j}|} 
          \sum_{i \in \tau ^{-1}(I)} \sum_{j \in \tau ^{-1}(J)} p_{ij} \right)\right) } }
\end{array}
\right.
$$


## Questions

### How does the initial Embeddings/Featurization works?


|                | **Categorical Features**<br><br><img src="https://githubusercontent.com" valign="middle">/`nn.Embedding`                                                                                                                                                                                                                                                    | **Continuous Features**                                                                                                                                                         |     |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| Atom Embedding | **• Atom Type: **C, N, O, P<br><br>**• Molecular Type: ** protein, DNA, RNA, Ligand<br><br>**• Cyclic / Modified: ** Ring, Methylation (True/False)<br><br>**• Method Conditioning: ** X-ray, NMR, Cryo-EM (One-hot)                                                                                                   | **• Partial Charge : ** (-0.35, 0.1)                                                                                                                                            |     |
| Pair Embedding | **• Bond Type: **single, double, aromatic, no bond (None)<br><br>**• Chain Flag / Relative Chain: ** check if  two nodes are on the same polymer chain<br><br>**• Symmetric Chain：** Homomultimer symmetry<br><br>**• Relative Position (RelPos) : ** It seems to be continuous, but it should be clipped (\[-32, 32]) | **• Contact / Pocket Constraints: ** Distance restrictions between two tokens, or several tokens with a small chain/molecule (Ligand), due to strict scientific methods done ne |     |
|                |                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                 |     |

- For Contact and Pocket Constraints, we will first use one-hot to label the relationship between the target tokens (Binder to Pocket? Pocket to Binder?) as **Categorical Features**. Then, Input the targer range (4-10A), then normalise and embed it with Fourier Embeddings as Continuous Features. 

- If is still don't works, we have another tough tactics as a "Reinsurer": Physics-based Steering during diffusion. When t is approaching 0 but the distance is still far apart, a gradient punishment will pull them together. 

	- For Time-Dependent Contact Potential: 
<br><img src="https://githubusercontent.com"><br>
	- For Steric Clash Penalty: 
<br><img src="https://githubusercontent.com"><br>
### Why use Mean Pooling instead of Cross-Attention for Atom-to-Token?

-   **Cross-Attention**: Requires computing an <img src="https://githubusercontent.com" valign="middle"> attention matrix. For large biomolecular complexes, tracking token-to-atom relationships parametrically introduces severe computational and memory bottlenecks.

-   **Mean Pooling**: Leverages the deterministic, known mapping <img src="https://githubusercontent.com" valign="middle"> between tokens (e.g., amino acids) and their constituent atoms. It efficiently compresses high-resolution local geometry back into the 1D token space (<img src="https://githubusercontent.com" valign="middle">) without needing to learn redundant alignment scores.

### Why use Bond features as an explicit Attention Bias (<img src="https://githubusercontent.com" valign="middle">)?

-   **Without Bias**: The attention mechanism would have to implicitly deduce chemical connectivity and spatial constraints entirely from node embeddings.

-   **With Bias**: Injecting 2D topological priors (e.g., bond orders, aromaticity) directly into the softmax logits forces the attention matrix to prioritize valid local chemistry, preventing physical violations and stabilizing small-molecule pose predictions.


# 2. Template Module

- **Input**: Template features <img src="https://githubusercontent.com" valign="middle"> (Coordinates, Sequence, Alignment Mask <img src="https://githubusercontent.com" valign="middle">  up to <img src="https://githubusercontent.com" valign="middle">),  <img src="https://githubusercontent.com" valign="middle">
    
- **Output**: Updated  <img src="https://githubusercontent.com" valign="middle">

## 2.1 Template Featurization (inculding Missing Embeddings Generation)

$$
\begin{aligned}
&
\left\{
\begin{array}{l}
\text{Distance: } D^{k}_{ij} = \lVert X^k_{i} -X^k_{j}\rVert_{2}\\

\text{Local Unit Vector: } U^{m}_{ij} = \frac{(R^{m}_{i})^{-1}}{D^m_{{ij}}} (X^k_{j} -X^k_{i}) \\ 

\text{Dihedral Angle + Atomic Feature: } A^m_{i}
\end{array}
\right. \\
\\

\Longrightarrow & T^{(0)'}_{ij} =  l(\text{Onehot}(\text{Concat}(Bin(D^{k}_{ij}, \text{Bins} = N), U^{m}_{ij}, A^m_{i}, A^m_{j}, \text{Amino Type}_{fasta})))
\end{aligned}

$$


## 2.2 Template Pair Stack (x2 Layers)

<br><img src="https://githubusercontent.com"><br>

## 2.3 Cross-Template Aggregation

<br><img src="https://githubusercontent.com"><br>

## 2.4 Residual Update to Trunk

<br><img src="https://githubusercontent.com"><br>

## Questions

### Where do "Missing Embeddings" come from if there isn't a dedicated lookup table for them?

- **Origin**: They emerge naturally from the `Linear` layer's bias (<img src="https://githubusercontent.com" valign="middle">) and the amino acid type embedding. When a residue lacks 3D coordinates in the PDB file, its geometric raw features (distances, angles, valid masks) are initialized to arrays of `0`. When this zero-filled vector is multiplied by the weight matrix <img src="https://githubusercontent.com" valign="middle">, it zeros out, leaving only the AA type embedding and the linear bias <img src="https://githubusercontent.com" valign="middle">. This resulting non-zero high-dimensional vector serves as the distinct "Missing Embedding," signaling to the model: _"I am an Alanine, but my spatial location is unknown."_
    

### Why is the `AlignmentMask` (<img src="https://githubusercontent.com" valign="middle">) used in Cross-Aggregation but NOT in the Template Pair Stack?

- **Template Pair Stack (Phase 4.2)**: Uses a basic sequence padding mask. The bias is `0` for missing coordinates because the primary goal of this phase is **Imputation**. The missing tokens must be allowed to "see" the resolved surrounding scaffolds to logically deduce and self-correct their own missing geometric features via Triangle Attention.
    
- **Cross-Aggregation (Phase 4.3)**: Uses the `AlignmentMask`. If a coordinate is missing/imputed, the mask is set to <img src="https://githubusercontent.com" valign="middle">. Because the model is now transferring data back into the main trunk (<img src="https://githubusercontent.com" valign="middle">), it acts as a strict filter. The trunk is only allowed to extract geometric priors from regions backed by actual experimental data, explicitly rejecting the "hallucinated" coordinates generated in Phase 4.2.
    

### What is the difference of Alignment Mask between AF3 and Boltz2 ?

- **AlphaFold 3**: Forces the `AlignmentMask` for all cross-chain regions (off-diagonal) to <img src="https://githubusercontent.com" valign="middle"> (or <img src="https://githubusercontent.com" valign="middle"> in log space), restricting the module to monomeric templates only.
    
- **Boltz-2**: Dynamically checks the provenance of the templates. If Chain A and Chain B originate from the exact same PDB ID, Boltz-2 unlocks the cross-chain `AlignmentMask` (sets it to <img src="https://githubusercontent.com" valign="middle">). This allows the templates to be mutually visible to one another across chains, natively unlocking Multimeric Templating capabilities for complex assemblies.

### What is the <img src="https://githubusercontent.com" valign="middle">?

- This is the *Gram-Schmidt process*, which outputs an *Orthogonal Matrix*  representing the amino acid (<img src="https://githubusercontent.com" valign="middle">). Therefore, <img src="https://githubusercontent.com" valign="middle"> reverse the responding vector from global to local. 


# 3. MSA Module

- **Input**: MSA representation <img src="https://githubusercontent.com" valign="middle">,   <img src="https://githubusercontent.com" valign="middle">
    
- **Output**: Updated  <img src="https://githubusercontent.com" valign="middle">
    
- **Structure**: 4 identical MSA Blocks <img src="https://githubusercontent.com" valign="middle"> 1 Outer Product Mean
    

## 3.1 MSA Block (x4 Layers)

### 3.1.1 MSA Row Attention 

 <img src="https://githubusercontent.com" valign="middle"> is mapped as Bias, which act as attention bonus

<br><img src="https://githubusercontent.com"><br>

### 3.1.2 ~~MSA Column Attention (Only in AF12, not in Boltz and AF3, covariance)~~

 <img src="https://githubusercontent.com" valign="middle"> and <img src="https://githubusercontent.com" valign="middle"> exchange information, so no pair bias

<br><img src="https://githubusercontent.com"><br>

### 3.1.3 MSA Transition (FFN)

<br><img src="https://githubusercontent.com"><br>

(Note: Basically, all FFN in AF3 and Boltz use SwiGLU, where <img src="https://githubusercontent.com" valign="middle">)

## 3.2 Outer Product Mean (OPM)

Turning the <img src="https://githubusercontent.com" valign="middle"> sequence feature (<img src="https://githubusercontent.com" valign="middle"> to  <img src="https://githubusercontent.com" valign="middle"> token pair feature (<img src="https://githubusercontent.com" valign="middle">), and send it back the the main trunk

<br><img src="https://githubusercontent.com"><br>


## Questions

### Why there is no Column Attention in MSA, but in PairFormer?

- In Pair Former, the tensor updated is <img src="https://githubusercontent.com" valign="middle">, which is symmetric, so it is essential to ensure it is an undirected graph. 
- In MSA, the tensor updated is <img src="https://githubusercontent.com" valign="middle">, where it is column attention is doing on different on sequence. Not to mention the fact that different species have different length so different properties and functions on the corresponding token, MSA module do not have ligands, ions, and non-classic nucleic acid evolution data, so the computing resources will focus on the deoising module. 

### Why no divide by N_seq in opm?

- You cannot divide the opm by the total tensor size (<img src="https://githubusercontent.com" valign="middle">) because it includes padding zeros, which severely dilutes valid evolutionary signals and causes vanishing gradients. You must divide only by the count of _valid_ sequences (using a mask) to compute the true, mathematically accurate mean.


# 4. PairFormer Block (x48 Layers)

-   **Input**: <img src="https://githubusercontent.com" valign="middle">

-   **Output**: Updated <img src="https://githubusercontent.com" valign="middle">

## 4.1 Triangle Multiplicative Update (Outgoing)

  

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

&\Longrightarrow x_{ij} = \sum_k L_{ik} \odot R_{jk} \\

\\

&\Longrightarrow z_{ij} = z_{ij} + g_{ij} \odot L(\text{LN}(x_{ij}))

\end{aligned}

$$

## 4.2 Triangle Attention (Starting Node)

  

$$

\begin{aligned}

\tilde{z}_{ij} = \text{LN}(z_{ij}) 

&\Longrightarrow 

\left\{

\begin{aligned}

q_{ij} &= \hat{L}(\tilde{z}_{ij}) \\

k_{ik} &= \hat{L}(\tilde{z}_{ik}) \\

v_{ik} &= \hat{L}(\tilde{z}_{ik}) \\

b_{jk} &= \hat{L}(\tilde{z}_{jk}) \\

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

## 4.3 Incoming TriMul and Ending TriAtt

**Note:** In practice (e.g., Boltz codebase), the Incoming and Ending operations are not implemented as separate modules. Due to the geometric symmetry of the pair representation <img src="https://githubusercontent.com" valign="middle">, reversing the edge direction (<img src="https://githubusercontent.com" valign="middle">) is mathematically equivalent to transposing the pair matrix.

Instead of redundant formulas, the network achieves **Incoming** and **Ending Node** updates by applying the exact same **Outgoing / Starting** modules to the transposed pair tensor, and then transposing the result back:

<br><img src="https://githubusercontent.com"><br>

## 4.4 Pair Transition

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

### What is <img src="https://githubusercontent.com" valign="middle"> ?

- <img src="https://githubusercontent.com" valign="middle"> is a scalar, which represents the 3rd edge bias (for Multi-head attention, each head will have their responding <img src="https://githubusercontent.com" valign="middle">). If <img src="https://githubusercontent.com" valign="middle"> is really far away, <img src="https://githubusercontent.com" valign="middle">, so <img src="https://githubusercontent.com" valign="middle">

### Why do we need both Outgoing/Starting and Incoming/Ending split in Triangle blocks?

- In 3D space, the pair matrix must be symmetric (<img src="https://githubusercontent.com" valign="middle">). If we only rely on outgoing (<img src="https://githubusercontent.com" valign="middle">), it will lead to a **Directed Graph**. This destroy the apriority of symmetry. Therefore, this ensure the features are spreaded in **Undirected Graph**. 
  
- By instinct, the Starting Node is doing attention on **ROW**, where the ending node is doing attention on **COLUMN**. Note that the heads have different weights. 

# 5. Denoising Module (Atom Transformer Block)

**Input:** Noisy atom coordinates <img src="https://githubusercontent.com" valign="middle">, Timestep <img src="https://githubusercontent.com" valign="middle">, Static pair feature <img src="https://githubusercontent.com" valign="middle">, Single token <img src="https://githubusercontent.com" valign="middle">.

**Output:** Updated atom features <img src="https://githubusercontent.com" valign="middle">, Predicted denoised coordinates <img src="https://githubusercontent.com" valign="middle">.

## 5.0 Geometric Pre-processing & Feature Fusion

Before any attention mechanism, the network must guarantee SE(3) invariance and fuse 1D, 2D, and 3D representations.



$$
\begin{aligned}
&\text{(1) Center of Mass (CoM) Alignment (Zero-mean centering)} \\
&\quad \mathbf{X}_{\text{CoM}} = \frac{\sum_{i=1}^N \mathbf{X}_{t}^{(i)} \cdot \mathbf{\text{Mask}_{\text{valid}}^{(i)}}}{\sum_{i=1}^N \mathbf{\text{Mask}_{\text{valid}}^{(i)}}} \\
&\quad \Longrightarrow \mathbf{X}_t \leftarrow \mathbf{X}_t - \mathbf{X}_{\text{CoM}} \\ \\

&\text{(2) Noisy Distance and RBF Positional Encoding} \\
&\quad d_{ij} = \| \mathbf{X}_t^{(i)} - \mathbf{X}_t^{(j)} \|_2 \\
&\quad \Longrightarrow \mathbf{E}_{\text{RBF}} = \exp \left( - \frac{(d_{ij} - c_k)^2}{2\sigma_k^2} \right) \quad 
 k \in \{1, 2, \dots, K\} \\ \\

&\text{(3) The Grand Feature Fusion} \\
&\quad a_{ia}^{(0)} = L(s_i) + \text{AtomTypeEmb}(a) 
\quad \text{(Initialize atom tokens)} \\
&\quad \tilde{z}_{ia, jb} = z_{\tau(ia), \tau(jb)} + L(\mathbf{E}_{\text{RBF}}(d_{ia, jb})) 
\quad \text{(Fuse static prior with noisy dynamic geometry)}
\end{aligned}
$$

## 5.1 Timestep Embedding & AdaLN (Adaptive Layer Normalization)

Injecting the timestep <img src="https://githubusercontent.com" valign="middle"> dynamically to scale and shift features.

<br><img src="https://githubusercontent.com"><br>

## 5.2 Conditioned Atom Attention

Atoms attend to other atoms, strictly guided by the fused pair/distance representation <img src="https://githubusercontent.com" valign="middle"> as an attention bias.

<br><img src="https://githubusercontent.com"><br>

## 5.3 Atom Transition & Coordinate Update

Updating atom features and projecting them into actual 3D displacement vectors.

<br><img src="https://githubusercontent.com"><br>

## Questions

### **Why CoM (Center of Mass) Alignment?**

To strictly preserve translational invariance. If we don't zero-mean the coordinates, the network might waste capacity learning meaningless global translations. Every update (<img src="https://githubusercontent.com" valign="middle">) must also be re-centered so the molecule doesn't artificially drift through space during the diffusion process.

### **Why use RBF Positional Encoding instead of Absolute Coordinates?**

Absolute coordinates (<img src="https://githubusercontent.com" valign="middle">) destroy SE(3) invariance (rotational and translational invariance). By converting coordinates into pairwise distances <img src="https://githubusercontent.com" valign="middle">, we achieve invariance. However, a single scalar distance is too weak for neural networks. RBF (Radial Basis Function) artificially expands this 1D distance into a smooth, high-dimensional continuous vector space, allowing the network to have smooth gradients as atoms move closer or further apart.

### **Why AdaLN instead of Concatenation for Timestep <img src="https://githubusercontent.com" valign="middle">?**

- **Concatenation:** Forcing <img src="https://githubusercontent.com" valign="middle"> onto the feature dimension requires the network to constantly learn how to extract it, causing signal decay in deep layers.
    
- **AdaLN:** By using <img src="https://githubusercontent.com" valign="middle"> (scale) and <img src="https://githubusercontent.com" valign="middle"> (shift), <img src="https://githubusercontent.com" valign="middle"> directly modulates the entire activation distribution. When <img src="https://githubusercontent.com" valign="middle"> is large (high noise), AdaLN shifts the network to focus on global structural features; when <img src="https://githubusercontent.com" valign="middle"> is small (low noise), it shifts to refining local chemical bonds. Setting initial weights of <img src="https://githubusercontent.com" valign="middle"> to zero ensures stable identity mapping early in training.
    

### **Why condition <img src="https://githubusercontent.com" valign="middle"> with RBF to get <img src="https://githubusercontent.com" valign="middle">? (The core logic of Diffusion)**

This is where the magic happens. <img src="https://githubusercontent.com" valign="middle"> represents the _perfect static prior_ (where atoms _should_ be). The RBF encoding represents the _current noisy geometry_ (where atoms _currently_ are). By fusing them (<img src="https://githubusercontent.com" valign="middle">), the attention bias (<img src="https://githubusercontent.com" valign="middle">) inherently calculates the geometric tension between the "ideal state" and the "current broken state". This tension is the exact mathematical driving force that tells the network how to predict <img src="https://githubusercontent.com" valign="middle">.

### **Why predict denoised coordinates <img src="https://githubusercontent.com" valign="middle"> instead of predicting noise <img src="https://githubusercontent.com" valign="middle">?**

Standard image DDPMs predict noise <img src="https://githubusercontent.com" valign="middle">. But in 3D molecules, predicting a raw noise matrix doesn't allow us to calculate physical constraints. By directly predicting the absolute denoised structure <img src="https://githubusercontent.com" valign="middle">, we can feed this output directly into FAPE (Frame Aligned Point Error) to calculate a physically meaningful loss based on steric clashes, bond lengths, and absolute chiralities.

# 6. Inference-Time Steering (Boltz-Steering )

**Input:** <img src="https://githubusercontent.com" valign="middle">, <img src="https://githubusercontent.com" valign="middle"> 

**Output:** <img src="https://githubusercontent.com" valign="middle"> 

## 6.1 Time-Dependent Clash Energy

$$
\begin{aligned}
&c_{ij} = r_{\text{vdw}}^{(i)} + r_{\text{vdw}}^{(j)} - \tau \\ &E_{\text{clash}}(\hat{\mathbf{X}}_0) = \sum_{i<j, \, \text{topological distance } >3} \max \left(0, \, c_{ij} - \| \hat{\mathbf{X}}_0^{(i)} - \hat{\mathbf{X}}_0^{(j)} \|_2 \right)^2 \\
\end{aligned}
$$


## 6.2 Steering Force via Gradients

$$
\mathbf{F}_{\text{clash}} = - \nabla_{\hat{\mathbf{X}}_0} E_{\text{clash}}(\hat{\mathbf{X}}_0)
$$

## 6.3 Apply Time-Dependent Grad Descent

$$
\hat{\mathbf{X}}_0^{\text{steered}} = \hat{\mathbf{X}}_0 + \lambda(t) \cdot \mathbf{F}_{\text{clash}}
$$

## 6.4 DDPM

$$
\mathbf{X}_{t-1} = \text{DDPM\_Step}(\hat{\mathbf{X}}_0^{\text{steered}}, \mathbf{X}_t, t)
$$

## Questions

### **How does the Steric Clash Penalty "push" atoms apart during Denoising?**

- During each denoising step, the system feeds <img src="https://githubusercontent.com" valign="middle"> into the clash energy function <img src="https://githubusercontent.com" valign="middle">. If atoms overlap, <img src="https://githubusercontent.com" valign="middle"> rises sharply. The system then computes partial derivatives <img src="https://githubusercontent.com" valign="middle">, which yield a 3D repulsive force vector in real space. Before moving to <img src="https://githubusercontent.com" valign="middle">, the model applies this force through gradient descent, pulling apart atoms that collide. This mechanism—called **BoltzSteering**—is the core safeguard that ensures the generated structure remains physically reasonable.

### **How does the Time-Dependent Contact Potential (<img src="https://githubusercontent.com" valign="middle">) control this?**

- **High noise (Large <img src="https://githubusercontent.com" valign="middle">)：** <img src="https://githubusercontent.com" valign="middle">. Allow them to overlap, so the model will focus on the macro structure
    
- **Low noise（<img src="https://githubusercontent.com" valign="middle">）：** <img src="https://githubusercontent.com" valign="middle"> rockets. Focus on the micro details will be observed

### Why not `no bond atom pair` in Steric Clash?

- For real molecules, 1-3 for bond angle and 1-4 for dihedral angles might be smaller than the sum of Van de Vaal radii, so we should consider these scenario. 