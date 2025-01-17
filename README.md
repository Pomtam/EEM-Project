# MY-RES Project: Improving Realism of Ensemble Ecosystem Models (EEMs)



## Introduction  
This project aims to improve the realism of the ensemble ecosystem model (EEM) generation process by integrating constraints on the steady-state abundances (𝑛\*) of species in a network. Building upon the Barbier et al. (2021) algorithm, known for its ability to sample interaction strengths (𝛼) characterising feasible ecosystems with a 100% acceptance rate, this improved algorithm would allow for the generation of feasible parameter sets based on supplied values of $n^*$ and growth rates $r$. Significantly, this eliminates the need (and the computational cost required) to check the feasibility (and realism) of sampled parameter sets.



## Key Benefits  
- **Efficient Feasibility Checking:** Generates feasible parameter sets directly, eliminating the computational cost of post-sampling feasibility checks.  
- **Algorithm Adaptations:** Modifies the Barbier et al. (2021) algorithm to support generalised Lotka-Volterra equations.  



## Major Modifications  
1. Removal of the expectation of negative interactions.  
2. Inclusion of self-regulation terms (𝛼ᵢᵢ) in computations.  
3. Retention of growth rates (𝑟) during computations.  
4. Flexible interaction strength distributions (𝛼ᵢⱼ) with adjustable means and variances.  
5. Support for predefined network structures (e.g., zero-interaction strengths).  

A streamlined version of the algorithm, based on Cong et al. (2017), achieves:  
- Simplified implementation.  
- Faster computation by avoiding the Gram-Schmidt process for rotation matrix generation.  

---



### Code Overview  
- **Developed in Python.**  
- **Key Libraries:** `numpy`, `matplotlib`.  
- Implements core functionalities:  
  - Sampling interaction matrices (𝛼) based on steady-state constraints.  
  - Calculating properties of multivariate Gaussian distributions for feasible ecosystems.  
  - Supporting pre-specified network structures with zero-interaction constraints.
 
### Future Work
- Enforce network stability 
