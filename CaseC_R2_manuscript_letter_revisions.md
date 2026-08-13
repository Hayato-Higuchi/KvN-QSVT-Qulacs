# Case C manuscript and response-letter revisions

## Manuscript

### Introduction, lines 208--211

Replace the four sentences with:

```tex
The third case is a nonlinear advection test at fixed domain length with varying numbers of grid points $N_x$,
intended to examine the effect of spatial refinement and the behavior of the algorithm as the system size increases.
We confirm that the spatial-discretization error decreases as $Delta x=L/N_x$ is reduced.
We also use $R=2$ to make the finite-$R$ difference between \textsf{KvN-QSVT} and \textsf{KvN-expm} visible.
```

### Case C objective, line 1011

Replace the Case C item with:

```tex
\item[Case (C)] To examine the effect of spatial refinement on the numerical solution, we perform 1D advection tests at fixed domain length $L=44$ while varying $N_x$ and setting $\Delta x=L/N_x$.
```

### QSVT truncation indices, line 1030

Replace the sentence with:

```tex
Case A uses the Jacobi--Anger truncation with $R=5$. The norm-deviation analysis in Case B uses $R=3$ so that the finite-$R$ QSVT contribution can be distinguished from the finite-$m$ KvN contribution. Case C uses $R=2$ to make the finite-$R$ difference between \textsf{KvN-QSVT} and \textsf{KvN-expm} visible. Case D uses $R=18$.
```

### Case C subsection, lines 1183--1211

Replace the subsection text with:

```tex
\subsection{1D nonlinear advection test at fixed domain length with various numbers of grid points~$N_x$}
To examine the effect of spatial refinement, we perform a 1D advection test on a sequence of grids at fixed domain length $L=44$, where the resolution is controlled by the number of grid points $N_x$ and $\Delta x=L/N_x$.
The accuracy metric is the $L^2$-norm deviation
$\Delta(t)=\left\lvert\left\|\mathbf{x}(t)\right\|_{L^2}-\left\|\mathbf{x}(0)\right\|_{L^2}\right\rvert$.
We also evaluate the initial spatial-discretization error by comparing the discrete spatial operator with its continuum expression.

The initial condition in Eq.~\eqref{eq:parameters_1D} is set as
\begin{align}
    u(x,t=0)=\sin(kx), \quad E(x,t=0)=0,
\end{align}
where $x\in[-L/2,L/2]$, $L=44$, $N_x=11,22,33,\text{ or }44$, $\Delta x=L/N_x=4,2,4/3,\text{ or }1$, $k=-2\pi/L$, and $N=2N_x$.
We set the plasma frequency $\sqrt{\frac{n_e}{\epsilon_0m_e}}q_e=-0.1$, $\Lambda=1$, the QSVT truncation index $R=2$, the time parameter $\tau=1$, and the KvN truncation order $m=2$.
The numbers of time steps are $N_t=128,250,374,\text{ and }500$ for $N_x=11,22,33,\text{ and }44$, respectively.
The corresponding final physical times are approximately $54.1$ in all cases.
As in Sec.~\ref{subsec:caseB}, we calculate the $L^2$ norm of $\mathbf{x}$.

The relative errors of the initial discrete spatial operator are $4.22\times10^{-1}$, $1.18\times10^{-1}$, $5.35\times10^{-2}$, and $3.03\times10^{-2}$ for $N_x=11,22,33,\text{ and }44$, respectively.
A log-log fit gives an observed convergence order of $1.90$.
Since $\Delta x=L/N_x$, this result directly verifies that the spatial-discretization error decreases under grid refinement at fixed domain length.

Figure~\ref{fig:L2norm_1D_nonlinear_QSVT_expm_various_Nx} shows the time evolution of the global $L^2$-norm deviation $\Delta(t)$ obtained using \textsf{KvN-QSVT} and \textsf{KvN-expm}.
The maximum \textsf{KvN-expm} norm deviations are $2.83\times10^{-2}$, $4.02\times10^{-2}$, $7.16\times10^{-2}$, and $5.35\times10^{-2}$ for $N_x=11,22,33,\text{ and }44$, respectively.
The long-time norm deviation is not monotonic in $N_x$ at fixed $m=2$ and is therefore not used alone as a measure of spatial-grid convergence.
With $R=2$, the \textsf{KvN-QSVT} curves visibly differ from the \textsf{KvN-expm} curves.
The maximum relative trajectory deviations $\left\|\mathbf{x}_{\mathrm{QSVT}}-\mathbf{x}_{\mathrm{expm}}\right\|_2/\left\|\mathbf{x}_{\mathrm{expm}}\right\|_2$ are $5.40\times10^{-3}$, $1.09\times10^{-2}$, $2.38\times10^{-2}$, and $2.97\times10^{-2}$ for $N_x=11,22,33,\text{ and }44$, respectively.
This result makes the finite-$R$ QSVT polynomial contribution visible while demonstrating agreement with \textsf{KvN-expm} within $2.97\times10^{-2}$ for all tested grids.
```

### Case C caption, line 1218

Replace the caption with:

```tex
\caption{\textcolor{red}{Time evolution of the discrete norm deviation $\left\lvert\|\mathbf{x}(t)\|_2-\|\mathbf{x}(0)\|_2\right\rvert$ in Case C at fixed domain length $L=44$ for $N_x=11,22,33,$ and $44$, corresponding to $\Delta x=4,2,4/3,$ and $1$. The KvN truncation order is $m=2$. Solid curves denote \textsf{KvN-QSVT} with $R=2$, and dashed curves with open markers denote \textsf{KvN-expm}. For each $N_x$, both methods use the same physical increment $\Delta t=\tau/\alpha$.}}
```

## Response letter

### Referee A, Minor Comment 2, lines 380--392

Replace the response with:

```tex
\begin{response}
Thank you for pointing out this issue.
We have repeated Case C at fixed domain length $L=44$ for $N_x=11,22,33,$ and $44$, corresponding to $\Delta x=L/N_x=4,2,4/3,$ and $1$.
The relative error of the initial discrete spatial operator decreases from $4.22\times10^{-1}$ to $3.03\times10^{-2}$, with an observed convergence order of $1.90$.
Since $\Delta r_1=\Delta x=L/N_x$ in this fixed-domain test, the revised calculation directly connects the spatial refinement used in Case C to Eq.~(A3).

We use $R=2$ in Case C to make the finite-$R$ QSVT contribution visible.
The maximum relative trajectory deviation between \textsf{KvN-QSVT} and \textsf{KvN-expm} increases from $5.40\times10^{-3}$ at $N_x=11$ to $2.97\times10^{-2}$ at $N_x=44$.
We have also clarified that the long-time $L^2$-norm deviation at fixed $m=2$ is not monotonic in $N_x$ and is therefore not used alone as evidence of spatial-grid convergence.
\end{response}
```

### Referee B, Minor Point 13, lines 909--921

Replace the repository-change list with:

```tex
{\color{red}Specifically, we have
(i) restored the missing definitions of $m$ and of the Hamiltonian dimension $M$ in the parameter cell of \texttt{analysis\_1D\_caseBC.ipynb}, which previously raised a \texttt{NameError};
(ii) corrected the output-directory name \texttt{output/caseBC} to \texttt{output/CaseBC} in \texttt{main\_expm\_1D\_caseBC.ipynb}, which failed on case-sensitive file systems;
(iii) revised Case C to use fixed domain length $L=44$, $\Delta x=L/N_x$, and the QSVT truncation index $R=2$, while retaining the time-step parameters used in the 2D Case D scripts;
(iv) added \texttt{classical\_rk4\_1D\_caseBC.py}, which generates the RK4 comparison figure, together with a README section documenting the parameter set of every figure;
(v) added \texttt{qsvt\_phase\_generation\_1D.py} and \texttt{qsvt\_R\_sweep\_1D\_caseB.py}, together with the generated phase-factor files and the $R$-labeled output data; and
(vi) added \texttt{case\_c\_grid\_convergence\_1D.py}, which generates the fixed-domain Case C data, validation results, and figure.}
```

The Case B $R$-dependence discussion and Fig. 5 remain unchanged because the sweep continues to use $R=3,5,7,$ and $9$.
