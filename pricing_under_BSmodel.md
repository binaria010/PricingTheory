<head>
  <script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
         displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
         processEscapes: true
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
  <meta name="google-site-verification" content="kuks5e4as6qBaGVCSzmHkQJa5Tss89_g5DmRXeUi7K8" />
</head>


<h2> Black-Scholes Model </h2>

<a href = "https://colab.research.google.com/drive/1zmVsqlT-ONA439dntqC3nrHy8JdljQwu" target="_blank">Colab file</a>

There are at least to different approaches to arrive at this model. One is by getting a partial differential  equation for the price of an option and the other is by the Martingale Representation Theorem (pricing under Risk-Neutral Measure)  which gives us directly the value of the option with no need of solving a PDE. The price of the latter approach is that the value of the option is expressed as a conditional expectation which is sometimes more difficult to compute/estimate than a PDE. Also, the Martingale setting allows us to arrive to a formula for the value of an option whose payoff at maturity time could be path-dependent and whose underlying asset follows a Generalized Brownian Motion, that is, the stock price has non constant mean return and volatility processes.

<h3> Some Ito Calculus. </h3>
First let us recall some definitions and formulas:

1. <strong> Ito Process </strong>. Is a stochastic process $\{X(t)\}$ of the form:

$$
\begin{align}
X(t) = X(0) + \int_0^t \Delta(s)dW(s) + \int_0^t\Theta(s)ds
\end{align}
$$

with $X(0)$ non random and $\{\Delta(t)\},~\{\Theta(t)\}$ stochastic process adapted to the filtration $\mathcal{F}_t$ induce by the Brownian motion $\{W(t)\}$ and such that 
$$
\mathbb{E}[\int_0^t\Delta^2(s)ds], ~~ \int_0^t|\Theta(s)|ds <\infty
$$

The formula above for $X(t)$ is equivalent to

$$
dX(t) = \Theta(t)dt +\Delta(t)dW(t).
$$



2. <strong> Ito Formula</strong>. Let $\{X(t)\}$ be an Ito process and $f:\mathbb{R}^2 \to \mathbb{R}$ be a $C^2$ function. Then for all $t>0$

$$
d(f(t, X(t))) = f_t(t, X(t))dt + f_x(t, X(t))dX(t) + \frac{1}{2}f_{xx}(t X(t))dX(t)dX(t) 
$$

where $f_t, ~f_x, ~f_{xx}$ denote partial derivatives of the function $f$ with respect to the firs and second variable. Aslo to compute $dX(t)dX(t)$ we use "the rules":

$$
dt*dt = dt*dW(t) =0\qquad \text{and }\quad dW(t)dW(t) = dt.
$$

3. <strong>Stochastic Differential Equation (SDE).</strong> An SDE is an equation of the form:
$$
dX(t) = \beta(t, X(t))dt + \gamma(t, X(t))dW(t)
$$

where $\beta$ and $\gamma$ are random processes called the <em> drift</em> and <em>difussion</em> respectively. In adition to this equation, an initial condition of the form $X(t_0)=x$ is especified. The problem is then to find a stochastic process $\{X(t)~~t\geq t_0\}$ such that:

$$
\begin{align*}
X(t_0) =& x\\
X(t) =& X(t_0) + \int_{t_0}^t\beta(u, X(u))du + \int_{t_0}^t\gamma(u, X(u))dW(u)
\end{align*}
$$

The paradigmatic example in this notebook is the <em>Geometric Brownian Motion.</em>

<strong>Geometric Brownian Motion</strong>. Is a process of the form:
$$
dX(t) = X(t)(\mu dt + \sigma dW(t))
$$

The solution of the above SDE is

$$
X(t) = X(0)\exp\left[(\mu - \frac{1}{2} \sigma^2)t + \sigma W(t)\right]
$$

the Generalized Geometric Brownian Motion is one with $\mu$ and $\sigma$ non constant adapted processes:

$$
dX(t) = X(t)(\mu(t) dt + \sigma(t) dW(t))
$$

The solution of the above SDE is

$$
X(t) = X(0)\exp\left[\int_0^t(\mu(s) - \frac{1}{2} \sigma(s)^2)ds + \int_0^t\sigma(s) dW(s)\right]
$$


<h3> Black-Scholes Equation. </h3>

Consider a stock whose price $S(t)$ is a Geometric Brownian motion

$$
dS(t) = S(t)(\mu dt + \sigma dW(t))
$$
and a European call option on this asset, that is, the payoff of this option is $V(T) = (S(T) - K)^{+}$, where $T$ is the maturity and $K$ is the strike.

As in the binomial model above, the goal is to price this derivative security. To do this we need to determine the initial capital $X(0)$ and a portfolio process $\Delta(t)$ (the amount of shares of the asset at time $t$) to perfectly hedge a short position on this option. This hedging portfolio will satisfy $X(t) = V(t)~~\forall t\in[0, T]$ with probability 1, where $V(t)$ represents the value of the option at time $t$. Then the value of this option, $V(0)$, is the initial capital  $X(0)$.

Black and Scholes argued that the value $V(t)$ of the option at any time $t$ should only depend on the time (or time to expiration to be more precise) and the value of the stock at that time and of course on the parameters of the model ($\sigma, r, K$). Let us denote the value of the option as the function $c(t, x)$ where $S(t)=x$. The goal is therefore determine the functional form of $c$. The evolution of the stochastic process $c(t, S(t))$ can be computed by means of the Ito formula:

$$
\begin{align*}
d(c(t, S(t))) =& c_t(t, S(t))dt + c_x(t, S(t))dS(t) + \frac{1}{2}c_{xx}(t, S(t))dS(s)dS(t)\\
= & \left(c_t(t, S(t))dt + \mu c_x(t, S(t))S(t) + \frac{1}{2}c_{xx}(t, S(t))\sigma^2 S(s)^2\right)dt + c_x(t, S(t))\sigma S(t)dW_t
\end{align*}
$$

The (short option) hedging portfolio is made out of a long position in the asset and a money market with interest rate $r$.  At time $t$ the agent holds $\Delta(t)$ shares of the asset and the remainder of the portfolio, $X(t)-\Delta(t)S(t)$  is invested in the money market, therefore the evolution of $X$ is:

$$
\begin{align*}
dX(t) =& \Delta(t)dS(t) + r(X(t) - \Delta(t)S(t))dt \\
=& \Delta(t)(\mu S(t) dt +\sigma S(t)dW(t)) + r(X(t) - \Delta(t)S(t))dt \\
=& rX(t) +(\mu -r)\Delta(t)S(t)dt + \sigma\Delta(t)S(t)dW(t).
\end{align*}
$$

One way to guarantee the equality of both prices is by making:

$$
d(X(t)) = d(c(t, S(t))) \qquad \text{and }\quad X(0) = c(0, S(0)).
$$

For if $d(X(t)) = d(c(t, S(t)))$ then:

$$
X(t) -X(0) = c(t,S(t)) - c(0, S(0))
$$
and since $X(0) = c(0, S(0))$ we get the desired equality $X(t) =c(t, S(t))$. Therefore

$$
rX(t)dt +(\mu -r)\Delta(t)S(t)dt + \sigma\Delta(t)S(t)dW(t) = \left(c_t(t, S(t))dt + \mu c_x(t, S(t))S(t) + \frac{1}{2}c_{xx}(t, S(t))\sigma^2 S(s)^2\right)dt + c_x(t, S(t))\sigma S(t)dW_t
$$

If we define the hedging process $\Delta(t)$ such that:

$$
\begin{equation}
\sigma\Delta(t)S(t) = c_x(t, S(t))\sigma S(t) \qquad \text{Hedging rule}
\end{equation}
$$

Then, the difference portfolio $X(t) - c(t,S(t))$ is risk free (since the uncertainty from the Brownian motion is gone), then this portfolio grows at rate $r$ and thus:

$$
\begin{align*}
d(X(t) - c(t,S(t))) =& rX(t)dt +(\mu -r)\Delta(t)S(t)dt -\left(c_t(t, S(t))dt + \mu c_x(t, S(t))S(t) + \frac{1}{2}c_{xx}(t, S(t))\sigma^2 S(s)^2\right)dt \\
=& r(X(t) - c(t,S(t)))dt 
\end{align*}
$$

From the second equality above and replacing the formula for $\Delta_t$ it follows that:

$$

$$













$$
c_t(t, S(t)) +rc_x(t,S(t))S(t) +\frac{1}{2}c_{xx}(t,S(t))\sigma^2 S(t)^2 - rc(t,S(t)) = 0
$$

That is, the function $c(t,x)$ satisfies the partial differential equation with terminal condition:

$$
\begin{cases}
& c_t(t, x) +rxc_x(t,x) +\frac{1}{2}\sigma^2 x^2c_{xx}(t,x) - rc(t,x) = 0, \quad \forall t\in[0,T)\\

& c(T,x) = (x-K)^{+}
\end{cases}
$$

This is known as the Blac-Scholes Merton equation.
Suppose we have found this function. If an investor starts with inital capital $X(0) =c(0,S(0))$ and uses the hedge $\Delta(t)=c_x(t,S(t))$, then $X(t) =c(t,S(t))$ for all $t\in[0, T)$. Now, taking the limit $t\uparrow T$ and using the continuity of $X$ and $c$ one can conclude that $X(T) = c(T, S(T))=(S(T)-K)^{+}$. This means that the short postion has been successfully hedge. No matter what path the stock price follows, at expiration date the agent hedging the short position has a portfolio whose value agrees with the derivative security payoff.
By the use of some methods from partial differential equations it is possible to arrive at a closed form of the solution of BS. In the next section we will arrive to the same expression for the price of the option by means of the Risk Neutral Measure. 

The solution of BS is:

$$
c(t,x) = x\Phi(d_{+}(T-t, x)) - Ke^{-r(T-t)}\Phi(d_{-}(T-t, x)), \quad t\in [0, T), \quad x>0
$$

with
$$
d_{\pm}(u, x) = \frac{1}{\sigma\sqrt{u}}\left[\log\frac{x}{K} +\left(r\pm\frac{1}{2}\sigma^2\right)u\right]
$$

and $\Phi$ the cummulative distribution function of a standard normal random variable.


<h3>Example</h3>

Consider an asset whose price follows a geometric Brownian motion

$$
dS(t) = S(t)(r dt +\sigma dW(t)), ~~t>0, ~~S(0)=s_0
$$

with $r=0.05, \sigma=1, s_0=1, T=1$. Consider an European call with strike $K=1$. Recall that the price of the this option can be obtained either from the risk-neutral pricing formula or from the closed form in terms of the standard normal cummulative distribution $\Phi$:

$$
V= V(0) =\mathbb{E}(e^{-rT}(S(T) - K)^{+}) = S(0)\Phi(d_{+}) - Ke^{-rT}\Phi(d_{-})
$$


A final note before we start: observe that we have written everthing in the risk-neutral measure so we drop the tilde out from the expectation formula.


```python
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
```


```python
r = 0.05
sigma = 1
s0 = 1
K = 1
T = 1
d_minus = (np.log(s0/K) + (r -0.5*sigma**2)*T)/(sigma*np.sqrt(T))
d_plus = (np.log(s0/K) + (r +0.5*sigma**2)*T)/(sigma*np.sqrt(T))

```


```python

V = s0*scipy.stats.norm.cdf(d_plus, loc = 0, scale =1) - K*np.exp(-r*T)*scipy.stats.norm.cdf(d_minus, loc = 0, scale =1)

print(f" The time 0 fair price of the option is: {V}")
```

     The time 0 fair price of the option is: 0.3984016248343718


With the same model as before for the stock price, what if we wanted to price the Asian option whose payoff is:

$$
V(T) = \left(\frac{1}{m}\sum_{j=1}^m S_{t_j} -K\right)^{+}
$$

where $0 = t_0 < t_1 < \dots < t_m =T$ is a fixed set of dates, $T$ is the maturity and $K$ is the strike price. 

The time -0 price of the Asian option is given by 

$$
V(0) = \mathbb{E}\left[e^{-rT}V(T) \right].
$$

In this case, since the sum of log normal random variables does not have a known (closed formula) distribution and also there is no an explicit solution for the partial differential equation for this problem (the one obtaind by <a href = "https://en.wikipedia.org/wiki/Feynman%E2%80%93Kac_formula">Feyman-Kac Theorem</a>) we will have to estimate this time-0 price using Monte Carlo simulations.

