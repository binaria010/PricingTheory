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


<h1> Pricing Options </h1>

The following is a toy example to understand the <strong> Risk Neutral Measure </strong> for pricing derivative securities. Under this risk neutral measure it is guaranteed a no arbitrage price.

First, let us define this notion of arbitrage:

<strong>Definition </strong> An arbitrage is a portfolio value processe $X(t)$ satisfying that $X(0)=0$ and for some time $T>0$:

$$
\mathbb{P}(X(T) \geq 0)=1\quad\text{and }\quad \mathbb{P}(X>0)>0.
$$

<h5> Toy Example</h5>

Consider a one-period model, the stock a time 0 has price $S(0)=4$, at time $t=1$ with probability $p$ it goes up by a factor of $u=2$ and with probability $1-p$ it goes down by a factor $d=1/2$. The interest rate is $r=1/4$. Therefore 

$$
S_1 = \begin{cases}
2*S_0 = 8 &\quad\text{with probability } p\\
0.5*S_0 = 2 &\quad\text{with probability } 1-p
\end{cases}
$$
Consider an European call option with maturity of $T=1$ and strike $K=5$. The payoff is therefore 

$$
(S_1 - K)^{+} = \begin{cases}
(8-5)^{+} = 3 &\quad\text{with probability } p\\
(2-5)^{+} = 0 &\quad\text{with probability } 1-p
\end{cases}
$$

The goal is to determine a price for this option. To achieve this, consider the following situation: You start with an initial wealth $X_0=1.2$ and buy $\Delta_0 =0.5$ shares of the stock whose price is $S$. Since the stock at time zero costs $4$ per share, you must use your initial wealth $X_0$ and borrow and additional of $0.80$ to do this. This leave you with a cash position of $X_0 - \Delta_0 S_0 = -0.8$ (a debt of 0.8 to the money market). At time 1 your cash position will be $(1+r)(X_0 - \Delta_0 S_0) = -1$ (i.e the cash at time 0 with rate growth of $r =1/4$) This means that at time 1 you will have a debt of 1 to the money market. 

On the other hand, at time 1 you will have your stock valued at $\Delta_0 S_1$ , that is, with probability $p$ it will be $\Delta_0 *S_1 = 0.5*8 = 4$ and with probability $1-p$ it will be valued at $\Delta_0 *S_1 = 0.5*2 =1$, therefore your wealth at time 1 will be

$$
X_1 = \Delta_0 S_1 + (1+r)(X_0 - \Delta_0 S_0)=\begin{cases}
4 + (-1)= 3 &\quad\text{with probability }p\\
1 + (-1)=0 &\quad\text{with probability } 1 - p
\end{cases}
$$

That is, your portofolio $X$ at time 1 has the same value as the call option. We say that we have <em> replicated the option</em> by trading in the stock and money markets.

The initial wealth $X_0=1.20$ needed to set up the replicating portfolio $X$ is the <em>no-arbitrage price of the option at time zero </em>. For, if one could sell the option for more than 1.2, say, for 1.21, then the seller could invest the excess 0.01 in the money market and use the remaining 1.2 to replicate the option. At time 1 the seller would be able to pay off the option regardeless of the value of the stock a time 1, S_1, and still have the remaining $(1+r)*0.01 = 0.0125$ resulting from the money market investment. This is an arbitrage because the seller of the option needs no money initially and without risk of loss has 0.0125 at time 1. On the other hand if one could buy the option for less than the $X_0 =1.2$, say, for 1.19 then one should buy the option and set up the replicating portfolio in reverse. That is, we sell-short $\Delta_0=1/2$ of the stock (we receive 2 dollars) at time 0 then with 1.19 buy the option, invest 0.8 in the money market and put the ramaining 0.01 in a separate money market. At time 1 if the stock goes up then the option pays 3 and the 0.8 is worth 1, thus with this 4 you are able to close the short position. On the contrary, if the stock goes down the option pays 0 and the 0.8 is at time 1 worth 1 dollar which is also the value you have to pay to close the short position. In either case you will end up with 0.0125 which is the 0.01 in the separate money market whose value at time 1 is $(1+r)0.01=0.0125$. Therefore there is again arbitrage.

How one would get to this fair price of 1.20?   To determine the price  at time 0 of this opton, $V_0$ suppose we begin with wealth $X_0$ and buy $\Delta_0$ shares of the stock $S$ ($X_0$ and $\Delta_0$ unknown). Then the cash position at time 0 is $X_0 - \Delta_0*S_0$ and thus, the value of this portfolio (of stock and money market) at time 1 is:

$$
X_1 = \Delta_0*S_1 +(1+r)(X_0 - \Delta_0*S_0) = (1+r)X_0 + \Delta_0(S_1 -(1+r)S_0)
$$
The goal is to choose $X_0$ and $\Delta_0$ so that at time 1 the value of this portfolio and the value of the option coincide. If we denote $V_1$ the value of the option at time 1, we know that its value is with probability $p$ 3 dollars and with probability $1-p$ 0 dollars (this is because the maturity is 1 and at this final time the value of the option is its payoff). Thus we llok for $X_0,, \Delta_0$ such that

$$
X_1 = (1+r)X_0 + \Delta_0(S_1 -(1+r)S_0) = V_1
$$

which is equivalent to 

$$
X_0 + \Delta_0\left(\frac{1}{1+r}S_1 -S_0\right) = \frac{1}{1+r}V_1
$$

Recall that $S_1$ is random and so  are $V_1, X_1$. The random variable $S_1$ can take two possible values: $S_0*u$ and $S_0*d$ where $u=2,~ d= \frac{1}{2}$ with probabilities $p$ and $1-p$ respectively. Also, $V_1$ take the values of 3 and 0 with this probabilities. Then we have two equations:

$$
\begin{aligned}
X_0 + \Delta_0\left(\frac{1}{1+r}S_0*u -S_0\right) =& \frac{1}{1+r}3  \qquad\text{(1)}\\
X_0 + \Delta_0\left(\frac{1}{1+r}S_0*d -S_0\right) =& \frac{1}{1+r}0  \qquad\text{(2)}
\end{aligned}
$$




To find the values of $X_0$ and $\Delta_0$ out of these equations let us multiply equation (1) by a the number $q$ and equation (2) by the number $1-q$ and add them. These numbers are for now unknown but they will be determined as well. Thus we get:

$$
\begin{equation}
X_0 + \Delta_0\left(\frac{1}{1+r}(q*S_0*u + (1-q)*S_0*d) -S_0\right) = \frac{1}{1+r}(3q +0(1-q)) \tag{3}
\end{equation}
$$

If $q$ is chosen such that $S_0 = \frac{1}{1+r}(q*S_0*u + (1-q)*S_0*d)$ then we will get from (3) that:

$$
X_0 =  \frac{1}{1+r}(3q +0(1-q))
$$

And 
$$
\Delta_0 = \frac{3 - 0}{S_0*u - S_0*d} \qquad \text{Delta-Hedging formula}
$$ 

obtained by substracting equation (2) from equation (1). 

Finally, the number $q = \frac{1+r - d}{u - d}$ obtained from  $S_0 = \frac{1}{1+r}(q*S_0*u + (1-q)*S_0*d)$.

Note that if $0<d<(1+r) < u <1$, then the number $q$ satisfies the following: $q,~ 1-q \in(0,1)$, and their sum is 1. This means that the numbers $q, 1-q$ can be interpreted as probabilities and the expression for $X_0$ is nothing but the discounted expected value of the random variable $V_1$ under the probability law $q$, i.e,

$$
X_0 = \frac{1}{1+r}\mathbb{E}_q(V_1)
$$




This new probability law is called the <strong> Risk Neutral Measure </strong>. Observe that, by its very own definition:

$$
S_0 = \frac{1}{(1+r)^0}S_0 = \frac{1}{1+r}(q*S_0*u + (1-q)*S_0*d) =\frac{1}{1+r}\mathbb{E}_q(S_1\vert S_0) 
$$

that is, the discounted expected value of $S_1$ given the value $S_0$, coincides with (the discounted)  $S_0$, meaning that the stock price is a Martingale process. From the equation for $X_1$, by taking expected value with respect to the probability law $q$ we can conclude the same for the price process of the portfolio, i.e:

$$
\frac{1}{1+r}\mathbb{E}_q(X_1\vert X_0) = X_0 = \frac{1}{(1+r)^0}X_0.
$$

In the examples below we will start already from the measure $q$ which can be defined in terms of the parameters of the problem such us th numbers $u, d, r, \sigma$ and the measure $p$ will not play any role. Finally, instead of discounting by multiplying by the fraction $\frac{1}{1+r}$ we will do it by muliplying by $\exp(-\widetilde{r})$ with $\widetilde{r}$ such that those two quantities coincide, also we will lose the tilde in the notation. 






<h2> Pricing under CRR binomial tree for a non path-dependent payoff </h2>

In this section we implement the <strong> Backwards Induction Algorithm </strong> for pricing options whose payoff $g$ is a function of the stock price at maturity time, i.e, $g(S_N)$. We make both the European and American (although they are, technically speaking, path-dependent) cases under the Cox-Ross-Rubinstein parametrisation for the binomial tree.

Examples of non path-dependent payoffs are:

* European call option with payoff $g(S_T) = (S(T) - K)^{+}$
* European put option whith payoff $g(S_T) = (K - S(T))^{+}$

where in both cases $T$ is the maturity, $S(t)$ is the stock price at time $t$ and $K$ is the strike value.
 

The algorithm is based in the following proposition, which is nothing but the <em>dynamic programming</em> formulation of the problem.

<strong>Proposition</strong>. Let $V^{n}:=e^{-r(N-n)}\mathbb{E}[g(S_N) \vert \mathcal{F}_n]$ be the $n$-time fair value of the option with payoff function $g$. Then the following recursive formula holds:

$$
V^{n} = \begin{cases}
&g(S_n) \quad \text{for } n=N;\\
&e^{-r}\mathbb{E}[V^{n+1}\vert \mathcal{F}_n]\quad \text{for } n =0, 1, \dots, N-1.
\end{cases}
$$

<strong> Backwards induction algorithm. </strong>

* Under the binomial tree model, at fixed level $n$ the stock price $S_n$ can be one of the values $S_n = s_k^{n} = u^{n-k}d^{k}S_0$ for $k=0, 1, \dots n$, then the random variable $V^{n}$ takes the values:

$$
V_k^{n} = e^{-r}(qV_k^{n+1} +(1-q)V_{k+1}^{n+1})\quad k=0,1,\dots, n
$$

then the algorithm is as follows:



<ol>

    <il> Compute the option price at terminal node $N$, which is simply the payoff function, i.e. $V^{N} = g(S_N)$. This random variable takes values:
        
        $$
        V_{k}^{N} = g(u^{N-k}d^{k}S_0)\quad\text{for each } k =0,1,\dots, N
        $$
    </il>
    <il> Loop backward in time: for $n=N-1,N-2,\dots, 0$ compute

        $$
            V_k^{n} = e^{-r}(qV_k^{n+1} +(1-q)V_{k+1}^{n+1}) \quad\text{for each } k =0,1,\dots, n
        $$
    </il>
    <il> The time zero price is $V_0^{0}$    </il>
</ol>


```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

```


```python
def CRR_binomial_model_European(N, T, S_0, r, sigma, payoff,*params):

    delta_t = T/N
    u = np.exp(sigma*np.sqrt(delta_t))
    d = 1/u
    q = (np.exp(r*delta_t) - d)/(u - d)

    V = np.zeros((N+1, N+1))
    V[:, N] = np.array([payoff(S_0 * u**(N - k) * d**k, *params) for k in range(N+1)])

    for n in range(N-1, -1, -1):
        for k in range(0, n+1):
            V[k, n] = np.exp(-r*delta_t)*(q * V[k, n+1] + (1 - q) * V[k+1, n+1])

    return V
```


```python
T = 1
N = 2
sigma = 0.2
r = 0.01
K = 100
S_0 = 100.0

put = lambda S, K: np.maximum(K - S, 0)

```


```python
CRR_binomial_model_European(N, T, S_0, r, sigma, put, K)
```




    array([[ 6.5354428 ,  0.        ,  0.        ],
           [ 0.        , 12.68890338,  0.        ],
           [ 0.        ,  0.        , 24.63616836]])




```python
def CRR_binomial_model_American(N, T, S_0, r, sigma, payoff,*params):

    delta_t = T/N
    u = np.exp(sigma*np.sqrt(delta_t))
    d = 1/u
    q = (np.exp(r*delta_t) - d)/(u - d)

    V = np.zeros((N+1, N+1))
    intrinsic_value = np.zeros((N+1, N+1))
    intrinsic_value[:, N] = np.array([payoff(S_0 * u**(N - k) * d**k, *params) for k in range(N+1)])
    V[:, N] = intrinsic_value[:, N]
    continuation_value = np.zeros((N, N))

    for n in range(N-1, -1, -1):
        for k in range(0, n+1):
            intrinsic_value[k, n] = payoff(S_0 * u**(n - k) * d**k, *params)
            continuation_value[k, n] = np.exp(-r*delta_t)*(q * V[k, n+1] + (1 - q) * V[k+1, n+1])
            V[k, n] = np.maximum(intrinsic_value[k,n], continuation_value[k, n])

    return V, continuation_value, intrinsic_value
```


```python
T = 1
N = 2
sigma = 0.2
r = 0.01
K = 100
S_0 = 100

put = lambda S, K: np.maximum(K - S, 0)

V, V_tilde, g = CRR_binomial_model_American(N, T, S_0, r, sigma, put, K)

print(V)
print(V_tilde)
print(g)

```

    [[ 6.79232596  0.          0.        ]
     [ 0.         13.18765546  0.        ]
     [ 0.          0.         24.63616836]]
    [[ 6.79232596  0.        ]
     [ 0.         12.68890338]]
    [[ 0.          0.          0.        ]
     [ 0.         13.18765546  0.        ]
     [ 0.          0.         24.63616836]]



```python
def Pricing_under_CRR_binomial_tree(N, T, S_0, r, sigma, payoff, *params, option = "European"):


    if option == "European":
        V = CRR_binomial_model_European(N, T, S_0, r, sigma, payoff, *params)
        return V[0, 0]
    
    if option == "American":
        V, _ , _= CRR_binomial_model_American(N, T, S_0, r, sigma, payoff, *params)
        return V[0,0]
```


```python
T = 1
N = 100
sigma = 0.2
r = 0.05
K = 100.0
S_0 = 100.0

put = lambda S, K: np.maximum(K - S, 0)
Pricing_under_CRR_binomial_tree(N, T, S_0, r, sigma, put, K, option="American")
```




    6.082354409142444




```python
Pricing_under_CRR_binomial_tree(N, T, S_0, r, sigma, put, K, option = 'European')
```




    5.553554112321353




```python
def Exact_price_European_Put(S_0, K, r, sigma, T):

    d1 = (np.log(S_0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S_0/K) + (r - 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    value = -S_0*scipy.stats.norm.cdf(-d1,loc= 0, scale=1) + K*np.exp(-r*T)*scipy.stats.norm.cdf(-d2,loc= 0, scale=1)

    return value
```


```python
exact_value = Exact_price_European_Put(S_0, K, r, sigma, T)
print(exact_value)
```

    7.438302065026413



```python

```
