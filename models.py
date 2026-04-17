import QuantLib as ql
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

class HestonSimulator:
    def __init__(self, S0=100, r=0.02, 
                 q=0., v0=0.1, 
                 kappa=1.5, theta=0.04, 
                 sigma=0.5, rho=-0.5, 
                 maturities = [3, 5, 10, 20, 120, 240], log_mon = np.log(np.array([0.9, 0.95, 1, 1.05, 1.1])),
                engine = "analytical"):
        self.S0 = ql.SimpleQuote(S0)
        self.v0 = ql.SimpleQuote(v0)
        self.r = ql.SimpleQuote(r)
        self.q = ql.SimpleQuote(q)
        self.calendar = ql.NullCalendar()
        self.rho = ql.SimpleQuote(rho)
        self.sigma = ql.SimpleQuote(sigma)
        self.kappa = ql.SimpleQuote(kappa)
        self.theta = ql.SimpleQuote(theta)

        self.dc = ql.Actual365Fixed()

        self.engine = engine
        

        self.maturities = [ql.Period(n, ql.Days) for n in maturities]
        self.log_mon = log_mon
        

    def simulate(self, T, nburn=5):
        start_date = ql.Date(17, ql.February, 2026)
        s_path, v_path = self.simulate_heston_path(T, nburn = nburn)

        m = len(self.maturities)
        n = len(self.log_mon)
        iv_cube = np.full((T,m,n), np.nan, dtype = float)
        price_cube = np.full((T,m,n), np.nan, dtype = float)
        dates = []

        for t in tqdm(range(T), desc = "Simualting surface", total = T):
            eval_date = self.calendar.advance(start_date, ql.Period(t,ql.Days))
            dates.append(eval_date)
            iv, prices = self.simulate_surface(s_path[t], v_path[t], eval_date)
            
            iv_cube[t,:,:] = iv
            price_cube[t,:,:] = prices

        dates = [date.ISO() for date in dates]
        return s_path, v_path, dates, iv_cube, price_cube


    def simulate_surface(self, S, vt, eval_date):
        strikes = [S*np.exp(mon) for mon in self.log_mon]

        ql.Settings.instance().evaluationDate = eval_date

        spot = ql.QuoteHandle(ql.SimpleQuote(S))
        divs = ql.YieldTermStructureHandle(ql.FlatForward(0,self.calendar, ql.QuoteHandle(self.q), self.dc))
        rates = ql.YieldTermStructureHandle(ql.FlatForward(0,self.calendar, ql.QuoteHandle(self.r), self.dc))

        process = ql.HestonProcess(
            rates, divs, spot, float(vt),
            float(self.kappa.value()),
            float(self.theta.value()),
            float(self.sigma.value()),
            float(self.rho.value()))

        model = ql.HestonModel(process)
        if self.engine == "analytical":
            engine = ql.AnalyticHestonEngine(model)

        n = len(strikes)
        m = len(self.maturities)
        iv = np.full((m,n), np.nan, dtype = float)
        prices = np.full((m,n), np.nan, dtype = float)

        for i, period in enumerate(self.maturities):
            expiry = self.calendar.advance(eval_date, period)
            exercise = ql.EuropeanExercise(expiry)

            T = self.dc.yearFraction(eval_date, expiry)

            discount = float(rates.discount(expiry))
            div = float(divs.discount(expiry))
            F = float(S) * div / discount

            for j, K in enumerate(strikes):
                payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
                option = ql.VanillaOption(payoff, exercise)
                option.setPricingEngine(engine)

                p = option.NPV() 
                prices[i,j] = p
                

                try:
                    stddev = ql.blackFormulaImpliedStdDev(
                        ql.Option.Call, float(K), float(F), float(p),
                        discount, 0.0, 0.2 * np.sqrt(T), 1e-8, 200
                    )
                    iv[i,j] = float(stddev) / np.sqrt(T)
                except RuntimeError:
                    iv[i,j] = np.nan
        
        return iv, prices
        
        

    def simulate_heston_path(self, T, S0=None, v0=None, dt=1/365, nburn=5):
        if S0 is None:
            S0 = self.S0.value()
        if v0 is None:
            v0 = self.v0.value()
        rho = self.rho.value()
        sigma = self.sigma.value()
        kappa = self.kappa.value()
        theta = self.theta.value()
        r = self.r.value()
        q = self.q.value()
        
        s_path = np.empty(T)
        v_path = np.empty(T)
        s_path[0] = S0
        v_path[0] = v0

        for i in range(1,T):
            s_path_burn = np.empty(nburn+1)
            v_path_burn = np.empty(nburn+1)
            s_path_burn[0] = s_path[i-1]
            v_path_burn[0] = v_path[i-1]
            
            for t in range(1, nburn+1):
            
                eps_1 = np.random.normal()
                eps_2 = np.random.normal()
                dW1 = eps_1 * np.sqrt(dt / nburn)
                dW2 = np.sqrt(dt / nburn) * (eps_1 * rho + eps_2 * np.sqrt(1-rho**2))
        
                v_path_burn[t] = v_path_burn[t-1] + kappa*(theta - v_path_burn[t-1]) * (dt/nburn) + sigma*np.sqrt(v_path_burn[t-1])* dW2
                if v_path_burn[t] < 0:
                    print(f"Waring: Heston vol is negative at {i}")
                    v_path_burn[t] = 1e-3
                s_path_burn[t] = s_path_burn[t-1] * np.exp((r-q-0.5*v_path_burn[t-1])*(dt/nburn) + np.sqrt(v_path_burn[t-1]) * dW1)

            s_path[i]=s_path_burn[-1]
            v_path[i]=v_path_burn[-1]

        return s_path, v_path

# обеспечивает безарбитражность только по времени
class LinearImputer(): 
    def __init__(self, maturities):
        
        self.tau = maturities / 365
        
    def impute(self, iv):
        T, M, K = iv.shape
        total_var = iv **2 * self.tau.reshape(1, -1, 1)
        for t in range(T):
            for k in range(K):
                total_var[t,:,k] = self.fill_1d(total_var[t,:,k])
        total_var[total_var <= 0] = 1e-3
        return np.sqrt(total_var / self.tau.reshape(1,-1,1))


    def fill_1d(self, iv_t):
        mask = np.isfinite(iv_t)
        interpol = interp1d(self.tau[mask], iv_t[mask],
                           kind = "linear", 
                           fill_value = "extrapolate",
                           bounds_error = False,
                           assume_sorted = True)
        return interpol(self.tau)
            
            

# (len(maturities), len(log_mon))
class IvPCA(TransformerMixin, BaseEstimator):
    def __init__(self, nfactors):
        self.pca = PCA(n_components = nfactors)
        
    def transform(self, iv):
        T, M, K = iv.shape
        X = iv.reshape(T,M*K, order = "C") # по страйкам -> по maturity
        return self.pca.transform(np.log(X)) # (T,nfactors)

    def fit(self,iv, y = None):
        T, M, K = iv.shape
        self.M = M
        self.K = K
        X = iv.reshape(T,M*K, order = "C")
        self.pca.fit(np.log(X))
        return self

    def inverse_transform(self, factors):
        if factors.ndim < 3:
            X = np.exp(self.pca.inverse_transform(factors))
            return X.reshape(factors.shape[0], self.M, self.K, order = "C")
        else:
            n = factors.shape[0]
            T = factors.shape[1]
            paths = np.stack([self.pca.inverse_transform(factors[i, :, :]).reshape(T, self.M, self.K, order = "C") for i in range(n)])
            return np.exp(paths)


class VARDynamic:
    def __init__(self, factors, maxlags=1):
        self.var = VAR(factors)
        self.res = self.var.fit(maxlags=maxlags)

    def simulate(self, nsteps, nsim):
        init = self.res.endog[-self.res.k_ar:]   # последние p лагов
        paths = np.stack(
            [
                self.res.simulate_var(
                    steps=nsteps,
                    initial_values=init
                )
                for _ in range(nsim)
            ],
            axis=0
        )
        return paths