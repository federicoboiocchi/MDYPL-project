Image recognition: case study
================
[Federico Boiocchi](https://github.com/federicoboiocchi)
29 August 2025

## Motivation:

In many real word scenarios it is of interest to understand and classify
the content of an image in a fully automated way. Over the years, Image
and pattern recognition problems have become progressively more
difficult to solve requiring as a consequence increasingly more advanced
statistical models. This vignette deals with the problem of
**handwritten digit recognition**

## Model:

A possible way to approach image recognition consists of using
**logistic regression**, a widely known generalized linear model, that
allows to model a transformation of the conditional expected value of a
binary response $y$ with a linear combination of explanatory variables
$\mathbf{X}\boldsymbol{\beta}$: 

$$
\text{responses sample}\quad (y_1,\dots,y_n)\sim \text{Ber}(\mu_i)\quad\text{with}\ i=1,\dots,n\quad \text{and}\ \underset{n \times p}{\mathbf{X}}\\
\mathbb{E}[Y_i\ |\ \mathbf{x}_i^\top]=\mu_i=g^{-1}(\mathbf{x}_i^\top\boldsymbol{\beta})\quad\text{with}\ g^{-1}(\cdot)=\frac{\text{exp}(\cdot)}{1+\text{exp}(\cdot)}
$$

Differently from black box algorithms, two main advantages of such
method are the statistical interpretability of the quantities involved,
and a well developed inferential theory related for model parameters. In
this context, logistic regression can be used as a binary classifier to
associate to each image a probability that it contains a certain element
of interest provided that the GLM has been estimated on correctly
labelled data.

## Objective:

The primary purpose of this analysis, inspired to the case study in [P.
Sterzinger, I. Kosmidis](https://arxiv.org/abs/2311.07419), is to show
through a real data application how to use the new features of
[**brglm2**](https://github.com/ikosmidis/brglm2) that allows to perform
**Diaconis-Ylvisaker** penalized logistic regression and highlight the
strength points of using this statistical model for binary
classification problems. At the same time we will illustrate empirically
the results in *Theorem 3.1* and *3.5* of [P. Sterzinger, I.
Kosmidis](https://arxiv.org/abs/2311.07419) .

## Dataset:

Importing the data needed for the analysis

``` r
mfeat.fou <- read.table("C:/Users/andre/Downloads/multiple+features/mfeat-fou",sep = "")
mfeat.kar <- read.table("C:/Users/andre/Downloads/multiple+features/mfeat-kar",sep = "")
mfeat.pix <- read.table("C:/Users/andre/Downloads/multiple+features/mfeat-pix",sep = "")
```

We are using the [**Multiple
features**](https://archive.ics.uci.edu/dataset/72/multiple+features)
dataset from UCI machine learning repository. The structure of the
dataset is the following:

- **Digit images**: 2000 images of handwritten digits from 0 to 9, for
  each digit exactly 200 images are available. These pictures consist of
  15 x 16 grid of pixels (each pixel takes value in $\{0,1,...,6\}$) the
  pixel values for each image are stored by row in a matrix with
  dimensions 2000 x 240. Additionally, the images are arranged such that
  the first 200 rows correspond to zeros, the next 200 to ones, and so
  on up to nines. The following is a sample of ten digits one from each
  class. Depending on the font and level of noise in the digitisation
  phase, difficulties may arise in classifying them.

![](Vignette_brglm2_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

- **Explanatory variables**: the dataset includes several explanatory
  variable, however in this analysis we will primary focus on the values
  of **Fourier** coefficients for each image and the values of the
  **Principal component scores** (or *Karhunen-Loeve* coefficients).
  Specifically 64 PC scores and 76 Fourier coefficients are available.
- **Karhunen-Loeve** coefficients is an alternative way of referring to
  **PCA scores** which are the coordinates of the data points in the
  space of variables when a change of basis is performed in order to
  have maximum variance along each orthogonal direction in the cloud of
  data points. In this case the matrix of data is considered to be the
  grid of pixel values for each image. It is not clear to which
  component each PC score refer to.
- **Fourier coefficients** $C(u,v)$ are the weights of linear
  combinations of two-dimensional cosine and sine waves that added
  together reconstruct each one of the images. Each Fourier coefficient
  corresponds to a horizontal and vertical frequency of the wave
  $(u,v)$, and thus indicates the importance of a particular pattern
  within the image. The intuition is that each pixel value $f(x,y)$ in
  the image can be approximated with a linear combination of cosine and
  sine waves:
  
  $$f(x,y)=\underset{u,v}{\sum}C(u,v)e^{i2\pi\left(\displaystyle\frac{ux}{M}+\frac{vy}{N}\right)}$$
  
  This is also known as fourier decomposition of an image.
- **Binary response**: a label taking value either 0 or 1, indicating
  whether the image contains a specific, arbitrarily chosen digit. Since
  we know the ground truth for all digits we will be able to assess the
  accuracy of the final classification.

## Problem:

We arbitrarily decide that we are interested in detecting the number 7
(the entire analysis could have been reproduced with another number);
Once the training and test sets are randomly defined from the complete
matrix of data, we define the binary response $y$ as 1 if the image in
the training is a 7 or 0 otherwise. The goal will be to correctly
predict images in the test set. We will then compare the predicted
probabilities for each digit class using **MDYPL logistic regressions**,
both with and without parameter rescaling, and with covariates given by
Fourier scores alone or by both Fourier and PCA scores. The comparsion
of the models with different explanatory variables will be carried out
both empirically and via *Likelihood ratio test*.

We upload the functions needed to perform mdypl logistic regression
before **brglm2** is actually updated (after the update it will be
possible to just upload brglm2 through `library(brglm2)`)

``` r
source("solve_se.R")
source("se.R")
source("utils.R")
source("mdyplFit.R")
library("statmod") # to define Gauss-Hermite nodes
library("nleqslv") # to find the solution for the state evolution system
library(latex2exp) # to manage labels written in latex
```

``` r
# 2000 x 76 matrix of Fourier coefficients
fou <- mfeat.fou 
# 2000 x 64 matrix of Karhunen-Loeve coefficients
kar <- mfeat.kar
# ground truth labels (ordered from 0 to 9)
pix <- rep(c(0:9),each=200)
ind01 <- numeric(2000)
```

``` r
#renaming the Fourier coefficients covariates
colnames(fou) <- paste("fou", 1:76, sep = "")
#renaming the Karhunen-Loeve coefficients covariates
colnames(kar) <- paste("kar",1:64,sep = "")
```

``` r
# binary response (ground truth labels: 0,1)
digit_of_interest <- 7
ind01[which(pix==digit_of_interest)] <- 1 
```

by changing `digit_of_interest` the whole analysis can be performed for
other digits.

``` r
data <- cbind(pix,ind01,fou,kar)
```

Training test split

``` r
# set seed to have a reproducible analysis
set.seed(123)
train_id <- sort(sample(c(1:2000),size=1000,replace=FALSE))
train <- data[train_id,]
pix_tr <- pix[train_id]

test <- data[-train_id,]
pix_ts <- pix[-train_id]
```

## Analysis

**TRAINING SET**

``` r
# definition of the training design matrix by removing the labels
X_tr <- as.matrix(train[,-c(1,2)])
# standardization of the variables
X_tr_std <- scale(X_tr,center=TRUE,scale=TRUE)
# response variable (available for the training)
y_tr <- as.matrix(train[,2])
```

## High-dimensional Maximum Diaconis-Ylvisaker penalized logistic regression

We will use the following logistic regression model: 

$$
\mathbb{P}(y_i=1\ |\ \mathbf{x}_i^\top )= \frac{1}{1+\text{exp}(-\mathbf{x}_i^\top\boldsymbol{\beta})}\qquad \text{for}\ \  i = 1,\dots,1000\\
$$

where $y_i$ is the label indicating wheter the image is a 7 or not
and the linear predictor includes both Fourier coefficients and PCA
scores or only Fourier coefficients depending on the model used. Since
we are using a *Diaconis-Ylvisaker* penalized logistic regression, the
estimator of the regression coefficients has the following form: 

$$
\widehat{\boldsymbol{\beta}}_{\text{DY}}=\underset{\boldsymbol{\beta}\in\mathbb{R}^p}{\text{argmax}}\ l(\boldsymbol{\beta};\mathbf{y},\mathbf{X})+\text{ln}(\pi(\boldsymbol{\beta}))\quad \text{with}\quad \pi(\beta)\sim \text{DY}(\alpha,\boldsymbol{\beta}_p),
$$ 

where $l(\boldsymbol{\beta};\mathbf{y},\mathbf{X})$ represents the
log-likelihood of a sample of Bernoulli whose mean is modeled with the
inverse logit of a linear predictor; while
$\text{ln}(\pi(\boldsymbol{\beta}))$ represents the natural logarithm of
the *Diaconis-Ylvisaker* prior. This prior depends on two parameters:
$\alpha$, which is the shrinkage parameter, and $\boldsymbol{\beta}_p$,
the mode of the prior, which is usually set to $\mathbf{0}$ if not
otherwise specified.

Computationally speaking, the DY penalized log-likelihood is equivalent
to that of a standard logistic regression applied to transformed
responses. Consequently, a standard Fisher–Scoring algorithm can be
applied to these pseudo-responses.

**Method mdyplFit for logistic regression in `glm`**

We compute the model by using the standard `glm()` function but using a
non-standard method, namely, `method = mdyplFit`. The output will be an
object of class glm, that will inherit all the features of a classic glm
object but computed with a new fitter.

`mod_fk` stands for model with all Fourier and PC scores coefficients as
explanatory variables. An intercept is also included in the model.

``` r
# MDYPL logistic regression: Intercept + Fourier + PC scores
mod_fk <- glm(y_tr ~ X_tr_std, family = binomial(), method = "mdyplFit") 
# summary of the object of class glm with method = mdyplFit
(summ_fk <- summary(mod_fk))
#> 
#> Call:
#> glm(formula = y_tr ~ X_tr_std, family = binomial(), method = "mdyplFit")
#> 
#> Deviance Residuals: 
#>     Min       1Q   Median       3Q      Max  
#> -0.9399  -0.4174  -0.3341  -0.2480   0.9397  
#> 
#> Coefficients:
#>                 Estimate Std. Error z value Pr(>|z|)    
#> (Intercept)   -2.3491346  0.1320712 -17.787   <2e-16 ***
#> X_tr_stdfou1   0.2242358  0.2802190   0.800    0.424    
#> X_tr_stdfou2  -0.0901030  0.3427133  -0.263    0.793    
#> X_tr_stdfou3   0.3029658  0.2608591   1.161    0.245    
#> X_tr_stdfou4   0.1842861  0.2032689   0.907    0.365    
#> X_tr_stdfou5  -0.1464839  0.3770292  -0.389    0.698    
#> X_tr_stdfou6  -0.1639585  0.2422420  -0.677    0.499    
#> X_tr_stdfou7   0.0921854  0.3402478   0.271    0.786    
#> X_tr_stdfou8  -0.0999543  0.2816207  -0.355    0.723    
#> X_tr_stdfou9   0.1575419  0.2361585   0.667    0.505    
#> X_tr_stdfou10  0.0470554  0.2773082   0.170    0.865    
#> X_tr_stdfou11  0.1094043  0.2675718   0.409    0.683    
#> X_tr_stdfou12  0.0114748  0.2193760   0.052    0.958    
#> X_tr_stdfou13 -0.0433878  0.2351901  -0.184    0.854    
#> X_tr_stdfou14 -0.0363165  0.2535971  -0.143    0.886    
#> X_tr_stdfou15  0.0147339  0.2125917   0.069    0.945    
#> X_tr_stdfou16  0.0782495  0.2089815   0.374    0.708    
#> X_tr_stdfou17 -0.0502979  0.1969276  -0.255    0.798    
#> X_tr_stdfou18 -0.1599507  0.2281710  -0.701    0.483    
#> X_tr_stdfou19 -0.0162336  0.1827806  -0.089    0.929    
#> X_tr_stdfou20 -0.0159335  0.2217771  -0.072    0.943    
#> X_tr_stdfou21  0.0329361  0.1914652   0.172    0.863    
#> X_tr_stdfou22 -0.0057575  0.1917290  -0.030    0.976    
#> X_tr_stdfou23 -0.0516953  0.2050003  -0.252    0.801    
#> X_tr_stdfou24  0.0602832  0.1871768   0.322    0.747    
#> X_tr_stdfou25  0.0063439  0.1951835   0.033    0.974    
#> X_tr_stdfou26 -0.0493805  0.2100704  -0.235    0.814    
#> X_tr_stdfou27  0.1077176  0.1876667   0.574    0.566    
#> X_tr_stdfou28  0.0211822  0.1876078   0.113    0.910    
#> X_tr_stdfou29  0.0582904  0.1753292   0.332    0.740    
#> X_tr_stdfou30 -0.0561225  0.2108420  -0.266    0.790    
#> X_tr_stdfou31 -0.0326313  0.2267547  -0.144    0.886    
#> X_tr_stdfou32  0.0247590  0.1908451   0.130    0.897    
#> X_tr_stdfou33  0.0430942  0.2006340   0.215    0.830    
#> X_tr_stdfou34 -0.0549373  0.1713532  -0.321    0.749    
#> X_tr_stdfou35  0.0855801  0.2000647   0.428    0.669    
#> X_tr_stdfou36  0.0173868  0.1977638   0.088    0.930    
#> X_tr_stdfou37 -0.0169768  0.1889908  -0.090    0.928    
#> X_tr_stdfou38 -0.0502547  0.1895333  -0.265    0.791    
#> X_tr_stdfou39  0.0109426  0.1773301   0.062    0.951    
#> X_tr_stdfou40 -0.0495723  0.2067758  -0.240    0.811    
#> X_tr_stdfou41 -0.0838637  0.1945499  -0.431    0.666    
#> X_tr_stdfou42  0.0168276  0.1807792   0.093    0.926    
#> X_tr_stdfou43 -0.0262967  0.1721598  -0.153    0.879    
#> X_tr_stdfou44  0.0407412  0.1718087   0.237    0.813    
#> X_tr_stdfou45 -0.0878460  0.1934136  -0.454    0.650    
#> X_tr_stdfou46 -0.0054546  0.1818099  -0.030    0.976    
#> X_tr_stdfou47 -0.0166141  0.1888215  -0.088    0.930    
#> X_tr_stdfou48 -0.0313886  0.1813563  -0.173    0.863    
#> X_tr_stdfou49 -0.0054458  0.2002325  -0.027    0.978    
#> X_tr_stdfou50 -0.0277297  0.2002356  -0.138    0.890    
#> X_tr_stdfou51 -0.0110464  0.2048950  -0.054    0.957    
#> X_tr_stdfou52  0.0069858  0.1832133   0.038    0.970    
#> X_tr_stdfou53  0.1447083  0.2121421   0.682    0.495    
#> X_tr_stdfou54  0.0143152  0.2086332   0.069    0.945    
#> X_tr_stdfou55  0.0431878  0.2125682   0.203    0.839    
#> X_tr_stdfou56 -0.0354328  0.1783172  -0.199    0.842    
#> X_tr_stdfou57 -0.0242353  0.2222290  -0.109    0.913    
#> X_tr_stdfou58 -0.0279169  0.2108590  -0.132    0.895    
#> X_tr_stdfou59 -0.0389479  0.1999811  -0.195    0.846    
#> X_tr_stdfou60  0.0276333  0.2046535   0.135    0.893    
#> X_tr_stdfou61  0.0521473  0.2126400   0.245    0.806    
#> X_tr_stdfou62  0.0395163  0.1926121   0.205    0.837    
#> X_tr_stdfou63  0.0152832  0.2299328   0.066    0.947    
#> X_tr_stdfou64  0.0362363  0.1912904   0.189    0.850    
#> X_tr_stdfou65  0.0306232  0.2104211   0.146    0.884    
#> X_tr_stdfou66  0.0688214  0.2204145   0.312    0.755    
#> X_tr_stdfou67  0.0056549  0.2106131   0.027    0.979    
#> X_tr_stdfou68  0.0914617  0.1914131   0.478    0.633    
#> X_tr_stdfou69  0.0257544  0.2663202   0.097    0.923    
#> X_tr_stdfou70  0.0999087  0.2457294   0.407    0.684    
#> X_tr_stdfou71  0.1248594  0.2644877   0.472    0.637    
#> X_tr_stdfou72  0.1400090  0.2453364   0.571    0.568    
#> X_tr_stdfou73  0.2466510  0.3025242   0.815    0.415    
#> X_tr_stdfou74  0.2252064  0.3123098   0.721    0.471    
#> X_tr_stdfou75 -0.0378829  0.2020629  -0.187    0.851    
#> X_tr_stdfou76  0.0716015  0.2817859   0.254    0.799    
#> X_tr_stdkar1   0.5301615  0.6943141   0.764    0.445    
#> X_tr_stdkar2  -0.3176600  0.8614336  -0.369    0.712    
#> X_tr_stdkar3   0.2701323  0.4648596   0.581    0.561    
#> X_tr_stdkar4   0.1658486  0.3868175   0.429    0.668    
#> X_tr_stdkar5  -0.2606437  0.4508169  -0.578    0.563    
#> X_tr_stdkar6   0.0099536  0.3686070   0.027    0.978    
#> X_tr_stdkar7   0.0332592  0.3298169   0.101    0.920    
#> X_tr_stdkar8  -0.1408570  0.2613928  -0.539    0.590    
#> X_tr_stdkar9   0.3429039  0.4282926   0.801    0.423    
#> X_tr_stdkar10 -0.0914762  0.2853161  -0.321    0.749    
#> X_tr_stdkar11  0.0372215  0.2977744   0.125    0.901    
#> X_tr_stdkar12 -0.1723518  0.2456587  -0.702    0.483    
#> X_tr_stdkar13 -0.0552339  0.3321832  -0.166    0.868    
#> X_tr_stdkar14 -0.1026561  0.2330130  -0.441    0.660    
#> X_tr_stdkar15 -0.2407604  0.3216269  -0.749    0.454    
#> X_tr_stdkar16  0.0710662  0.2254582   0.315    0.753    
#> X_tr_stdkar17  0.1538120  0.2565939   0.599    0.549    
#> X_tr_stdkar18 -0.0998209  0.4555802  -0.219    0.827    
#> X_tr_stdkar19  0.1573443  0.2366639   0.665    0.506    
#> X_tr_stdkar20  0.2008782  0.3238655   0.620    0.535    
#> X_tr_stdkar21  0.0987102  0.4172787   0.237    0.813    
#> X_tr_stdkar22  0.1301852  0.4206788   0.309    0.757    
#> X_tr_stdkar23  0.4019991  0.2491801   1.613    0.107    
#> X_tr_stdkar24  0.0491334  0.2688257   0.183    0.855    
#> X_tr_stdkar25  0.2454017  0.2513706   0.976    0.329    
#> X_tr_stdkar26  0.0593057  0.2225516   0.266    0.790    
#> X_tr_stdkar27 -0.1483314  0.2303897  -0.644    0.520    
#> X_tr_stdkar28 -0.0105186  0.2992722  -0.035    0.972    
#> X_tr_stdkar29  0.0020017  0.3288971   0.006    0.995    
#> X_tr_stdkar30  0.1328250  0.2665356   0.498    0.618    
#> X_tr_stdkar31 -0.1032149  0.3649659  -0.283    0.777    
#> X_tr_stdkar32 -0.0540414  0.2351230  -0.230    0.818    
#> X_tr_stdkar33  0.0449954  0.2271894   0.198    0.843    
#> X_tr_stdkar34 -0.0099501  0.2079058  -0.048    0.962    
#> X_tr_stdkar35  0.0307248  0.2360667   0.130    0.896    
#> X_tr_stdkar36  0.2327857  0.2876595   0.809    0.418    
#> X_tr_stdkar37  0.1247543  0.2166558   0.576    0.565    
#> X_tr_stdkar38  0.0900636  0.2296410   0.392    0.695    
#> X_tr_stdkar39 -0.1353631  0.1884208  -0.718    0.473    
#> X_tr_stdkar40  0.0950893  0.2250553   0.423    0.673    
#> X_tr_stdkar41  0.0068532  0.2089066   0.033    0.974    
#> X_tr_stdkar42  0.0491540  0.2420316   0.203    0.839    
#> X_tr_stdkar43 -0.1003438  0.2753709  -0.364    0.716    
#> X_tr_stdkar44  0.1517107  0.2885685   0.526    0.599    
#> X_tr_stdkar45 -0.1580684  0.2019283  -0.783    0.434    
#> X_tr_stdkar46  0.1203946  0.1769501   0.680    0.496    
#> X_tr_stdkar47  0.0191984  0.1966677   0.098    0.922    
#> X_tr_stdkar48  0.0029478  0.1881906   0.016    0.988    
#> X_tr_stdkar49  0.1050297  0.2109634   0.498    0.619    
#> X_tr_stdkar50  0.0176480  0.1952747   0.090    0.928    
#> X_tr_stdkar51  0.0901939  0.1806149   0.499    0.618    
#> X_tr_stdkar52  0.0185500  0.1923768   0.096    0.923    
#> X_tr_stdkar53  0.1318793  0.2106697   0.626    0.531    
#> X_tr_stdkar54  0.0342727  0.1796738   0.191    0.849    
#> X_tr_stdkar55 -0.0413463  0.1762395  -0.235    0.815    
#> X_tr_stdkar56  0.0465789  0.1880955   0.248    0.804    
#> X_tr_stdkar57 -0.0315085  0.1871347  -0.168    0.866    
#> X_tr_stdkar58  0.0273410  0.1878315   0.146    0.884    
#> X_tr_stdkar59  0.1445634  0.1629674   0.887    0.375    
#> X_tr_stdkar60 -0.0591732  0.1926804  -0.307    0.759    
#> X_tr_stdkar61 -0.0551500  0.1658793  -0.332    0.740    
#> X_tr_stdkar62 -0.0077934  0.1722651  -0.045    0.964    
#> X_tr_stdkar63  0.0187850  0.1668718   0.113    0.910    
#> X_tr_stdkar64 -0.0006928  0.1650868  -0.004    0.997    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> (Dispersion parameter for binomial family taken to be 1)
#> 
#>     Null deviance: 618.85  on 999  degrees of freedom
#> Residual deviance: 159.10  on 859  degrees of freedom
#> AIC:  663.69
#> 
#> Type of estimator: MPL_DY (maximum Diaconis-Ylvisaker prior penalized likelihood) with alpha = 0.88
#> Number of Fisher Scoring iterations: 6
```

In low-dimensional setting when the high dimensional correction
(`hd_correction`) is not required,
$\widehat{\boldsymbol{\beta}}_{\text{DY}}$ behaves as a ML estimator and
therefore the quantities in the `summary` above such as
$\text{SE}(\widehat{\boldsymbol{\beta}}_{\text{DY}})$ the z values and
the p-values are computed in the classical way. On the contrary the
summary will be different when including the `hd_correction`.

``` r
# Regression coefficients of the full model (Fourier and PCA), intercept included
b_fk <- coef(mod_fk) 
ones <- rep(1,1000)
```

`b_fk` corresponds to the `Estimate` column in the `summary`

Linear predictor (important to take in to account the intercept column)

``` r
eta_fk <- cbind(ones,X_tr_std)%*%b_fk  
```

It’s worth noting that especially in an high-dimensional context, the
shrinkage property of the hyperparameter $\alpha$ of the DY prior makes
the estimated regression coefficients
$\widehat{\boldsymbol{\beta}}_{\text{DY}}$ biased away from their real
values $\boldsymbol{\beta}$. Therefore they will need to be corrected
using the solution of the state evolution system.

We also estimate a model with only the Fourier coefficients and the
intercept

``` r
# MDYPL logistic regression: Intercept + Fourier
mod_f <- glm(y_tr ~ X_tr_std[,1:76], family = binomial(), method = "mdyplFit") 
# regression coefficients
b_f <- coef(mod_f)
# linear predictor
eta_f <-  cbind(ones,X_tr_std[,1:76])%*%b_f 
```

Inverse logit function used to compute predicted probabilities

``` r
link <- function(x){
  1/(1+exp(-x))
}
```

Predicted probabilities on the training set for all 1000 images using
the model with Fourier and Karhunen-Loeve as explanatory variables and
then only Fourier.

``` r
probs_fk <- link(eta_fk)
probs_f <- link(eta_f)
```

Equivalently they could have been computed using the the built-in
function `fitted()`.

``` r
fitted_prob_fk <- fitted(mod_fk)
fitted_prob_f <- fitted(mod_f)
```

Checking whether `fitted()` works as expected

``` r
all(round(probs_fk,12)==round(fitted_prob_fk,12))
#> [1] TRUE
all(round(probs_f,12)==round(fitted_prob_f,12))
#> [1] TRUE
```

## Graph of predicted probabilities vs statistical units clustered by labels

``` r
size<-0.7 # font size
my_colors <- c("green2","red","gold","purple","grey50","darkorange","darkblue","brown","blue","tomato")
```

We create a function that plots predicted probability vs statistical
units both for the training and test set.

``` r
pp <- function(probs,pix,df,title){
  plot(c(1:1000), probs, type = "n",main=title,xlab="statistical units",ylab="Predicted probabilities")
  
  for(i in 0:9){
    indices <- which(pix==i)
    text(indices,probs[indices], labels = t(df[indices,1]), cex = size, col = my_colors[i+1])
  }
  abline(h=0.5,col="grey40",lty="dashed",lwd=1.5)
}
```

**MDYPL predicted probabilities using Fourier coefficients (training)**

``` r
pp(probs_f,pix_tr,train,"MDYPL logistic regression (Fourier)")
```

![](Vignette_brglm2_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

**MDYPL predicted probabilities using Fourier and Karhunen-Loeve
coefficients (training)**

``` r
pp(probs_fk,pix_tr,train,"MDYPL logistic regression (Fourier and PCA)")
```

![](Vignette_brglm2_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

We can observe that while the full model classifies the digit 7
perfectly, the model using only Fourier coefficients does not. It
assigns high predicted probabilities of being a 7 to images that are
actually 1s, 2s, and 4s. This is likely due to the similar patterns
among these four digits, which result in similar feature values and make
it difficult to distinguish between them.

**Rescaled mdypl predicted probabilities (Fourier + Karhunen-Loeve
coefficients)**

**Hd correction details:**

In order to rescale the estimates we need to solve the state evolution
system derived from [**Approximate Message
Passing**](https://arxiv.org/abs/2105.02180) (AMP) theory. We do this by
specifying inside the `summary()` argument `hd_correction = TRUE`, this
option solves the SE system and returns the corrected DY estimates of
the regression coefficients, corrected z values, standard errors and
p-values. In this case since we have included an intercept, the state
evolution system to be solved has 4 equations in 4 unknowns
$(\mu,b,\sigma,\iota)$. The SE system in order to be solved requires
three parameter to be inputed $(k,\gamma,\theta_0)$. $k$ is estimated
with $p/n$. While since we do not know the value of the signal strength
$\gamma$ we approximate it with a known transformation of the **SLOE**
estimator [S. Yadlowsky et al. 2021](https://arxiv.org/abs/2103.12725)
of the corrupted signal strenght. This operation is done by specifying
`corrupted = TRUE`. Moreover, we supply to $\iota$ the estimated
intercept $\hat{\theta_0}^{\text{DY}}$ and solve the system with respect
to $(\mu,b,\sigma,\theta_0)$. This approach turns out to be more
accurate, than supplying $\theta_0$ and solve the system with respect of
$\iota$. On the contrary in the case of `corrupted = FALSE` (and
presence of an intercept) both the oracle value of the intercept and the
true signal strength $\gamma$ have to be supplied. In that case the SE
system is solved with respect to $(\mu,b,\sigma,\iota)$.

``` r
# The summary function computes the rescaling and solves the state evolution system
(summ_res <- summary(mod_fk,hd_correction=TRUE,corrupted=TRUE))
#> 
#> Call:
#> glm(formula = y_tr ~ X_tr_std, family = binomial(), method = "mdyplFit")
#> 
#> Deviance Residuals: 
#>      Min        1Q    Median        3Q       Max  
#> -1.02673  -0.12115  -0.06795  -0.03163   0.37986  
#> 
#> Coefficients:
#>                Estimate Std. Error z value Pr(>|z|)   
#> (Intercept)   -4.791772         NA      NA       NA   
#> X_tr_stdfou1   0.563662   0.360084   1.565   0.1175   
#> X_tr_stdfou2  -0.226492   0.418043  -0.542   0.5880   
#> X_tr_stdfou3   0.761566   0.324401   2.348   0.0189 * 
#> X_tr_stdfou4   0.463240   0.244679   1.893   0.0583 . 
#> X_tr_stdfou5  -0.368217   0.465631  -0.791   0.4291   
#> X_tr_stdfou6  -0.412143   0.300083  -1.373   0.1696   
#> X_tr_stdfou7   0.231727   0.431727   0.537   0.5914   
#> X_tr_stdfou8  -0.251255   0.359864  -0.698   0.4851   
#> X_tr_stdfou9   0.396013   0.298392   1.327   0.1845   
#> X_tr_stdfou10  0.118283   0.337137   0.351   0.7257   
#> X_tr_stdfou11  0.275010   0.327853   0.839   0.4016   
#> X_tr_stdfou12  0.028844   0.272857   0.106   0.9158   
#> X_tr_stdfou13 -0.109064   0.304926  -0.358   0.7206   
#> X_tr_stdfou14 -0.091289   0.317998  -0.287   0.7741   
#> X_tr_stdfou15  0.037037   0.271181   0.137   0.8914   
#> X_tr_stdfou16  0.196696   0.267243   0.736   0.4617   
#> X_tr_stdfou17 -0.126434   0.261443  -0.484   0.6287   
#> X_tr_stdfou18 -0.402068   0.285318  -1.409   0.1588   
#> X_tr_stdfou19 -0.040806   0.227465  -0.179   0.8576   
#> X_tr_stdfou20 -0.040052   0.273156  -0.147   0.8834   
#> X_tr_stdfou21  0.082792   0.245199   0.338   0.7356   
#> X_tr_stdfou22 -0.014473   0.241198  -0.060   0.9522   
#> X_tr_stdfou23 -0.129947   0.258650  -0.502   0.6154   
#> X_tr_stdfou24  0.151534   0.248044   0.611   0.5413   
#> X_tr_stdfou25  0.015947   0.248478   0.064   0.9488   
#> X_tr_stdfou26 -0.124128   0.264125  -0.470   0.6384   
#> X_tr_stdfou27  0.270770   0.238734   1.134   0.2567   
#> X_tr_stdfou28  0.053246   0.234766   0.227   0.8206   
#> X_tr_stdfou29  0.146525   0.223881   0.654   0.5128   
#> X_tr_stdfou30 -0.141075   0.260458  -0.542   0.5881   
#> X_tr_stdfou31 -0.082025   0.291332  -0.282   0.7783   
#> X_tr_stdfou32  0.062237   0.242362   0.257   0.7973   
#> X_tr_stdfou33  0.108326   0.250811   0.432   0.6658   
#> X_tr_stdfou34 -0.138096   0.223573  -0.618   0.5368   
#> X_tr_stdfou35  0.215123   0.252448   0.852   0.3941   
#> X_tr_stdfou36  0.043705   0.252936   0.173   0.8628   
#> X_tr_stdfou37 -0.042675   0.242627  -0.176   0.8604   
#> X_tr_stdfou38 -0.126325   0.239154  -0.528   0.5973   
#> X_tr_stdfou39  0.027507   0.225144   0.122   0.9028   
#> X_tr_stdfou40 -0.124610   0.260418  -0.478   0.6323   
#> X_tr_stdfou41 -0.210808   0.245047  -0.860   0.3896   
#> X_tr_stdfou42  0.042300   0.229354   0.184   0.8537   
#> X_tr_stdfou43 -0.066102   0.216789  -0.305   0.7604   
#> X_tr_stdfou44  0.102411   0.218869   0.468   0.6398   
#> X_tr_stdfou45 -0.220819   0.253715  -0.870   0.3841   
#> X_tr_stdfou46 -0.013711   0.230623  -0.059   0.9526   
#> X_tr_stdfou47 -0.041763   0.242550  -0.172   0.8633   
#> X_tr_stdfou48 -0.078902   0.230474  -0.342   0.7321   
#> X_tr_stdfou49 -0.013689   0.252185  -0.054   0.9567   
#> X_tr_stdfou50 -0.069704   0.254177  -0.274   0.7839   
#> X_tr_stdfou51 -0.027767   0.261458  -0.106   0.9154   
#> X_tr_stdfou52  0.017560   0.239446   0.073   0.9415   
#> X_tr_stdfou53  0.363753   0.272568   1.335   0.1820   
#> X_tr_stdfou54  0.035984   0.265090   0.136   0.8920   
#> X_tr_stdfou55  0.108561   0.272225   0.399   0.6900   
#> X_tr_stdfou56 -0.089067   0.232910  -0.382   0.7022   
#> X_tr_stdfou57 -0.060920   0.284262  -0.214   0.8303   
#> X_tr_stdfou58 -0.070175   0.271698  -0.258   0.7962   
#> X_tr_stdfou59 -0.097903   0.261203  -0.375   0.7078   
#> X_tr_stdfou60  0.069462   0.262752   0.264   0.7915   
#> X_tr_stdfou61  0.131083   0.270510   0.485   0.6280   
#> X_tr_stdfou62  0.099332   0.249913   0.397   0.6910   
#> X_tr_stdfou63  0.038418   0.299697   0.128   0.8980   
#> X_tr_stdfou64  0.091087   0.250133   0.364   0.7157   
#> X_tr_stdfou65  0.076978   0.268156   0.287   0.7741   
#> X_tr_stdfou66  0.172996   0.283114   0.611   0.5412   
#> X_tr_stdfou67  0.014215   0.273542   0.052   0.9586   
#> X_tr_stdfou68  0.229907   0.252937   0.909   0.3634   
#> X_tr_stdfou69  0.064739   0.342164   0.189   0.8499   
#> X_tr_stdfou70  0.251141   0.311033   0.807   0.4194   
#> X_tr_stdfou71  0.313859   0.332961   0.943   0.3459   
#> X_tr_stdfou72  0.351941   0.312744   1.125   0.2604   
#> X_tr_stdfou73  0.620007   0.375972   1.649   0.0991 . 
#> X_tr_stdfou74  0.566102   0.382465   1.480   0.1388   
#> X_tr_stdfou75 -0.095226   0.260885  -0.365   0.7151   
#> X_tr_stdfou76  0.179985   0.354314   0.508   0.6115   
#> X_tr_stdkar1   1.332668   0.915855   1.455   0.1456   
#> X_tr_stdkar2  -0.798502   1.167568  -0.684   0.4940   
#> X_tr_stdkar3   0.679032   0.613516   1.107   0.2684   
#> X_tr_stdkar4   0.416894   0.500636   0.833   0.4050   
#> X_tr_stdkar5  -0.655180   0.599872  -1.092   0.2747   
#> X_tr_stdkar6   0.025020   0.484882   0.052   0.9588   
#> X_tr_stdkar7   0.083604   0.408803   0.205   0.8380   
#> X_tr_stdkar8  -0.354072   0.329278  -1.075   0.2822   
#> X_tr_stdkar9   0.861958   0.576889   1.494   0.1351   
#> X_tr_stdkar10 -0.229944   0.361343  -0.636   0.5245   
#> X_tr_stdkar11  0.093564   0.380178   0.246   0.8056   
#> X_tr_stdkar12 -0.433241   0.320916  -1.350   0.1770   
#> X_tr_stdkar13 -0.138842   0.443881  -0.313   0.7544   
#> X_tr_stdkar14 -0.258047   0.292628  -0.882   0.3779   
#> X_tr_stdkar15 -0.605200   0.411832  -1.470   0.1417   
#> X_tr_stdkar16  0.178639   0.288240   0.620   0.5354   
#> X_tr_stdkar17  0.386637   0.331927   1.165   0.2441   
#> X_tr_stdkar18 -0.250920   0.600576  -0.418   0.6761   
#> X_tr_stdkar19  0.395517   0.303422   1.304   0.1924   
#> X_tr_stdkar20  0.504948   0.432886   1.166   0.2434   
#> X_tr_stdkar21  0.248128   0.561078   0.442   0.6583   
#> X_tr_stdkar22  0.327247   0.565457   0.579   0.5628   
#> X_tr_stdkar23  1.010506   0.321979   3.138   0.0017 **
#> X_tr_stdkar24  0.123507   0.335520   0.368   0.7128   
#> X_tr_stdkar25  0.616867   0.322705   1.912   0.0559 . 
#> X_tr_stdkar26  0.149077   0.288624   0.517   0.6055   
#> X_tr_stdkar27 -0.372861   0.288953  -1.290   0.1969   
#> X_tr_stdkar28 -0.026441   0.389550  -0.068   0.9459   
#> X_tr_stdkar29  0.005032   0.437012   0.012   0.9908   
#> X_tr_stdkar30  0.333883   0.340372   0.981   0.3266   
#> X_tr_stdkar31 -0.259451   0.483334  -0.537   0.5914   
#> X_tr_stdkar32 -0.135844   0.305250  -0.445   0.6563   
#> X_tr_stdkar33  0.113105   0.291335   0.388   0.6978   
#> X_tr_stdkar34 -0.025012   0.265017  -0.094   0.9248   
#> X_tr_stdkar35  0.077233   0.305657   0.253   0.8005   
#> X_tr_stdkar36  0.585154   0.376070   1.556   0.1197   
#> X_tr_stdkar37  0.313595   0.272225   1.152   0.2493   
#> X_tr_stdkar38  0.226393   0.284956   0.794   0.4269   
#> X_tr_stdkar39 -0.340262   0.239264  -1.422   0.1550   
#> X_tr_stdkar40  0.239026   0.294658   0.811   0.4173   
#> X_tr_stdkar41  0.017227   0.263348   0.065   0.9478   
#> X_tr_stdkar42  0.123559   0.306498   0.403   0.6869   
#> X_tr_stdkar43 -0.252234   0.355286  -0.710   0.4777   
#> X_tr_stdkar44  0.381355   0.370533   1.029   0.3034   
#> X_tr_stdkar45 -0.397337   0.252476  -1.574   0.1155   
#> X_tr_stdkar46  0.302636   0.220500   1.373   0.1699   
#> X_tr_stdkar47  0.048259   0.253715   0.190   0.8491   
#> X_tr_stdkar48  0.007410   0.237991   0.031   0.9752   
#> X_tr_stdkar49  0.264013   0.272704   0.968   0.3330   
#> X_tr_stdkar50  0.044362   0.245314   0.181   0.8565   
#> X_tr_stdkar51  0.226721   0.226193   1.002   0.3162   
#> X_tr_stdkar52  0.046629   0.244701   0.191   0.8489   
#> X_tr_stdkar53  0.331505   0.271233   1.222   0.2216   
#> X_tr_stdkar54  0.086151   0.229397   0.376   0.7072   
#> X_tr_stdkar55 -0.103932   0.230400  -0.451   0.6519   
#> X_tr_stdkar56  0.117085   0.245303   0.477   0.6331   
#> X_tr_stdkar57 -0.079203   0.243227  -0.326   0.7447   
#> X_tr_stdkar58  0.068727   0.240662   0.286   0.7752   
#> X_tr_stdkar59  0.363389   0.205401   1.769   0.0769 . 
#> X_tr_stdkar60 -0.148744   0.251258  -0.592   0.5539   
#> X_tr_stdkar61 -0.138631   0.213248  -0.650   0.5156   
#> X_tr_stdkar62 -0.019590   0.225923  -0.087   0.9309   
#> X_tr_stdkar63  0.047220   0.213262   0.221   0.8248   
#> X_tr_stdkar64 -0.001741   0.215032  -0.008   0.9935   
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> (Dispersion parameter for binomial family taken to be 1)
#> 
#>     Null deviance: 618.85  on 999  degrees of freedom
#> Residual deviance:  21.78  on 859  degrees of freedom
#> AIC:  925.38
#> 
#> Type of estimator: MPL_DY (maximum Diaconis-Ylvisaker prior penalized likelihood) with alpha = 0.88
#> Number of Fisher Scoring iterations: 6
#> 
#> High-dimensionality correction applied with
#> Dimentionality parameter (kappa) = 0.14
#> Estimated signal strength (gamma) = 11.71
#> State evolution parameters (mu, b, sigma) = (0.4, 1.83, 2.2) with max(|funcs|) = 5.630178e-09
```

``` r
# Rescaled regression coeff according to the solution of SE system
b_fk_res <- coef(summ_res)[,"Estimate"]
```

As we can observe this summary is different from the one without
correction and the reason lies in the adjustments performed using the
solution of the state evolution system. The difference can be seen in
the following graphs, where the intercept has been removed,due to its
different order of magnitude, in order to allow for a meaningful
comparison:

![](Vignette_brglm2_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->![](Vignette_brglm2_files/figure-gfm/unnamed-chunk-24-2.png)<!-- -->

``` r
# scaled linear predictor
eta_fk_res <- cbind(ones,X_tr_std)%*%b_fk_res  
# scaled linear predictor
probs_fk_res <- link(eta_fk_res)
```

**MDYPL predicted probabilities using Fourier and Karhunen-Loeve
coefficients (training)**

``` r
pp(probs_fk_res,pix_tr,train,"rescaled MDYPL (Fourier and PCA)")
```

![](Vignette_brglm2_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

## TEST SET

``` r
# design matrix of the test set (removed labels)
X_ts <- as.matrix(test[,-c(1,2)])
# response in the test set (used only to measure the accuracy of the classification)
y_ts <- test[,2]
```

The standardization of the test design matrix is performed after the
splitting phase in order not to include information from the training
set.

``` r
X_ts_std <- scale(X_ts,center=TRUE,scale=TRUE)
```

Linear predictors of the model with both subsets of covariates
(**Fourier** and **PCA**) and with and without rescaling and only
**Fourier** no rescaling.

``` r
eta_fk_res <- cbind(ones,X_ts_std)%*%b_fk_res
eta_fk <- cbind(ones,X_ts_std)%*%b_fk  
eta_f <-  cbind(ones,X_ts_std[,1:76])%*%b_f 
```

Predicted probabilities

``` r
probs_fk_res <- link(eta_fk_res)
probs_fk <- link(eta_fk)
probs_f <- link(eta_f)
```

**MDYPL predicted probabilities using Fourier coefficients (test)**

``` r
pp(probs_f,pix_ts,test,"MDYPL test (Fourier)")
```

![](Vignette_brglm2_files/figure-gfm/unnamed-chunk-31-1.png)<!-- -->
**MDYPL predicted probabilities using Fourier and Karhunen-Loeve
coefficients (test)**

``` r
pp(probs_fk,pix_ts,test,"MDYPL test (Fourier and PCA)")
```

![](Vignette_brglm2_files/figure-gfm/unnamed-chunk-32-1.png)<!-- -->

**MDYPL predicted probabilities using Fourier and Karhunen-Loeve
coefficients rescaled signal(test)**

``` r
pp(probs_fk_res,pix_ts,test,"MDYPL test (Fourier and PCA)")
```

![](Vignette_brglm2_files/figure-gfm/unnamed-chunk-33-1.png)<!-- -->

As expected, the overall performance on the test set is worse than that
on the training set. More interestingly, the model that includes both
subsets of covariates performs better in classifying the digit 7 and,
more generally, across all digit classes. Moreover, as one would
anticipate from empirical observation, the classifier has difficulties
distinguishing between the digits 7, 4, and 1, due to their strong
similarity in handwritten form. We also observe that performance
improves when the regression coefficients are rescaled according to the
solution of the SE system $(\mu_{*},b_{*},\sigma_{*},\theta_{0*})$. The
corrected estimates are defined as follows:

$$
\widehat{\boldsymbol{\beta}}^{\text{RES}}_{\text{DY}}= \frac{\widehat{\boldsymbol{\beta}}_{\text{DY}}}{\mu_{*}}
$$

where the nonlinear system is the one specified in section 6 of [P.
Sterzinger, I. Kosmidis](https://arxiv.org/abs/2311.07419).

## Performance comparison

``` r
# predicted labels
pred_lab_fk_res <- ifelse(probs_fk_res>0.5,1,0)
pred_lab_fk <- ifelse(probs_fk>0.5,1,0)
pred_lab_f <- ifelse(probs_f>0.5,1,0)

# accuracy function 
acc <- function(x,y){
  out <- sum(x==y)/length(x)
}
accuracy_fk_res <- acc(y_ts,pred_lab_fk_res)
accuracy_f <- acc(y_ts,pred_lab_f)
accuracy_fk <- acc(y_ts,pred_lab_fk)
```

    #>   res_fk   Fou Fou_Kar
    #> 1  0.995 0.964   0.993

## Likelihood ratio tests

Empirically, the model using both **Fourier** and **Karhunen–Loève**
coefficients outperforms the model that relies solely on Fourier
coefficients. Therefore, in terms of **hypothesis testing** we expect to
find evidence against the fact that the model with only **Fourier**
coefficients is better than the one with both subsets of features.

Computation of Likelihood ratio statistics

``` r
k <- dim(X_tr)[2]/dim(X_tr)[1]
alpha <- 1/(1+k)

mod_f <- glm(y_tr ~ X_tr_std[,1:76],family=binomial(),method="mdyplFit",alpha=alpha)
mod_fk <- glm(y_tr ~ X_tr_std,family=binomial(),method="mdyplFit",alpha=alpha)

(PLR <- 2*(as.numeric(logLik(mod_fk))-as.numeric(logLik(mod_f))))
#> [1] 64.35935
```

**Penalized Likelihood Ratio test statistics (PLR)**

The PLR has the following form: 

$$
\Lambda_{I}= \underset{\boldsymbol{\beta}\in\mathbb{R}^p}{\sup}\ l(\boldsymbol{\beta};\mathbf{y},\mathbf{X})-\underset{\beta_j=0\ ,\ j\in I}{\underset{\boldsymbol{\beta}\in\mathbb{R}^p}{\sup}}\ l(\boldsymbol{\beta};\mathbf{y},\mathbf{X})
$$

The Hypotesis test we are interested in is the following 

$$
H_0: \Lambda_{\text{pop}}=0\qquad\text{vs}\qquad H_1: \Lambda_{\text{pop}}\neq0
$$

An alternative way of writing the same Hypothesis test system is the
following: 

$$
H_0: \boldsymbol{\beta}\in \{\boldsymbol{\beta}\in\mathbb{R}^p\ |\ \beta_j=0\ \ \forall j\in I\} \qquad\text{vs}\qquad H_1: \boldsymbol{\beta}\in\mathbb{R}^p
$$ 

with $I$ the set of indices associated with Karhunen-Loeve
coefficients. In other words we are comparing the two nested models to
understand according to data which is more plausible. As with every
hypothesis testing procedure we arbitrarily fix a probability of
committing a type I error or significance level $\alpha=0.05$. Obviously
the significance level has nothing to do with the shrinkage
hyperparamater of the DY prior.

We know from **Wilks’ theorem** that for the Penalized likelihood ratio
test statistics (PLR) the following result holds: 

$$
2\Lambda_{I}\sim\chi^2_{p-r}
$$ 

with $\chi^2_{p-r}$ being a Chi-squared distribution with$p-r$
degrees of freedom; while $p$ is the total number of dimensions and $r$
the number of constraints. For a regression model $k=p-r$ is the
difference between the number of explanatory variables in the full and
nested model. In our case $k=p-r=64$. By computing the p-value we
discover that the PLR suggests not to reject $H_0$ which seems in stark
contradiction with empirical evidence

``` r
(pval <- pchisq(q=PLR,df=64,lower.tail=FALSE))
#> [1] 0.4638874
```

the p-value is way above any meaningful significance level, therefore we
accept $H_0$.

**Rescaled PLR**

According to the theory detailed in *Theorem 3.5* of [P. Sterzinger, I.
Kosmidis](https://arxiv.org/abs/2311.07419), we compute the full model
and solve the SE system in order to find the parameters to adjust PLR.

``` r
summ_fk <- summary(mod_fk,hd_correction=TRUE,corrupted=TRUE)

se_pars <- summ_fk$se_parameters
# adjusting parameters
b <- se_pars[2] 
sigma <- se_pars[3]

(rPLR <- PLR*b/(0.14*sigma^2))
#> [1] 173.3375
```

we have adjusted the classical penalized likelihood ratio test
statistics according to the following result: $$
2\Lambda_{I}\overset{d}{\longrightarrow}\frac{k\sigma^2_{*}}{b_{*}}\chi^2_k
$$ Consequently we are able to compute a new adjusted `p-value` to
perform the same hypothesis test as before

``` r
(pval_r <- pchisq(q=rPLR,df=64,lower.tail=FALSE))
#> [1] 5.094556e-12
```

This time since the adjusted `p-value` is way below the significance
level we have strong evidence against the null hypothesis and we are
able to reject it. This result is way more coherent with what the
empirical evidence shows.

**Performance of the rescaled PRL (rPLR)**

The performance of `rPLR` is assessed by simulating 500 times the
response vector for each combination of the following intercept and
signal strenght:

``` r
int <- c(-3,-2,-1,0)  # intercept values
ss2 <- c(1,2,4,8,16)  # signal strenght values

grid <- expand.grid(int,ss2)
n_set <- dim(grid)[1] # number of different settings

X_fk <- scale(X_tr,center=TRUE,scale=FALSE)
X_fki <- cbind(ones,X_fk)
k <- dim(X_fk)[2]/dim(X_fk)[1]
alpha <- 1/(1+k)
```

Each row is a sample of 500 `PLR` for a specific combination of
intercept and signal strength. The same is repeated for `rPLR`. The
details about this simulation are provided in the last paragraph of
section 8 of [P. Sterzinger, I.
Kosmidis](https://arxiv.org/abs/2311.07419).

``` r
PLR <- matrix(data=NA,ncol=500,nrow=20)
rPLR <- matrix(data=NA,ncol=500,nrow=20)

for(t in 1:n_set){
  
  inter <- grid[t,1]
  ss2 <- grid [t,2]

  se_pars <- try(solve_se(kappa = k, ss = sqrt(ss2), alpha = alpha,
                          intercept = inter,
                          start = c(0.5,1,1,inter),
                          corrupted = FALSE, gh = NULL, prox_tol = 1e-10,
                          transform = TRUE, init_method = "Nelder-Mead",
                          init_iter = 15), silent = FALSE)


  b <- se_pars[2]
  sigma <- se_pars[3]

  for(i in 1:500){
  b_fk <- c(rnorm(76,0,1),rep(0,64))
  b_fks <- sqrt(ss2) * b_fk /sd(X_fk %*% b_fk)
  b_fksi <- c(inter,b_fks)

  probs <- plogis(drop(X_fki%*%b_fksi))
  y <- rbinom(1000, 1, probs)
  
  full <- glm(y ~ X_fk,family = binomial(),method="mdyplFit",alpha=alpha)
  nest <- glm(y ~ X_fk[,1:76],family = binomial(),method="mdyplFit",alpha=alpha)
  
  PLR[t,i] <- 2*(as.numeric(logLik(full))-as.numeric(logLik(nest)))
  rPLR[t,i] <- PLR[t,i]*b/(k*sigma^2)
  }
}
```

**QQplot table for different combinations of intercept and signal
strenght**

``` r
par(mfrow=c(5,4), 
    mar=c(1,1,1,1),  
    oma=c(1,1,1,8))   

th_q <- qchisq(ppoints(500),df=64) # Theoretical quantiles

for(i in 1:20){
  yy <- as.numeric(rPLR[i,])
  qqplot(th_q,yy,ylim = c(0,125),cex.axis=0.7,cex=0.95,lwd=0.5)
  points(th_q,sort(as.numeric(PLR[i,])),col="red",cex=0.95,lwd=0.5)
  abline(a=0,b=1,lty="dashed",col="gold",lwd=2.5)
  b0 <- grid[i,1]
  ss2 <- grid[i,2]
  legend(x=25,y=135, cex = 1,
       legend = c(
         bquote(beta[0] == .(b0)),
         bquote(gamma^2 == .(ss2))
       ),bty="n")
}
mtext("QQplot comparison PLR and rescaled PLR", outer=TRUE, cex=0.9)
legend("topright", inset=c(-1.2,-2), legend=c("PLR", "rescaled PLR","identity line"),
       col=c("red","black","gold"), pch=c(1,1,NA),
       lty=c(NA, NA, 2),xpd=NA,cex=0.9)
```

![](Vignette_brglm2_files/figure-gfm/unnamed-chunk-43-1.png)<!-- -->

As we can see, applying the correction suggested by Theorem 3.5 restores
the result of Wilks’s theorem: the corrected penalized likelihood ratio
statistic follows a Chi-squared distribution with 64 degrees of freedom,
corresponding to the difference between the total number of parameters
and those constrained to be zero. It is also evident that as the
intercept and signal strength increase, the correction becomes
increasingly important, since the two QQ plots diverge more markedly.








