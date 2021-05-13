# Trending-Event-Detection-for-COVID-19-Twitter-chatter

## blog post.
$r$

https://yangzhengjie33.medium.com/trending-event-detection-using-anomaly-detection-for-covid-19-twitter-chatter-8029254ee659

## A list of the python libraries used in the project.

### Environment Setup

Please use Python3.6+, and you should have following Python library installed:
- numpy
- scipy
- statsmodels
- NLTK
- matplotlib

## The motivation for the project. Why did you decide to work on the data and topic?

We want to work on the twitter data to find the trending words related to twitter and when the trending happened? The intuition comes from the dataset from https://zenodo.org/record/4748942#.YJyKmWZKi3I and the twitter paper Automatic Anomaly Detection in the Cloud Via Statistical Learning.

## An explanation of what each file in your repository does.

The folder called dataset and our data is organized by date in it (from **2020-03-22** to **2020-08-22**), within each subfolder, you should be able to find following files:

|Filename|Description|
|---|---|
|DATE_top1000bigrams.csv | Top 1000 bi-grams ordered by counts from the tweets |
|DATE_top1000terms.csv   | Top 1000 terms ordered by counts from the tweets |
|DATE_top1000trigrams.csv| Top 1000 tri-grams ordered by counts from the tweets  |


## A summary of the results you got from the project.

### Robust Extreme Studentized Deviate (ESD) Test Algorithm

#### Test statistics
In Robust ESD test, we first define the maximum number of anomalies $K$, and then compute the following test statistic for the $k=1,2,...,K$ most extreme values in the data set:

$$
C_k=\frac{\max_k |x_k-\tilde{x}|}{\hat{\sigma}}
$$

where $\tilde{x}$ is the median of the series and $\hat{\sigma}$ is the median absolute deviation (MAD), which is defined as $\text{MAD} = \text{median}(|X-\text{median}(X)|)$.

Tips: to calcuate MAD, you can use the mad function from statsmodels (i.e., `from statsmodels.robust.scale import mad`)

#### Critical value
The test statistic is then used to compare with a critical value, which is computed using following equation:

$$
\lambda_k=\frac{(n-k) t_{p,\ n-k-1}}{\sqrt{(n-k-1+t_{p,\ n-k-1}^2)(n-k+1)}}
$$

Note here to compute the $t_{p,\ n-k-1}$ in critical value, you can use following code

```python
from scipy.stats import t as student_t
p = 1 - alpha / (2 * (N - k + 1))
t = student_t.ppf(p, N - k - 1)
```

#### Anomaly direction
The another thing we need to do is to determine the direction or the anomaly (i.e., the anomaly is going up or going down), this information is useful because in some anomaly detection task we may only care anomaly goes to only one direction (e.g., for the error count or error rate, we only care if it increases a lot).

To determine the direction of anomaly, we can use the sign of the $|x_k-\tilde{x}|$ in $C_k$. If sign is positive, then the anomaly is greater than median and it's very likely to be a spike, otherwise, it should be a dip.

## Acknowledgment of external sources

The dataset A large-scale COVID-19 Twitter chatter dataset for open scientific research - an international collaboration from https://zenodo.org/record/4748942#.YJyKmWZKi3I.

The twitter paper Automatic Anomaly Detection in the Cloud Via Statistical Learning.


