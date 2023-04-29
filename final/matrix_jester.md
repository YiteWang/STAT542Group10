matrix_jester
================
jcrull2
2023-04-16

## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax
for authoring HTML, PDF, and MS Word documents. For more details on
using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that
includes both content as well as the output of any embedded R code
chunks within the document. You can embed an R code chunk like this:

### Import data

``` r
#install.packages("recommenderlab")

library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.2 ──
    ## ✔ ggplot2 3.4.0     ✔ purrr   0.3.4
    ## ✔ tibble  3.1.8     ✔ dplyr   1.0.9
    ## ✔ tidyr   1.2.1     ✔ stringr 1.4.1
    ## ✔ readr   2.1.3     ✔ forcats 0.5.2

    ## Warning: package 'ggplot2' was built under R version 4.2.2

    ## Warning: package 'tidyr' was built under R version 4.2.2

    ## Warning: package 'readr' was built under R version 4.2.2

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
library(recommenderlab)
```

    ## Warning: package 'recommenderlab' was built under R version 4.2.3

    ## Loading required package: Matrix
    ## 
    ## Attaching package: 'Matrix'
    ## 
    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack
    ## 
    ## Loading required package: arules

    ## Warning: package 'arules' was built under R version 4.2.3

    ## 
    ## Attaching package: 'arules'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     recode
    ## 
    ## The following objects are masked from 'package:base':
    ## 
    ##     abbreviate, write
    ## 
    ## Loading required package: proxy

    ## Warning: package 'proxy' was built under R version 4.2.2

    ## 
    ## Attaching package: 'proxy'
    ## 
    ## The following object is masked from 'package:Matrix':
    ## 
    ##     as.matrix
    ## 
    ## The following objects are masked from 'package:stats':
    ## 
    ##     as.dist, dist
    ## 
    ## The following object is masked from 'package:base':
    ## 
    ##     as.matrix
    ## 
    ## Registered S3 methods overwritten by 'registry':
    ##   method               from 
    ##   print.registry_field proxy
    ##   print.registry_entry proxy

``` r
library(softImpute)
```

    ## Warning: package 'softImpute' was built under R version 4.2.3

    ## Loaded softImpute 1.4-1
    ## 
    ## 
    ## Attaching package: 'softImpute'
    ## 
    ## The following object is masked from 'package:tidyr':
    ## 
    ##     complete

``` r
data(Jester5k)
paste0('Number of nonmissing ratings: ',length(getRatingMatrix(Jester5k)) - length(which(getRatingMatrix(Jester5k)!=0)))
```

    ## [1] "Number of nonmissing ratings: 136791"

``` r
paste0('Number of missing ratings: ',length(which(getRatingMatrix(Jester5k)!=0)))
```

    ## [1] "Number of missing ratings: 363209"

### EDA

``` r
data(Jester5k)
hist(x=getRatings(Jester5k),breaks=50,xlab="Rating",main="Histogram of Ratings")
```

![](matrix_jester_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
hist(x=rowMeans(Jester5k),breaks=50,xlab="Rating",main="Histogram of Mean Ratings by Respondent")
```

![](matrix_jester_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->

``` r
hist(x=colMeans(Jester5k),breaks=50,xlab="Rating",main="Histogram of Mean Ratings by Joke")
```

![](matrix_jester_files/figure-gfm/unnamed-chunk-2-3.png)<!-- -->

``` r
hist(x=rowCounts(Jester5k),breaks=50,xlab="Number by Respondent",main="Number of Ratings by Respondent")
```

![](matrix_jester_files/figure-gfm/unnamed-chunk-2-4.png)<!-- -->

``` r
hist(x=colCounts(Jester5k),breaks=50,xlab="Number by Joke",main="Number of Ratings by Joke")
```

![](matrix_jester_files/figure-gfm/unnamed-chunk-2-5.png)<!-- -->

### Split data into train/test rows (90/10), giving 18 ratings per test row to algorithm and evaluate recommenders on remaining ratings from each test respondent

``` r
start.time <- Sys.time()

set.seed(42)
print(paste0("Minimum number of ratings per respondent: ",min(rowCounts(Jester5k))))
```

    ## [1] "Minimum number of ratings per respondent: 36"

``` r
#n_matrix=normalize(real_matrix)
#Algorithms normalize matrix by subtracting row means
#Split data into 80% train/20% test, evaluate on all but three per row in test
e <- evaluationScheme(Jester5k, method="split", train=0.9,given=18, goodRating=3)
r_random <- Recommender(getData(e, "train"), "RANDOM")
r_popular <- Recommender(getData(e, "train"), "POPULAR")
r_ubcf <- Recommender(getData(e, "train"), "UBCF")
r_ibcf <- Recommender(getData(e, "train"), "IBCF")
r_svdf <- Recommender(getData(e, "train"), "SVDF")

p_random <- predict(r_random, getData(e, "known"), type="ratings")
p_popular <- predict(r_popular, getData(e, "known"), type="ratings")
p_ubcf <- predict(r_ubcf, getData(e, "known"), type="ratings")
p_ibcf <- predict(r_ibcf, getData(e, "known"), type="ratings")
p_svdf <- predict(r_svdf, getData(e, "known"), type="ratings")

error <- rbind(RANDOM = calcPredictionAccuracy(p_random, getData(e, "unknown")),POPULAR = calcPredictionAccuracy(p_popular, getData(e, "unknown")),UBCF = calcPredictionAccuracy(p_ubcf, getData(e, "unknown")),IBCF = calcPredictionAccuracy(p_ibcf, getData(e, "unknown")),SVDF = calcPredictionAccuracy(p_svdf, getData(e, "unknown")))
error
```

    ##             RMSE      MSE      MAE
    ## RANDOM  7.916357 62.66871 6.475932
    ## POPULAR 4.496329 20.21698 3.560837
    ## UBCF    4.685868 21.95736 3.680720
    ## IBCF    4.555334 20.75107 3.514664
    ## SVDF    4.838041 23.40664 3.661751

``` r
paste0("RMSE with just the training mean: ",round((mean((getRatings(getData(e,"unknown"))-mean(getRatings(getData(e,"train"))))^2))^.5,3))
```

    ## [1] "RMSE with just the training mean: 5.331"

``` r
end.time <- Sys.time()
time.taken <- end.time - start.time
print(paste0("Runtime: ",round(time.taken,3)," minutes"))
```

    ## [1] "Runtime: 1.607 minutes"

### softImpute (***Randomly setting 27589 obs to NA for evaluation and training on rest***)

``` r
#start.time <- Sys.time()

#print(paste0("Number of values used to calculate RMSE above: ",length(getRatings(getData(e,"unknown")))))
#set.seed(42)
#jester <- getRatingMatrix(Jester5k)
#test_obs = sample(setdiff(1:length(jester),which(jester==0)),27589,replace=FALSE)
#test = jester[test_obs]
#train = jester
#train[which(train==0)] <- NA
#train[test_obs] <- NA
#missing_train = which(is.na(train))
#nonmissing_mean = mean(train[-missing_train])
#train <- as.matrix(train)


#standardized_matrix = train-nonmissing_mean
#fits_als <- softImpute(standardized_matrix, trace=FALSE, type = "als",lambda=0)
#hardImpute_matrix_als = (fits_als$u %*% diag(fits_als$d) %*% t(fits_als$v)) + nonmissing_mean
#fits_svd <- softImpute(standardized_matrix, trace=FALSE, type = "svd",lambda=0)
#hardImpute_matrix_svd = (fits_svd$u %*% diag(fits_svd$d) %*% t(fits_svd$v)) + nonmissing_mean

##Tuning max.rank and lambda for ALS
#best_rmse_als = 69
#best_lambda_als = 0
#best_max_rank_als = 0
#for (max_rank in 2:14){  
#  for (lambda in seq(0,8,0.1)){
#    set.seed(42)
#    fits <- softImpute(standardized_matrix, trace=FALSE, type = "als",lambda=lambda,rank.max=max_rank)
#    new_matrix = (fits$u %*% diag(fits$d) %*% t(fits$v)) + nonmissing_mean
#    rmse = (mean((test-new_matrix[test_obs])^2))^.5
#    if (rmse<best_rmse_als){
#      best_rmse_als = rmse
#      best_lambda_als = lambda
#      best_max_rank_als = max_rank
#    }
#  }
#}

##Tuning max.rank and lambda for SVD
#best_rmse_svd = 69
#best_lambda_svd = 0
#best_max_rank_svd = 0
#for (max_rank in 2:14){  
#  for (lambda in seq(0,8,0.1)){
#    set.seed(42)
#    fits <- softImpute(standardized_matrix, trace=FALSE, type = "svd",lambda=lambda,rank.max=max_rank)
#    new_matrix = (fits$u %*% diag(fits$d) %*% t(fits$v)) + nonmissing_mean
#    rmse = (mean((test-new_matrix[test_obs])^2))^.5
#    #print(paste0("RMSE with lambda=",lambda," and max rank = ",max_rank,": ",round(rmse)))
#    if (rmse<best_rmse_svd){
#      best_rmse_svd = rmse
#      best_lambda_svd = lambda
#      best_max_rank_svd = max_rank
#    }
#  }
#}

#paste0("RMSE with just the mean: ",round((mean((test-nonmissing_mean)^2))^.5,3))
#print(paste0("RMSE with ALS hardImpute: ",round((mean((test-hardImpute_matrix_als[test_obs])^2))^.5,3)))
#print(paste0("RMSE with SVD hardImpute: ",round((mean((test-hardImpute_matrix_svd[test_obs])^2))^.5,3)))

#print(paste0("Best parameters for ALS: lambda=",best_lambda_als," ,max.rank=",best_max_rank_als))
#print(paste0("RMSE with tuned softImpute with ALS: ",round(best_rmse_als,3)))

#print(paste0("Best parameters for SVD: lambda=",best_lambda_svd," ,max.rank=",best_max_rank_svd))
#print(paste0("RMSE with tuned softImpute with SVD: ",round(best_rmse_svd,3)))

#end.time <- Sys.time()
#time.taken <- end.time - start.time
#print(paste0("Runtime: ",round(time.taken,3)," minutes"))
```

Output of the above:

\[1\] “RMSE with just the mean: 5.226”

\[1\] “RMSE with ALS hardImpute: 4.268”

\[1\] “RMSE with SVD hardImpute: 4.268”

\[1\] “Best parameters for ALS: lambda=8 ,max.rank=6”

\[1\] “RMSE with tuned softImpute with ALS: 4.114”

\[1\] “Best parameters for SVD: lambda=8 ,max.rank=6”

\[1\] “RMSE with tuned softImpute with SVD: 4.115”

\[1\] “Runtime: 3.562 hours”

Try larger lambda
