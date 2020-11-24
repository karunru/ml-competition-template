import cudf
import cupy as cp
import numpy as np
from scipy import stats

# https://www.guruguru.ml/competitions/10/discussions/c26ab5b4-2646-49ca-b040-e014cb68a1ef/
# https://github.com/Anguschang582/Probspace---Re_estate---1st-place-solution/blob/master/function.R


def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))


# mean_var <- function(x){ return( sd(x,na.rm=T) /  mean(x,na.rm=T) )}
def mean_variance(x):
    return np.nanstd(x) / np.nanmean(x)


# hl_ratio <- function(x){ return( sum(ifelse(x > mean(x),1,0)) / sum(ifelse(x >= mean(x),0,1)) )}
def hl_ratio(x):
    return np.sum(x >= np.nanmean(x)) / np.sum(x < np.nanmean(x))


# beyond1std_ratio <- function(x){ return( sum(ifelse(x > (mean(x,na.rm=T) + sd(x,na.rm=T)),1,0)) / length(x) )}
def beyond1std_ratio(x):
    return np.sum(x >= (np.nanmean(x) + np.nanstd(x))) / len(x)


# iqr_ratio <- function(x){ return( quantile(x,0.75,na.rm=T) / quantile(x,0.25,na.rm=T) )}
def iqr_ratio(x):
    return np.quantile(x, 0.75) / np.quantile(x, 0.25)


# range_diff <- function(x){ return( max(x,na.rm=T) - min(x,na.rm=T) )}
def range_diff(x):
    return np.nanmax(x) - np.nanmin(x)


# range_per <- function(x){ return( max(x,na.rm=T) / min(x,na.rm=T) )}
def range_per(x):
    return np.nanmax(x) / np.nanmin(x)


# sw_stat <- function(x){ return( ifelse(sum(is.na(x) == F) > 3 & sum(is.na(x) == F) < 5000 &
#                                          length(x) > 3 & length(x) < 5000 & sum(diff(x),na.rm=T)!=0 ,
#                                        shapiro.test(as.numeric(x))$statistic, NA) )}
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
def sw_stat(x):
    return stats.shapiro(x)[0] if len(x) > 3 else np.nan


# x_diff <- function(x){ return(x - mean(x,na.rm=T)) }
def x_diff(x):
    return x - np.nanmean(x)


# x_ratio <- function(x){ return(x / mean(x,na.rm=T)) }
def x_ratio(x):
    return x / np.nanmean(x)


# x_zscore <- function(x){ return( (x-mean(x,na.rm=T)) / sd(x,na.rm=T)) }
def x_zscore(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


# freq1ratio <- function(x){ return(
#   ifelse(sort(table(x),decreasing = T)[1] == sort(table(x),decreasing = T)[2],
#          NA, as.numeric( sort(table(x),decreasing = T)[1] / length(x) )  )  )}


# freq1count <- function(x){ return(
#   ifelse(sort(table(x),decreasing = T)[1] == sort(table(x),decreasing = T)[2],
#          NA, ( sort(table(x),decreasing = T)[1] )  )  )}
#
# entropy_freqtable <- function(x){ return( as.numeric(entropy(table(x)))) }
