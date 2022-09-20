---
title: "NFW Distribution Sampling"
output: github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## GitHub Documents

This package lets you sample from the NFW as a true PDF with the only variable being the concerntration (con in the package).

## Load

Load with:

```
library(NFWdist)
```

## Checks

Here we see that our random draws (using the **rnfw** via the **qnfw** function) produces the right PDFs (as per **dnfw**):

```
for(con in c(1,5,10,20)){
  plot(density(rnfw(1e6,con=con), bw=0.01))
  lines(seq(0,1,len=1e3), dnfw(seq(0,1,len=1e3),con=con),col='red')
  legend('topright',legend=paste('con =',con))
}
```
