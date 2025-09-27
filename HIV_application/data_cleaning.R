# Experiment on real data: HIV-1 resistance


# Our attention is focused on HIV 1 resistance to Protease inhibitors, in particular
# to the antiviral NFV (Nelfinavir).
# link for the matrix predictors and responses (matrix of positions of the mutations and drug resistance measurements
# for 7 protease inhibitors) https://hivdb.stanford.edu/_wrapper/pages/published_analysis/genophenoPNAS2006/DATA/PI_DATA.txt

# link for the TSM list containing relevant mutation of the HIV-1 protease regardless of the specific protein inhibitor used
# https://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/MUTATIONLISTS/NP_TSM/PI


rm(list = setdiff(ls(), c("PI_data", "PI_TSM")))

setwd("C:\\Users\\andre\\Desktop\\MDYPL_R_warwick\\HIV_mdypl")

# position selected by treatement selected mutation (approximately the ground truth)
pos <- PI_TSM$V1
npos <- length(pos) # number of mutations selected
amm <- PI_TSM$V2

# Structure of PI_TSM: each row is a relevant position in the chain of
# HIV protease that has provably shown mutations. In the second column
# each cell is a list of mutation at a specific position in the Protease chain
# specified in the first column.

# The following is the code to create the full notations (like M46I etc ...)
# for the mutations in table PI_TSM (whose structure is not directly
# M46I, I309 etc...)

# wt is the wild type sequence (not-mutated of HIV-1) protease
# it has been taken from https://www.uniprot.org/uniprotkb/O90777/entry
wt <- unlist(strsplit("PQVTLWQRPIVTIKIGGQLKEALLDTGADDTVLEEMSLPGKWKPKMIGGIGGFIKVRQYDQVSIEICGHKAIGTVLIGPTPVNIIGRNLLTQLGCTLNF", split = ""))
names(wt) <- as.character(1:99)

amm <- strsplit(amm, " ")
amm_unlist <- unlist(amm)
namm <- length(amm_unlist)
wtmut <- list()
k <- 1
for (i in 1:npos) {
  position <- pos[i]
  amm_position <- unlist(amm[[i]])
  nmu <- length(amm_position)
  for (j in 1:nmu) {
    wtmut[[k]] <- paste(c(wt[position], position, amm_position[j]), collapse = "")
    k <- k + 1
  }
}
length(wtmut)

# wtmut contains the full list of mutations in TSM list for HIV-1 protease
# the list is composed by mutation with the notation such as M46I

data <- PI_data
# adjusting the column names
colnames(data) <- data[1, ] # naming the columns with the first row
data <- data[-1, ] # removing the first column of id's
n <- dim(data)[1] # the number of initial rows
# - are missing value that are structural
# the NA in the response drug resistance are important to take into account

summary(is.na(data)) # assessing the presence of NA in the drug resistance measurements
# We choose the drug Nelfinavir (among the Protease inhibitors)
# because it has the least amount of missing values.

# These are the indices of missing values in the response of drug resistance
ind_na_nfv <- which(is.na(data$NFV) == TRUE) # isolating the inidices
length(ind_na_nfv)

# The response is computed as a log-fold change, namely is the log
# base = 10 of the ratio between
# the concentration of drug to inhibit 50% of the replication of the virus
# when the HIV-1 Protease is mutated
# divided by the concentration of drug needed to reduce by 50% the replication of
# the virus when HIV 1 protease is wild type
y <- as.numeric(data$NFV[-ind_na_nfv])
# if y=1 the patient virus is resistant as wild type
# if y>1 patient virus is more resistant than wild type
# if y<1 patient virus is less resistant than wild type
# We also remove from the predictors the rows that have a missing value in the response
# of log-fold change of Nelfinavir
data <- data[-ind_na_nfv, ]
# We remove all responses (in this way dataX is the matrix of predictors)
dataX <- data[, -c(1:10)]
# structure of dataX: we have a matrix approximately 844x99
# where each columns represent a position. in this way we have for a fixed
# position/column all ammino acids mutated at that specific position
# obviously we could have several different mutations at the same position.
# We would like to have specific mutations like M46I instead of positions like P70
# as features. In dataX the value in cell (i,j) is either - if the the HIV-1
# Protease sample of the patient i doesn't show a mutation in position j
# or it is the first letter of the ammino acid that represents the mutation if the mut. is actually present.
# we would like to have only "-" or letters in dataX; We don't want any other symbol.
# therefore we replace "." with "-".
mut_list <- list()
n_X <- dim(dataX)[1]
p_X <- dim(dataX)[2]
anyNA(dataX)
for (j in 1:p_X) {
  for (i in 1:n_X) {
    if (dataX[i, j] == ".") {
      dataX[i, j] <- "-"
    }
  }
}
# In the following for cycles we create the list of all mutations in the standard notation
# that can be found in dataX. So mut_list is a list of characters such as
# M46I, I10L and so on.
k <- 1
for (j in 1:p_X) {
  for (i in 1:n_X) {
    if (dataX[i, j] != "-") {
      mut_list[k] <- paste(c(wt[j], j, dataX[i, j]), collapse = "")
      k <- k + 1
    }
  }
}
# Obviously we will have several mutations that will be exactly equal.
# therefore we take the vector of unique mutations umut.
numut <- length(unique(mut_list))
umut <- unique(mut_list)
# with the following code  we want to create a list of lists.
# more precisely we want a list for each HIV-1 sample including all mutations appearing for that
# sample. so we will have a list of n_X lists, where n_X list is the
# number of filtered patients. each list in the big list will have a different number
# of mutations since different samples have different mutations

mut_grouped_by_sample <- list() # list of list of mutations for each sample of HIV1 protease
for (i in 1:n_X) {
  mut_list_i <- list()
  k <- 1
  for (j in 1:p_X) {
    if (dataX[i, j] != "-") {
      mut_list_i[k] <- paste(c(wt[j], j, dataX[i, j]), collapse = "")
      k <- k + 1
    }
  }
  mut_grouped_by_sample[[i]] <- mut_list_i
}
str(mut_grouped_by_sample)
# now we are able to create a matrix having on the columns unique mutations in standard notations
# and on the rows the HIV-1 Protease samples. The entry in cell (i,j)
# will be either 0 or 1 depending whether sample i-th of HIV-1 Protease contains the mutation
# that labels column j-th or not.

X <- matrix(data = 0, nrow = n_X, ncol = length(umut))
colnames(X) <- umut
for (i in 1:n_X) {
  for (j in 1:length(umut)) {
    X[i, j] <- as.numeric(ifelse(umut[j] %in% unlist(mut_grouped_by_sample[[i]]), 1, 0))
  }
}

# We remove mutations that appears in less than 3 samples (namely we remove columns)
ind_rm <- which(apply(X, 2, sum) < 3)
length(ind_rm)
X <- X[, -ind_rm]

# we also remove duplicates in order to have a full rank matrix
X <- X[, which(!duplicated(t(X)) == T)]

dim(X)
# we standardize the response
# y <- (y - mean(y)) / sd(y)
# we now apply the six methods
y_X <- cbind(y, X)
write.csv(y_X, "data_hiv")
