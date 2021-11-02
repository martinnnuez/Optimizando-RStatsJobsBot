# Code

# ---------------------------------
## Read data, clean characters not ascii and create a corpus:
# ---------------------------------
library(tidytext)
library(tm)
library(qdap)
library(dplyr)
library(stringr)

tweets <- readr::read_csv("rstatsjobs.csv")

# Clean characteres that are not ascii
tweets_text <- textclean:::replace_non_ascii(tweets$text)

# Make a vector source: tweet_source
tweet_source <- VectorSource(tweets_text)

# Count number of emails
emails <- tweets_text %>% str_count(pattern = "[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+")
webpage <- tweets_text %>% str_count(pattern = "(f|ht)tp\\S+\\s*")

# Make a volatile corpus: coffee_corpus
corpus <- VCorpus(tweet_source)

# ---------------------------------
## Clean corpus function:
# ---------------------------------

clean_corpus <- function(corpus){
  # Remove brackets
  RemoveBrackets <- function(x) {
    gsub("[()]", "", x)
  }
  corpus <- tm_map(corpus,content_transformer(RemoveBrackets)) # remove brackets

  # Remove emails, web pages, simbols that are problematic and stop words
  RemoveEmail <- function(x) {
    require(stringr)
    str_replace_all(x,"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+", "")
  }
  corpus <- tm_map(corpus,content_transformer(RemoveEmail)) # removing email ids

  # Remove web pages
  RemoveWebpage <- function(x) {
    require(stringr)
    str_replace_all(x,"(f|ht)tp\\S+\\s*", "")
  }
  corpus <- tm_map(corpus,content_transformer(RemoveWebpage)) # removing email ids

  # Remove words with numers in the middle and --
  RemoveWordNum <- function(x) {
    paste(unlist(str_extract_all(x, "(\\b[^\\s\\d]+\\b)")), collapse = " ")
  }
  corpus <- tm_map(corpus,content_transformer(RemoveWordNum))

  # converts corpus to a Plain Text Document
  corpus <- tm_map(corpus, content_transformer(PlainTextDocument))

  # First step is to transform all text to lower case:
  corpus <- tm_map(corpus, content_transformer(tolower))

  #Replace symbols
  #Replace common symbols with their word equivalents (e.g. “$” becomes “dollar”)
  corpus <- tm_map(corpus, content_transformer(qdap:::replace_symbol))

  # Remove stop words
  RemoveStopwords <- function(x) {
    removeWords(x, c(stopwords("english"),"’s","’re","’t","’ve"))
  }
  corpus <- tm_map(corpus,content_transformer(RemoveStopwords))

  #Remove punctuation
  corpus <- tm_map(corpus, content_transformer(removePunctuation))

  # Remove numbers
  corpus <- tm_map(corpus, content_transformer(removeNumbers))

  # Remove whitespace
  corpus <- tm_map(corpus, content_transformer(stripWhitespace))

  # Stem document
  corpus <- tm_map(corpus, content_transformer(stemDocument))
  return(corpus)
}

corpus <- clean_corpus(corpus)

###################################################
# 1) Ngrams and sparcity combined:
###################################################
# Ngrams
library(RWeka)
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 8))

# Make a bigram TDM
# Document term matrix with n grams
DTM <- DocumentTermMatrix(
  corpus,
  control = list(tokenize = tokenizer)
)

sp <- seq(0.990,0.999,0.001)


resultados <- lapply(sp, function(i){
  # Remove sparse terms
  sparse_DTM <- removeSparseTerms(DTM, i)

  # Convert to data frame
  tweetsSparse <- as.data.frame(as.matrix(sparse_DTM))

  colnames(tweetsSparse) <- make.names(colnames(tweetsSparse))

  # ---------------------------------
  ## New variables addition:
  # ---------------------------------
  # Add responses and meaninngfull variables
  tweetsSparse$is_job_post <- tweets$is_job_post
  tweetsSparse$emails_count <- emails
  tweetsSparse$webpage_count <- webpage
  tweetsSparse$screen_name <- tweets$screen_name
  tweetsSparse$source <- tweets$source

  # ---------------------------------
  ## Separate train and test:
  # ---------------------------------
  # Stratified splitting:
  set.seed(1996)

  train.index <- caret:::createDataPartition(tweetsSparse$is_job_post, p = .9, list = FALSE)
  train <- tweetsSparse[ train.index,]
  test  <- tweetsSparse[-train.index,]

  # ---------------------------------
  ## Grafico sparcity y una metrica:
  # ---------------------------------
  library(h2o)
  h2o.init()
  train_h2o <- as.h2o(x = train, destination_frame = "train_h2o")
  test_h2o <- as.h2o(x = test, destination_frame = "test_h2o")

  # Convert last varibales to factor and repeat
  train_h2o$is_job_post <- as.factor(train_h2o$is_job_post)
  train_h2o$screen_name <- as.factor(train_h2o$screen_name)
  train_h2o$source <- as.factor(train_h2o$source)
  test_h2o$screen_name <- as.factor(test_h2o$screen_name)
  test_h2o$source <- as.factor(test_h2o$source)
  test_h2o$is_job_post <- as.factor(test_h2o$is_job_post)

  aml <- h2o.automl(
    y = "is_job_post" ,
    training_frame = train_h2o,
    validation_frame = test_h2o,
    seed = 1996,
    nfolds = 10,
    max_models = 10,
    balance_classes = TRUE,
    include_algos = c("GBM"),
    stopping_metric = c("mean_per_class_error")
  )

  results <- as.data.frame(aml@leaderboard)
  lb <- h2o.get_leaderboard(object = aml, extra_columns = 'ALL')
  model_ids <- as.vector(aml@leaderboard$model_id)
  model1 <- h2o.getModel(model_ids[1])
  h2o.saveModel(object = model1, path = getwd(), force = TRUE)
  model2 <- h2o.getModel(model_ids[2])
  h2o.saveModel(object = model2, path = getwd(), force = TRUE)

})
