pacman::p_load(dplyr,stringr)

# 讀取原始資料
sbir = read.csv('./data/SBIR.csv',stringsAsFactors = F)
# 對sbir新增doc_id
sbir$doc_id = 1:nrow(sbir)

# 讀取字典檔
dict = read.csv('./SBIR關鍵字 - 工作表1.csv',stringsAsFactors = F)
dict$entity = str_replace_all(dict$entity,'\\.|,','')
replaceKW = dict[str_detect(dict$entity,'-'),] %>% select(entity)
replaceKW$cleanKW = str_replace_all(replaceKW$entity,'-',' ')

dict$entity = str_replace_all(dict$entity,'-',' ')
dict$entity = str_replace_all(dict$entity,'5G','fiveG')

# 同一個詞類不同關鍵字的比較
table(dict$class)

doc_dtm = sapply(1:nrow(dict), function(i) str_detect(
  sbir$Abstract, regex(dict$alias[i], ignore_case=T) ))
colnames(doc_dtm)=dict$entity

doc_freq_df = data.frame(doc_dtm) 
doc_freq_df$doc_id = 1:nrow(doc_freq_df)
doc_freq_df = doc_freq_df %>% melt(id = 'doc_id')
names(doc_freq_df)<-c("doc_id","entity","metion")
doc_freq_df = doc_freq_df %>% group_by(entity) %>% mutate(doc_freq = sum(metion)) %>% ungroup()

freq_mx = sapply(1:nrow(dict), function(i) str_count(
  sbir$Abstract, regex(dict$alias[i], ignore_case=T) ))

colnames(freq_mx)=dict$entity
dict$word_freq = colSums(freq_mx)

word_freq_df = data.frame(freq_mx) 
word_freq_df$doc_id = 1:nrow(word_freq_df)
word_freq_df = word_freq_df %>% melt(id = 'doc_id')
names(word_freq_df)<-c("doc_id","entity","word_freq")


total_freq = merge(doc_freq_df,word_freq_df)
total_freq$entity = str_replace_all(total_freq$entity,'\\.',' ')
total_freq$entity = str_replace_all(total_freq$entity,'fiveG','5G')

for(i in 1:nrow(replaceKW)){
  total_freq[total_freq$entity == replaceKW[i,2],'entity'] = replaceKW[i,1]
}

kwdocDF = merge(sbir[,c('doc_id','Company','Branch','Award.Year','Award.Amount')], total_freq)
kwdocDF[!(kwdocDF$entity %in% dict$entity), 'entity'] %>% unique()
dict = read.csv('./SBIR關鍵字 - 工作表1.csv',stringsAsFactors = F)
dict$entity = str_replace_all(dict$entity,'\\.|,','')
dict$doc_freq = colSums(doc_dtm);
dict = dict %>% arrange(desc(doc_freq))
kwdocDF = merge(kwdocDF, dict[,c(2,4)])

# spacy斷詞後的結果
sbir_token = read.csv('./sbir_token.csv',stringsAsFactors = F)
tf_idf <- document_term_frequencies(sbir_token[, c("doc_id", "token")])
tf_idf <- document_term_frequencies_statistics(tf_idf)

stopwords <- stopwords::stopwords("en", source = "smart")
tf_idf <- tf_idf %>%
  filter(nchar(term) >1) %>% 
  filter(!(tolower(term) %in% stopwords))

# 每個字平均tf-idf
word_tfidf <-tf_idf%>% 
  group_by(term) %>% 
  summarise(tf_idf = mean(tf_idf),freq = sum(freq))

# python處理完tag後的結果
sbirTags = read.csv('./SBIRtags.csv',stringsAsFactors = F)


save(dict, kwdocDF, sbir,word_tfidf, sbirTags, file = 'SBIRdata.rdata')