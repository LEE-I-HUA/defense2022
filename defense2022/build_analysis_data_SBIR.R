pacman::p_load(dplyr, stringr)

# 讀取已經完成udpipe處理的文章
# 你們的應該是load("XX.Rdata")
#annotation = readRDS("./space_annotation_replace.rds")

# 讀取原始檔
X = read.csv("SBIR_STTR1125.csv", stringsAsFactors = F)

#更改欄位名稱
names(X)[41] <- "artContent"  #Abstract_com改為artContent
names(X)[8] <- "artTitle"     #Contracr改為artTitle
names(X)[10] <- "artDate"     #Contract.End.Date改為artDate

#S = annotation %>% group_by(doc_id, paragraph_id, sentence_id) %>% summarise(tx = sentence[1]) %>% ungroup()
S = read.csv("token_SBIR_STTR_1125.csv", stringsAsFactors = F)
S <- subset(S, select = c(doc_id,sen_id,sent))
S =S[!duplicated(S[,c('doc_id','sen_id')]),]

#關鍵字字典
E = read.csv("Entity_SBIR1125.csv",stringsAsFactors=F) # 1114
#E <- E[-c(36),]
# 檢查有無重複
E$entity[which(duplicated(E$entity))]

start_time <- Sys.time()
#mx = sapply(1:nrow(E), function(i) str_detect(
#  S$tx, regex(E$alias[i], ignore_case=T) ))

mx = sapply(1:nrow(E), function(i) str_detect(
  S$sent, regex(E$alias[i], ignore_case=T) ))

Sys.time() - start_time # 2.193616 mins

extr = function(z, ignore=TRUE, n=1) {
  str_extract_all(X$artContent,regex(z, ignore_case=ignore)) %>% 
    unlist %>% table %>% {.[.>=n]} %>% sort }

dim(mx) # 52178  1113
colnames(mx)=E$entity
E$freq = colSums(mx); range(E$freq) # 1 2525
i = order(E$class, -colSums(mx))
E = E[i,]; mx = mx[,i]
# E[E$freq < 10,] %>% View
# E = subset(E, freq >=10)
write.csv(E, "Entity_SBIR1125.csv", row.names=F, quote=T)

# sentence term matrix
xSent = mx

# paragraph term matrix
#xPara = t(sapply(
#  split(data.frame(mx), group_indices(S, doc_id, paragraph_id)),
#  colSums)) > 0

# document term matrix
xDocu = t(sapply(split(data.frame(mx), S$doc_id), colSums)) > 0

XX = list(Sent=xSent, Docu=xDocu)#, Para=xPara
CO = list(Sent = t(xSent) %*% xSent,
#          Para = t(xPara) %*% xPara,
          Docu = t(xDocu) %*% xDocu )
CR = list(Sent = cor(xSent),
#          Para = cor(xPara),
          Docu = cor(xDocu))

save(X, E, S, XX, CO, CR, file="SBIR_analysis_data.RData",compress=T)
