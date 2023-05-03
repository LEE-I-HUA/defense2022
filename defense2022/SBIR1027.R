pacman::p_load(dplyr, stringr)

X = read.csv("SBIR_STTR1112.csv", stringsAsFactors = F)

#####取代EMI字(手動找文章)
#case1
str1="EMI proposes to develop a novel digital self-calibration"
a = str_replace_all(X[str_detect(X$Abstract,str1),'Abstract'], "\\bEMI\\b", "Enertia Microsystems")
X[grep(str1,X$Abstract),'Abstract'] = a
#case2
str2 = "The EMI is home to the largest independent rock excavation and drilling facility in the western hemisphere"
b = str_replace_all(X[str_detect(X$Abstract,str2),'Abstract'], "\\(EMI\\)", "")
b = str_replace_all(b, "\\bEMI\\b", "Earth Mechanics Institute")
X[grep(str2,X$Abstract),'Abstract'] = b

#####取代字
##X:原始資料集  col:要替換的資料欄位  full:要尋找的字的正規表達式  abb:要替換的字  after:替換後的字
replaceword = function(X,col,full,abb,after) {
  index = grep(full,X[,col]) #1977 3453 4177
  for(i in index){
    X[i,col]=str_replace_all(X[i,col], abb, after)
  }
  return(X)
}

#####取代BDA字
#把battle damage assessment 的BDA 取代成BDAssessment
X = replaceword(X,'Abstract',"(B|b)attle (D|d)amage (A|a)ssessment","\\bBDA\\b","BDAssessment")  #491 1125 1874 3480 3993 4482 5074 6464 
#X[6464,'Abstract']

#####取代FST字
X = replaceword(X,'Abstract',"Fleet Synthetic Training","\\bFST\\b","FleetSyntheticTraining") #1977 3453 4177

#####多一個加公司名稱的欄位Abstract_com
X$Abstract_com = sapply(1:nrow(X), function(i) 
  str_c(X$Abstract[i],X$Company[i],sep = "."))


#####取代MRL字(此部份需要將公司名稱加到Abstract後(需有Abstract_com)才可進行)
X = replaceword(X,'Abstract_com',"\\bMATERIALS RESOURCES LLC\\b","\\bMRL\\b","MRLLC")#1155 1864 2015 2444 3082 3620 4300 6453 6755 6768 6776 6952

write.csv(X, "SBIR_STTR1125.csv", row.names=F, quote=T)

####字典找重複縮寫字
E = read.csv("Entity_SBIR1130.csv", stringsAsFactors = F)

extr = function(z, ignore=TRUE, n=1) {
  str_extract_all(E$alias,regex(z, ignore_case=ignore)) %>% 
    unlist %>% table %>% {.[.>=n]} %>% sort }

a = extr("\\([A-Z]{3,5}\\)",n=3)%>%as.data.frame()
names(a)[1] <- "entity"
a$entity = str_replace_all(a$entity,"[(.*)]","")
a$alias  <-  sapply(1:nrow(a), function(i) 
  paste0('\\b',a[i,1],'\\b'))
b = sapply(1:nrow(a), function(i) 
  extr(a[i,3])) 

extr("\\bISI\\b")

#ISI Intellisense Systems, Inc
#CFD Research Corporation /Computational Fluid Dynamics( \(CFD\)|)
#SMC Space Missile Command (SMC)/Space and Missile Center(SMC)
#Modular Open System Architecture (MOSA)/Modular Open Systems Approach( \(MOSA\)|)
#LOS
#RAM 


####要放到字典裡面的公司名稱
# 出現次數>2
com <- X %>% group_by(Company)%>% count()
com2 = com %>% filter(n>2) %>% as.data.frame()
end = function(z) {
  a = str_extract_all(z, pattern = ".")
  if(a[[1]][length(a[[1]])]=='.'){
    z=substr(z,1,nchar(z)-1)
  }
  return(z)
}

com2[,1]  <-  sapply(1:nrow(com2), function(i) 
  end(com2[i,1])) 

write.csv(com2, "SBIR_keyword.csv", row.names=F, quote=T)

#把公司放到字典裡
#MRL MATERIALS RESOURCES LLC 手動把MRL拿掉
E = read.csv("Entity_SBIR1105.csv",stringsAsFactors=F) #473
E2=data.frame()

E2  <-  sapply(1:nrow(com2), function(i) 
  paste0('\\b',com2[i,1],'\\b')) 
E2=E2%>% as_data_frame()
names(E2)[1] <- "alias"

E2$entity <- com2$Company
E2$tooltip <- com2$Company
E2$class <- '補助案公司'
E2$chinese <- com2$Company
E2$freq <- com2$n
E2$ignore <- as.logical(FALSE)

E3 = rbind(E,E2)

write.csv(E3, "SBIR_keyword.csv", row.names=F, quote=T)

####處理url
X1111 = read.csv("sbirl1112.csv", stringsAsFactors = F)
url = read.csv("sbir_url1112.csv", stringsAsFactors = F)
names(url)[2] <- "Contract"

X1111$url[which(duplicated(X1111$url))]

#4242 5173
X1111[4242,]
X1111[5173,]

X1 <- X1111[-c(4242, 5173), ]
X1 <- X1[,-c(40,42)] 
names(X1)[2] <- "Award Title"
write.csv(X1, "SBIR_STTR1112.csv", row.names=F, quote=T)
