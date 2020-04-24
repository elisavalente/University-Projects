library(plyr)
library(corrplot)
library(ggplot2)
library(RColorBrewer)
library(MASS)
library(ISLR)
library(FNN)
library(DAAG)
library(plotrix)
library(car)
library(leaps)
library(glmnet)
#Cores
col4 <- colorRampPalette(c("#7F0000", "red", "#FF7F00", "yellow", "#7FFF7F","cyan", "#007FFF", "blue", "#00007F"))
col24 <- colorRampPalette(brewer.pal(11,"RdYlGn"))

#Importar dataset
dataset <- read.csv("student-por.csv")
dataset_numerico=dataset


# Binarização
dataset_numerico$school <-as.numeric(dataset_numerico$school)
dataset_numerico$sex <- as.numeric(dataset_numerico$sex)
dataset_numerico$address <- as.numeric(dataset_numerico$address)
dataset_numerico$famsize <- as.numeric(dataset_numerico$famsize)
dataset_numerico$Pstatus <- as.numeric(dataset_numerico$Pstatus)
dataset_numerico$schoolsup <- as.numeric(dataset_numerico$schoolsup)
dataset_numerico$famsup <- as.numeric(dataset_numerico$famsup)
dataset_numerico$paid <- as.numeric(dataset_numerico$paid)
dataset_numerico$activities <- as.numeric(dataset_numerico$activities)
dataset_numerico$nursery <- as.numeric(dataset_numerico$nursery)
dataset_numerico$higher <- as.numeric(dataset_numerico$higher)
dataset_numerico$internet <- as.numeric(dataset_numerico$internet)
dataset_numerico$romantic <- as.numeric(dataset_numerico$romantic)
dataset_numerico$Mjob <-as.numeric(dataset_numerico$Mjob)
dataset_numerico$Fjob <-as.numeric(dataset_numerico$Fjob)
dataset_numerico$reason <-as.numeric(dataset_numerico$reason)
dataset_numerico$guardian <-as.numeric(dataset_numerico$guardian)

#Estudar correlação

dados_cor <- cor(dataset_numerico)
corrplot(dados_cor,tl.col = "black", method="square",order="hclus",tl.srt = 45,is.corr=FALSE,type = "lower",col = col24(100))

#Nomes das colunas
nome_colunas<-names(dataset)

summary(dataset)

#Passar categorias para os nomes
dataset$Medu <- revalue(as.character(dataset$Medu),c("0"="0-Nenhuma","1"="1-EscolaPrimária","2"="2-(5-9ano)","3"="3-EnsinoSecundário","4"="4-EnsinoSuperior"))
dataset$Fedu <- revalue(as.character(dataset$Fedu),c("0"="0-Nenhuma","1"="1-EscolaPrimária","2"="2-(5-9ano)","3"="3-EnsinoSecundário","4"="4-EnsinoSuperior"))
dataset$traveltime <- revalue(as.character(dataset$traveltime),c("1"="1-(<15min)","2"="2-(15-30min)","3"="3-(30-60min)","4"="4-(>60min)"))
dataset$studytime <-revalue(as.character(dataset$studytime),c("1"="1-(<2hr)","2"="2-(2-5hr)","3"="3-(5-10hr)","4"="4-(>10hr)"))
dataset$failures <-revalue(as.character(dataset$failures),c("0"="0","1"="1","2"="2","3"="3+"))
dataset$famrel<-revalue(as.character(dataset$famrel),c("1"="1-MuitoMá","2"="2-Má","3"="3-Média","4"="4-Boa","5"="5-Excelente"))
dataset$freetime <-revalue(as.character(dataset$freetime),c("1"="1-MuitoBaixo","2"="2-Baixo","3"="3-Médio","4"="4-Alto","5"="5-MuitoAlto"))
dataset$goout <- revalue(as.character(dataset$goout),c("1"="1-MuitoBaixa","2"="2-Baixa","3"="3-Média","4"="4-Alta","5"="5-MuitoAlta"))
dataset$Dalc <-revalue(as.character(dataset$Dalc),c("1"="1-MuitoBaixo","2"="2-Baixo","3"="3-Médio","4"="4-Alto","5"="5-MuitoAlto"))
dataset$Walc <-revalue(as.character(dataset$Walc),c("1"="1-MuitoBaixo","2"="2-Baixo","3"="3-Médio","4"="4-Alto","5"="5-MuitoAlto"))
dataset$health <- revalue(as.character(dataset$health),c("1"="1-MuitoMau","2"="2-Mau","3"="3-Médio","4"="4-Bom","5"="5-MuitoBom"))

##############Gráficos de visualização dos dados##################
#Boxplot
for(col in nome_colunas){
  pdf(paste("imagens/",col,".pdf"),width=20, height=8)
  par(mfrow=c(1,1))
  boxplot(dataset$G3~dataset[,col],main="BoxPlot",xlab=col,ylab="G3",col=brewer.pal(4, "Blues"))
  dev.off()
}

#Alcool à semana vs sexo vs G3
ggplot(dataset, aes(x = sex, y = G3, fill = sex)) + geom_boxplot() + facet_wrap(~Dalc, ncol = 5, labeller = "label_both") + ggtitle("Consumo de Álcool à semana") + theme(plot.title = element_text(hjust = 0.5))

#Alcool ao fim de semana vs sexo vs G3
ggplot(dataset, aes(x = sex, y = G3, fill = sex)) + geom_boxplot() + facet_wrap(~Walc, ncol = 5) + ggtitle("Consumo de Álcool ao fim-de-semana") + theme(plot.title = element_text(hjust = 0.5)) 

#Relação familiar com o consumo de alcool à semana
ggplot(dataset, aes(x = famsize, y = frequency(Dalc), fill = Dalc)) + geom_bar(stat = "identity", position="fill") + facet_wrap(~famrel, ncol = 5) + ggtitle("Relação familiar") + theme(plot.title = element_text(hjust = 0.5))

#Realação familiar com o consumo de alcool ao fds
ggplot(dataset, aes(x = famsize, y = frequency(Walc), fill = Walc)) + geom_bar(stat = "identity", position="fill") + facet_wrap(~famrel, ncol = 5)  + ggtitle("Relação familiar") + theme(plot.title = element_text(hjust = 0.5))

#Saídas à noite e consumo de alcool
ggplot(dataset, aes(x = goout, y = frequency(Walc), fill = Walc)) + geom_bar(stat = "identity", position="fill") +ggtitle("Saídas à noite e consumo de álcool")  + theme(plot.title = element_text(hjust = 0.5))

#FreeTime vs Romantic vs Stdytime
ggplot(dataset, aes(x = romantic, y = frequency(romantic)/length(romantic) , fill = studytime)) + geom_bar(stat = "identity") + facet_wrap(~freetime, ncol = 5) +ggtitle("Tempo livre")+theme(plot.title = element_text(hjust = 0.5))


#Percentagens dos diferentes consumos de alcool
par(mfrow=c(1,1))
pie3D(x = table(Dalc), col=brewer.pal(5, "Blues"), explode=0.1,theta = 1, labels = table(Dalc))
pie3D(x = table(Walc), col=brewer.pal(5, "Blues"), explode=0.1,theta = 1, labels = table(Walc))

#################### Modelos ################################
#Remover G1,G2,school,reason,freetime e health
dataset <- read.csv("student-por.csv")
dataset <-subset(dataset, select = -c(G1,G2,school,reason, freetime, health) )
attach(dataset)



#Regressão Linear - Todos os preditores
lm.fit=lm(G3~.,data=dataset)
vif(lm.fit)
summary(lm.fit)
par(mfrow=c(2,2))
plot(lm.fit)

#Verificar possíveis interações

# Sair à noite e beber álcool
lm.fit_interaction1=lm(G3~.+(Dalc*goout), data = dataset)
summary(lm.fit_interaction1)
plot(lm.fit_interaction1)

# Tempo de estudo e reprovações 
lm.fit_interaction2=lm(G3~.+(studytime*failures), data = dataset)
summary(lm.fit_interaction2)
plot(lm.fit_interaction2)

# Utilizar subconjuntos de preditores
regfit = regsubsets(G3~.,nvmax = 15,data = dataset) 
summary(regfit)

regfit.sum = summary(regfit)
which.min(regfit.sum$bic)    
which.max(regfit.sum$adjr2)
which.min(regfit.sum$cp)

par(mfrow=c(1,1))
plot(regfit,scale="adjr2")
plot(regfit,scale="Cp")
plot(regfit,scale="bic")


# Modelo com as 15 melhores variáveis
best15=names(coef(regfit,15))
model_15best = lm(G3~sex+age+address+Medu+Fjob+guardian+studytime+failures+schoolsup+activities+higher+internet+romantic+Dalc)
summary(model_15best)
par(mfrow=c(2,2))
plot(model_15best)


# Regressão Polinomial

polyfitError = matrix(, nrow = 3, ncol = 4)
for (i in 1:3){
    for(k in 1:4){
      polyfit = lm (G3~poly( studytime,i) + poly( failures,3) + poly(Dalc,k) + higher ,data=dataset )
      polyfitSum = summary.lm(polyfit)
      polyfitError[i,k] = polyfitSum$adj.r.squared
    }
  }
polyfitError


# Ridge Regression
x=model.matrix(G3~.,dataset)[,-1]
y=na.omit(dataset$G3)
grid=10^seq(10, -2, length=100)

ridge.mod=glmnet(x,y,alpha=0,lambda=grid)

