FROM 42n4/rstudio-ml:latest
RUN apt-get update
RUN installGithub.r 42n4/dmr.data 42n4/dmr.util 42n4/dmr.claseval 42n4/dmr.stats 42n4/dmr.trans 42n4/dmr.linreg 42n4/dmr.regeval 42n4/dmr.dissim 42n4/dmr.dectree 42n4/dmr.linclas 42n4/dmr.disc 42n4/dmr.kcenters 42n4/dmr.cluseval
RUN apt-get autoremove -y \
    && apt-get install -y --no-install-recommends libxml2-dev gfortran default-jdk postgresql-9.5 postgresql-client-9.5 libmariadb-client-lgpl-dev postgresql-server-dev-9.5 libpcre3-dev lzma libbz2-dev liblzma-dev libicu-dev libmagick++-dev libpng-dev freeglut3 freeglut3-dev libglew-dev libglu1-mesa libglu1-mesa-dev libgl1-mesa-glx libgl1-mesa-dev texlive-luatex texlive-latex-base tk-dev libopencv-dev python-opencv cargo python-numpy python-scipy python-matplotlib  python3-numpy python3-scipy python3-matplotlib swi-prolog tk8.5 libccfits-dev
RUN R CMD javareconf
RUN install2.r -n 8  --deps TRUE \
    codetools   \
    devtools    \
    remotes     \
    e1071       \
    rpart       \
    rpart.plot  \
    lattice     \
    ipred       \
    cluster     \
    quadprog    \
    kernlab     \
    Matrix	\
    randomForest\
    reshape	\
    sqldf	\
    dplyr	\
    party	\
    partykit	\
    rgl	 	\
    scatterplot3d\
    fpc		\
    pvclust	\
    dendextend	\
    nFactors	\
    FactoMineR	\
    RRF	    	\
    mclust	\
    gbm		\
    mda		\
    klaR	\
    tseries	\
    forecast	\
    sets	\
    arules	\
    doMC	\
    ranger	\
    TTR		\
    clusterGeneration\
    nnet 	\
    neuralnet 	\
    RSNNS 	\
    rockchalk 	\
    h2o		\
    AppliedPredictiveModeling \ 
    fastAdaboost\
    rotationForest\
    rpartScore 	\
    adabag	\
    C50		\
    RWeka	\
    xgboost	\
    pROC	\
    DMwR	\
    ROSE	\
    Boruta	\
    Metrics 	\
    hydroGOF	\
    bartMachine \
    ada 	\
    bst		\
    deepboost	\
    evtree	\
    nodeHarvest	\
    obliqueRF   \
    pbapply     \
    readbitmap  \
    Cairo       \
    downloader 	\
    urca 	\
    timeSeries 	\
    fBasics	\
    rgl		\
    knitr 	\
    filehash	\
    tidyverse 	\
    cowplot 	\
    lsmeans	\
    mboost	\
    iterators 	\
    foreach	\
    dbplyr 	
RUN installGithub.r 42n4/dmr.regtree 42n4/dmr.attrsel 42n4/dmr.ensemble 42n4/dmr.kernel 42n4/dmr.bayes 42n4/dmr.hierclus 42n4/dmr.miscost 42n4/dmr.rpartutil 42n4/tkrplot 42n4/fUnitRoots yihui/tikzDevice dahtah/imager rstudio/reticulate dustinfife/fifer 42n4/rolog RevolutionAnalytics/dplyrXdf
RUN Rscript -e "devtools::install_github('topepo/caret/pkg/caret')" 
RUN install2.r -n 4  --deps TRUE ade4 FITSio GA phonTools snow
RUN installGithub.r 42n4/rmongodb stefano-meschiari/latex2exp eddelbuettel/rinside
RUN installGithub.r pwasiewi/quasarcl

#failed install inside docker
#RUN install2.r -n 4 --deps TRUE rattle odbc
#RUN installGithub.r floybix/playwith
#RUN installGithub.r swarm-lab/videoplayR
#RUN apt-get install -y python-rpy
#RUN installGithub.r gaborcsardi/pkgconfig
#RUN installGithub.r igraph/rigraph
#RUN install2.r -n 4 -r http://bioconductor.org/packages/3.6/bioc --deps TRUE \ 
#    KEGGgraph \
#    biomaRt

RUN rm -rf /tmp/* \
&& apt-get autoremove -y \
&& apt-get autoclean -y \
&& rm -rf /var/lib/apt/lists/*
#https://docs.microsoft.com/en-us/machine-learning-server/r/sample-microsoftml
#https://docs.microsoft.com/en-us/machine-learning-server/r/sample-built-in-data
