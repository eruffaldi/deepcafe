

what = [3,4;2,4;3,2;3,3;2,4;1,1;1,3];

stats = multiclassinfo(what(:,1),what(:,2));
stats.confusionMat

