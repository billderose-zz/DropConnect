function [er, bad] = nntest(nn, x, y)
    labels = nnpredict(nn, x);
    bad = find(labels ~= y);    
    er = numel(bad) / size(x, 1);
end