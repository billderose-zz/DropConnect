%creditMat = csvread('imputeddata.csv',1);
input = creditMat(:, 2:11);
desired = creditMat(:, 1)';
desired = desired';
train_x = input(1:10000, :);
train_y = desired(1:10000, :);

test_x = input(16500:17000, :);
test_y = desired(16500:17000, :);

[~, nIn] = size(train_x);
[~, nOut] = size(train_y);
nn = nnsetup([nIn, 40, nOut]);
nn.dropConnectFraction = 0.0;
%nn.momentum = .8;
%nn.learningRate = 1;
%nn.scaling_learningRate = .999;
opts.numepochs = 1500;
opts.batchsize = 500;
opts.plot = 0;
opts.validation = 1;
[nn, L] = nntrain(nn, train_x, train_y, opts, test_x, test_y);